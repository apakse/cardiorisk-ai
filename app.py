import streamlit as st
import pandas as pd
import numpy as np
import os, json, pickle, hashlib
from datetime import datetime
from pathlib import Path

# ── sklearn ────────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
import plotly.express as px
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data");  DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True)

DATA_FILE   = DATA_DIR  / "cardiac_data.csv"
MODEL_FILE  = MODEL_DIR / "model.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
META_FILE   = MODEL_DIR / "meta.json"

# ── Expected feature columns (canonical) ──────────────────────────────────
FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp",
    "cholesterol", "fasting_blood_sugar", "resting_ecg",
    "max_heart_rate", "exercise_angina", "st_depression",
    "st_slope", "num_vessels", "thalassemia"
]
TARGET = "cardiac_arrest"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: #f8fafc;}
.risk-card {
    padding: 28px 32px; border-radius: 16px; text-align: center;
    margin: 16px 0; box-shadow: 0 4px 24px rgba(0,0,0,.10);
}
.risk-low  {background: linear-gradient(135deg,#d4edda,#c3e6cb); border-left: 6px solid #28a745;}
.risk-med  {background: linear-gradient(135deg,#fff3cd,#ffeeba); border-left: 6px solid #ffc107;}
.risk-high {background: linear-gradient(135deg,#f8d7da,#f5c6cb); border-left: 6px solid #dc3545;}
.risk-pct  {font-size: 64px; font-weight: 800; line-height: 1.1;}
.risk-label{font-size: 22px; font-weight: 600; margin-top: 8px;}
.metric-box{background:#fff; border-radius:12px; padding:16px 20px;
            box-shadow:0 2px 10px rgba(0,0,0,.07); text-align:center;}
.metric-val{font-size:36px; font-weight:700; color:#1a3a5c;}
.metric-lbl{font-size:13px; color:#6c757d; margin-top:4px;}
.sidebar-logo{font-size:28px; font-weight:800; color:#dc3545; letter-spacing:-1px;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame | None:
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return None

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False)

def load_model():
    if MODEL_FILE.exists() and SCALER_FILE.exists():
        with open(MODEL_FILE, "rb") as f:  model  = pickle.load(f)
        with open(SCALER_FILE, "rb") as f: scaler = pickle.load(f)
        return model, scaler
    return None, None

def save_model(model, scaler, meta: dict):
    with open(MODEL_FILE,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_FILE, "wb") as f: pickle.dump(scaler, f)
    with open(META_FILE,   "w")  as f: json.dump(meta, f, indent=2)

def load_meta() -> dict:
    if META_FILE.exists():
        return json.load(open(META_FILE))
    return {}

def train_model(df: pd.DataFrame):
    """Train a calibrated Gradient Boosting model and return model, scaler, metrics."""
    df = df.copy().dropna(subset=FEATURES + [TARGET])

    # keep only known columns
    available = [c for c in FEATURES if c in df.columns]
    X = df[available].copy()
    y = df[TARGET].astype(int)

    # Simple imputation for missing feature cols
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0

    X = X[FEATURES]  # enforce order

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )

    base = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    )
    # Calibrate so probabilities are reliable
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X_s, y, cv=5, scoring="roc_auc").mean()

    meta = {
        "trained_at":   datetime.now().isoformat(),
        "n_samples":    int(len(df)),
        "n_positive":   int(y.sum()),
        "auc_test":     round(float(auc), 4),
        "acc_test":     round(float(acc), 4),
        "auc_cv":       round(float(cv),  4),
        "features":     FEATURES,
        "checksum":     hashlib.md5(df.to_csv(index=False).encode()).hexdigest(),
        "y_test":       y_test.tolist(),
        "y_prob":       y_prob.tolist(),
        "y_pred":       y_pred.tolist(),
        "cm":           confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, scaler, meta

def predict_risk(model, scaler, values: dict) -> float:
    row = pd.DataFrame([{f: values.get(f, 0) for f in FEATURES}])
    row_s = scaler.transform(row[FEATURES])
    prob  = model.predict_proba(row_s)[0][1]
    return float(prob)

def risk_card(pct: float):
    if pct < 0.30:
        cls, label, emoji = "risk-low",  "LOW RISK",    "💚"
    elif pct < 0.60:
        cls, label, emoji = "risk-med",  "MODERATE RISK","🟡"
    else:
        cls, label, emoji = "risk-high", "HIGH RISK",   "🔴"

    st.markdown(f"""
    <div class="risk-card {cls}">
        <div class="risk-pct">{emoji} {pct*100:.1f}%</div>
        <div class="risk-label">{label}</div>
        <p style="margin-top:10px;color:#555;font-size:14px;">
            Probability of cardiac arrest based on the provided characteristics
        </p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">❤️ CardioRisk AI</div>', unsafe_allow_html=True)
    st.caption("Health Informatics — Cardiac Arrest Risk Predictor")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔍 Risk Prediction", "📂 Upload Patient Data", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()
    meta = load_meta()
    if meta:
        st.markdown("**Model Status**")
        st.success(f"✅ Model trained")
        st.caption(f"Samples: **{meta.get('n_samples', '?')}**")
        st.caption(f"AUC (CV): **{meta.get('auc_cv', '?')}**")
        st.caption(f"Last trained: {meta.get('trained_at','?')[:16]}")
    else:
        st.warning("⚠️ No model yet\nUpload data first")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Risk Prediction
# ═══════════════════════════════════════════════════════════════════════════

if page == "🔍 Risk Prediction":
    st.title("🔍 Cardiac Arrest Risk Prediction")
    st.markdown("Enter the patient's clinical characteristics below to calculate their probability of cardiac arrest.")

    model, scaler = load_model()
    if model is None:
        st.error("No trained model found. Please upload patient data first to train the model.")
        st.stop()

    with st.form("predict_form"):
        st.subheader("Patient Demographics")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 18, 100, 55)
        sex = c2.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        fbs = c3.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                            format_func=lambda x: "No" if x == 0 else "Yes")

        st.subheader("Cardiac Characteristics")
        c4, c5, c6 = st.columns(3)
        cp = c4.selectbox("Chest Pain Type", [0,1,2,3],
            format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                    2:"Non-Anginal Pain",3:"Asymptomatic"}[x])
        rbp  = c5.number_input("Resting Blood Pressure (mmHg)", 80, 220, 130)
        chol = c6.number_input("Cholesterol (mg/dL)", 100, 600, 240)

        c7, c8, c9 = st.columns(3)
        ecg  = c7.selectbox("Resting ECG", [0,1,2],
            format_func=lambda x: {0:"Normal",1:"ST-T Wave Abnormality",
                                    2:"Left Ventricular Hypertrophy"}[x])
        mhr  = c8.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = c9.selectbox("Exercise Induced Angina", [0,1],
                              format_func=lambda x: "No" if x==0 else "Yes")

        st.subheader("ST Segment & Vessels")
        c10, c11, c12 = st.columns(3)
        std   = c10.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = c11.selectbox("ST Slope", [0,1,2],
            format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
        nv    = c12.selectbox("Num Major Vessels (0–3)", [0,1,2,3])

        thal  = st.selectbox("Thalassemia", [0,1,2,3],
            format_func=lambda x: {0:"Normal",1:"Fixed Defect",
                                    2:"Reversible Defect",3:"Unknown"}[x])

        submitted = st.form_submit_button("🧠 Calculate Risk", use_container_width=True, type="primary")

    if submitted:
        values = {
            "age": age, "sex": sex, "chest_pain_type": cp,
            "resting_bp": rbp, "cholesterol": chol,
            "fasting_blood_sugar": fbs, "resting_ecg": ecg,
            "max_heart_rate": mhr, "exercise_angina": exang,
            "st_depression": std, "st_slope": slope,
            "num_vessels": nv, "thalassemia": thal,
        }
        pct = predict_risk(model, scaler, values)
        st.divider()
        risk_card(pct)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#dc3545" if pct >= .6 else "#ffc107" if pct >= .3 else "#28a745"},
                "steps": [
                    {"range": [0,  30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60,100], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color":"#343a40","width":4}, "value": pct*100}
            },
            title={"text": "Cardiac Arrest Probability"}
        ))
        fig.update_layout(height=300, margin=dict(t=60,b=0,l=30,r=30))
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.subheader("Clinical Interpretation")
        high_risk_factors = []
        if age > 60:        high_risk_factors.append(f"Age ({age} yrs — elevated risk above 60)")
        if chol > 240:      high_risk_factors.append(f"High cholesterol ({chol} mg/dL)")
        if rbp > 140:       high_risk_factors.append(f"High resting BP ({rbp} mmHg)")
        if cp == 3:         high_risk_factors.append("Asymptomatic chest pain (paradoxically high risk)")
        if exang == 1:      high_risk_factors.append("Exercise-induced angina present")
        if std > 2:         high_risk_factors.append(f"Significant ST depression ({std})")
        if nv > 0:          high_risk_factors.append(f"{nv} major vessel(s) affected")
        if thal in [1,2]:   high_risk_factors.append("Thalassemia defect detected")
        if mhr < 120:       high_risk_factors.append(f"Low max heart rate ({mhr} bpm)")

        if high_risk_factors:
            st.warning("**Contributing risk factors identified:**")
            for f in high_risk_factors:
                st.markdown(f"- {f}")
        else:
            st.success("No major individual risk factors flagged from the provided values.")

        st.info("⚕️ **Disclaimer:** This tool is for informational and educational purposes only. "
                "It does not constitute medical advice. Always consult a qualified healthcare professional "
                "for clinical decisions.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Upload Patient Data
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📂 Upload Patient Data":
    st.title("📂 Upload Patient Data")
    st.markdown("Upload CSV files containing cardiac patient records. The model retrains automatically on each upload.")

    # Template download
    st.subheader("1. Download CSV Template")
    sample_data = {
        "age":               [63, 37, 41, 56, 57],
        "sex":               [1,  1,  0,  1,  0],
        "chest_pain_type":   [3,  2,  1,  1,  0],
        "resting_bp":        [145,130,130,120,120],
        "cholesterol":       [233,250,204,236,354],
        "fasting_blood_sugar":[1,  0,  0,  0,  0],
        "resting_ecg":       [0,  1,  0,  1,  1],
        "max_heart_rate":    [150,187,172,178,163],
        "exercise_angina":   [0,  0,  0,  0,  1],
        "st_depression":     [2.3,3.5,1.4,0.8,0.6],
        "st_slope":          [0,  0,  2,  2,  2],
        "num_vessels":       [0,  0,  0,  0,  0],
        "thalassemia":       [1,  2,  2,  2,  2],
        "cardiac_arrest":    [1,  1,  1,  0,  0],
    }
    template_df = pd.DataFrame(sample_data)
    st.download_button(
        "⬇️ Download CSV Template",
        template_df.to_csv(index=False).encode(),
        "cardiac_template.csv",
        "text/csv",
        use_container_width=False
    )

    with st.expander("📋 Column descriptions"):
        descs = {
            "age":"Patient age in years",
            "sex":"0 = Female, 1 = Male",
            "chest_pain_type":"0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic",
            "resting_bp":"Resting blood pressure (mmHg)",
            "cholesterol":"Serum cholesterol (mg/dL)",
            "fasting_blood_sugar":"Fasting blood sugar >120 mg/dL: 0=No, 1=Yes",
            "resting_ecg":"0=Normal, 1=ST-T Wave Abnormality, 2=LV Hypertrophy",
            "max_heart_rate":"Maximum heart rate achieved",
            "exercise_angina":"Exercise-induced angina: 0=No, 1=Yes",
            "st_depression":"ST depression induced by exercise relative to rest",
            "st_slope":"Slope of peak exercise ST: 0=Upsloping, 1=Flat, 2=Downsloping",
            "num_vessels":"Number of major vessels coloured by fluoroscopy (0–3)",
            "thalassemia":"0=Normal, 1=Fixed Defect, 2=Reversible Defect",
            "cardiac_arrest":"TARGET — 0=No cardiac arrest, 1=Cardiac arrest",
        }
        for k, v in descs.items():
            st.markdown(f"**`{k}`** — {v}")

    # Upload
    st.subheader("2. Upload Your CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df)} rows, {len(new_df.columns)} columns")

            # Validate
            missing_cols = [c for c in FEATURES + [TARGET] if c not in new_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()

            # Preview
            st.dataframe(new_df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Total Records",    len(new_df))
            col2.metric("Positive Cases",   int(new_df[TARGET].sum()))

            st.subheader("3. Append & Retrain")
            mode = st.radio("How to handle existing data?",
                            ["Append to existing data", "Replace existing data"],
                            horizontal=True)

            if st.button("🚀 Save & Train Model", type="primary", use_container_width=True):
                with st.spinner("Training model…"):
                    existing = load_data()
                    if existing is not None and mode == "Append to existing data":
                        combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates()
                    else:
                        combined = new_df.copy()

                    save_data(combined)

                    if len(combined) < 20:
                        st.warning("Need at least 20 rows to train reliably. "
                                   "Data saved; please upload more records.")
                        st.stop()

                    if combined[TARGET].nunique() < 2:
                        st.error("Dataset must contain both positive (1) and negative (0) cases.")
                        st.stop()

                    model, scaler, meta = train_model(combined)
                    save_model(model, scaler, meta)

                st.success("✅ Model trained and saved!")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Training Samples", meta["n_samples"])
                col_b.metric("AUC (Test)",       f"{meta['auc_test']:.3f}")
                col_c.metric("AUC (CV 5-fold)",  f"{meta['auc_cv']:.3f}")
                st.balloons()

        except Exception as ex:
            st.error(f"Error reading file: {ex}")

    # Show current dataset stats
    existing = load_data()
    if existing is not None:
        st.divider()
        st.subheader("Current Training Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(existing))
        c2.metric("Positive Cases", int(existing[TARGET].sum()))
        c3.metric("Negative Cases", int((existing[TARGET]==0).sum()))
        c4.metric("Features", len(FEATURES))

        fig = px.histogram(existing, x="age", color=TARGET,
                           color_discrete_map={0:"#28a745",1:"#dc3545"},
                           title="Age Distribution by Outcome",
                           labels={TARGET:"Cardiac Arrest"},
                           barmode="overlay", opacity=0.7)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    meta = load_meta()
    if not meta:
        st.warning("No model has been trained yet. Upload data to train the model.")
        st.stop()

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-box"><div class="metric-val">{meta["auc_test"]:.3f}</div><div class="metric-lbl">AUC — Test Set</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-val">{meta["auc_cv"]:.3f}</div><div class="metric-lbl">AUC — 5-Fold CV</div></div>',  unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-box"><div class="metric-val">{meta["acc_test"]:.3f}</div><div class="metric-lbl">Accuracy — Test</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-box"><div class="metric-val">{meta["n_samples"]}</div><div class="metric-lbl">Training Samples</div></div>',   unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    # ROC Curve
    with col_left:
        st.subheader("ROC Curve")
        y_test = meta["y_test"]
        y_prob = meta["y_prob"]
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                name=f"Model (AUC={meta['auc_test']:.3f})",
                line=dict(color="#dc3545", width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                name="Random", line=dict(dash="dash", color="#adb5bd")))
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=340, legend=dict(x=0.55, y=0.1)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    with col_right:
        st.subheader("Confusion Matrix")
        cm = np.array(meta["cm"])
        labels = ["No Arrest", "Cardiac Arrest"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                        x=labels, y=labels,
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(meta["y_test"], meta["y_pred"],
                                   target_names=["No Arrest", "Cardiac Arrest"],
                                   output_dict=True)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    # Risk distribution
    st.subheader("Predicted Risk Distribution")
    prob_series = pd.Series(meta["y_prob"], name="Predicted Probability")
    fig = px.histogram(prob_series, nbins=30,
                       color_discrete_sequence=["#dc3545"],
                       title="Distribution of Predicted Probabilities on Test Set")
    fig.update_layout(height=280)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Last trained: {meta.get('trained_at','?')[:19]}  |  "
               f"Positive cases: {meta['n_positive']} / {meta['n_samples']}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ═══════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About CardioRisk AI")

    st.markdown("""
    ## What this system does
    **CardioRisk AI** is a health informatics tool that uses machine learning to estimate
    the probability of a patient experiencing a cardiac arrest based on clinical characteristics.

    ## How the model works
    The system uses a **Calibrated Gradient Boosting Classifier** — an ensemble of decision trees
    that learns patterns from historical patient data. The calibration step ensures the output
    probabilities are reliable (a 70% prediction means roughly 70 in 100 similar patients
    experienced cardiac arrest).

    The model **automatically retrains** every time new patient data is uploaded,
    so accuracy improves as more records are added.

    ## Features used for prediction
    | Feature | Clinical Significance |
    |---|---|
    | Age | Risk increases significantly after 45–55 |
    | Sex | Males have higher baseline risk |
    | Chest Pain Type | Asymptomatic CP paradoxically signals higher risk |
    | Resting Blood Pressure | Hypertension is a major cardiac risk factor |
    | Cholesterol | High LDL promotes atherosclerosis |
    | Fasting Blood Sugar | Diabetes significantly raises risk |
    | Resting ECG | ST-T changes indicate ischaemia |
    | Max Heart Rate | Lower max HR suggests reduced cardiac reserve |
    | Exercise Angina | Pain on exertion indicates coronary disease |
    | ST Depression | Ischaemia indicator during stress testing |
    | ST Slope | Shape of ST change during peak exercise |
    | Num Major Vessels | Degree of coronary artery disease |
    | Thalassemia | Blood disorder affecting oxygen delivery |

    ## Recommended dataset
    The model was designed for the **UCI Heart Disease Dataset** structure.
    You can download a free version from:
    - [Kaggle — Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
    - [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

    ## Disclaimer
    > This tool is for **educational and research purposes only**. It is not a medical device
    > and must not be used as a substitute for professional clinical judgement.
    > All predictions should be reviewed by a qualified healthcare provider.
    """)
