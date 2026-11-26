import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Styling (simple & clean)
# -----------------------------
CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #f6fbff 0%, #ffffff 45%, #f6fbff 100%);
}
.header-card {
    padding: 18px 22px;
    border-radius: 16px;
    background: #0b5cab;
    color: white;
    box-shadow: 0 10px 25px rgba(11, 92, 171, 0.25);
    margin-bottom: 16px;
}
.header-title {
    font-size: 28px;
    font-weight: 800;
    margin: 0;
}
.header-subtitle {
    margin-top: 6px;
    opacity: 0.95;
    font-size: 14px;
}
.section-card {
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid #e6eef7;
    background: white;
    box-shadow: 0 6px 18px rgba(31, 73, 125, 0.06);
    margin-bottom: 12px;
}
.section-title {
    font-size: 16px;
    font-weight: 800;
    color: #0b5cab;
    margin-bottom: 10px;
}
.result-card {
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid #e6eef7;
    background: #ffffff;
    box-shadow: 0 8px 22px rgba(31, 73, 125, 0.08);
}
.small-note {
    font-size: 12px;
    color: #5b6b7b;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def nice_label(col_name: str) -> str:
    return col_name.replace("_", " ").title()


def yesno_to_int(val: str) -> int:
    """Convert Yes/No string to 1/0 for the model."""
    return 1 if val == "Yes" else 0


@st.cache_resource
def load_bundle(bundle_path="xgb_model.pkl"):
    """Load pre-trained pipeline + metadata bundle (dict)."""
    if not os.path.exists(bundle_path):
        return None
    return joblib.load(bundle_path)


# -----------------------------
# Sidebar (Threshold slider only)
# -----------------------------
st.sidebar.title("⚙️ Settings")

DEFAULT_THRESHOLD = 0.50

threshold = st.sidebar.slider(
    "Decision threshold (Diabetes = 1)",
    0.05, 0.95, DEFAULT_THRESHOLD, 0.01
)

st.sidebar.caption("Default threshold is 0.50. You can adjust it using the slider.")
st.sidebar.markdown("---")
st.sidebar.info("This is an educational demo. Not medical advice.")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">🩺 Diabetes Prediction System</div>
        <div class="header-subtitle">
            Uses a pre-trained XGBoost model for consistent predictions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load Model Bundle
# -----------------------------
MODEL_FILE = "xgb_model.pkl"
bundle = load_bundle(MODEL_FILE)

if bundle is None:
    st.error(
        f"❌ Model bundle file not found: **{MODEL_FILE}**\n\n"
        "👉 First run your training script and save the bundle using joblib."
    )
    st.stop()

# Enforce dict bundle (since you saved metadata)
if not isinstance(bundle, dict):
    st.error(
        "❌ Loaded a pipeline-only model. This app requires a **dict bundle** with metadata.\n\n"
        "✅ Please re-save your model like this:\n"
        "joblib.dump({'pipeline': pipeline, 'categorical_columns': ..., 'numeric_columns': ..., "
        "'cat_options': ..., 'num_bounds': ..., 'num_defaults': ...}, 'xgb_model.pkl')"
    )
    st.stop()

pipeline = bundle.get("pipeline", None)
categorical_columns = bundle.get("categorical_columns", [])
numeric_columns = bundle.get("numeric_columns", [])
cat_options = bundle.get("cat_options", {})
num_bounds = bundle.get("num_bounds", {})
num_defaults = bundle.get("num_defaults", {})

if pipeline is None:
    st.error("❌ 'pipeline' not found in the bundle. Please save model as dict bundle with key 'pipeline'.")
    st.stop()


# -----------------------------
# UI Sections
# -----------------------------
SECTION_MAP = {
    "Personal & Lifestyle": [
        "age",
        "gender",
        "ethnicity",
        "education_level",
        "income_level",
        "employment_status",
        "smoking_status",
        "alcohol_consumption_per_week",
        "physical_activity_minutes_per_week",
        "diet_score",
        "sleep_hours_per_day",
        "screen_time_hours_per_day",
    ],
    "Body Measurements & Vitals": [
        "bmi",
        "waist_to_hip_ratio",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
    ],
    "Cholesterol & Lipids": [
        "cholesterol_total",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
    ],
    "Medical History": [
        "family_history_diabetes",
        "hypertension_history",
        "cardiovascular_history",
    ],
}


def filter_existing(fields):
    return [f for f in fields if (f in numeric_columns) or (f in categorical_columns)]


for sec in list(SECTION_MAP.keys()):
    SECTION_MAP[sec] = filter_existing(SECTION_MAP[sec])


# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):
    st.markdown(
        '<div class="section-card"><div class="section-title">🧾 Patient Inputs</div>',
        unsafe_allow_html=True
    )

    inputs = {}

    for section_name, fields in SECTION_MAP.items():
        if not fields:
            continue

        st.subheader(section_name)
        cols = st.columns(3)

        for i, field in enumerate(fields):
            col = cols[i % 3]

            if field in categorical_columns:
                options = cat_options.get(field, ["Unknown"])
                inputs[field] = col.selectbox(nice_label(field), options, index=0)

            elif field in numeric_columns:
                lo, hi = num_bounds.get(field, (0.0, 100.0))
                default = num_defaults.get(field, (lo + hi) / 2)

                # Medical history fields: show Yes/No instead of 0/1
                is_medical_history = field in [
                    "family_history_diabetes",
                    "hypertension_history",
                    "cardiovascular_history",
                ]

                if is_medical_history:
                    default_choice = "Yes" if float(default) >= 0.5 else "No"
                    yn = col.selectbox(
                        nice_label(field),
                        options=["No", "Yes"],
                        index=1 if default_choice == "Yes" else 0,
                        help="No = 0, Yes = 1"
                    )
                    inputs[field] = yesno_to_int(yn)  # Convert to 0/1 for model
                else:
                    step = 1.0 if float(lo).is_integer() and float(hi).is_integer() else 0.1
                    inputs[field] = col.number_input(
                        nice_label(field),
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(np.clip(default, lo, hi)),
                        step=step,
                        format="%.2f" if step == 0.1 else "%.0f"
                    )

        st.markdown("---")

    submitted = st.form_submit_button("🔍 Predict Diabetes Risk")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Prediction Output
# -----------------------------
if submitted:
    # Build input row in expected order
    row = {c: inputs.get(c, np.nan) for c in (categorical_columns + numeric_columns)}
    input_df = pd.DataFrame([row])

    prob_diabetes = float(pipeline.predict_proba(input_df)[0][1])
    pred_class = int(prob_diabetes >= threshold)

    # Risk band
    if prob_diabetes < 0.30:
        band = "Low risk"
        icon = "🟢"
    elif prob_diabetes < 0.70:
        band = "Moderate risk"
        icon = "🟠"
    else:
        band = "High risk"
        icon = "🔴"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("📌 Prediction Result")

    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    c1.metric("Predicted Class", f"{pred_class} ({'Has Diabetes' if pred_class==1 else 'Does Not Have Diabetes'})")
    c2.metric("Probability (Diabetes=1)", f"{prob_diabetes*100:.2f}%")
    c3.metric("Risk Level", f"{icon} {band}")

    st.progress(min(max(prob_diabetes, 0.0), 1.0))

    st.caption(f"Decision rule: Diabetes = 1 if probability ≥ threshold ({threshold:.2f}).")

    st.markdown(
        '<p class="small-note">⚠️ This is a demo prediction and not a medical diagnosis.</p>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)