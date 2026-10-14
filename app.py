import json
import joblib
import os
import logging
import numpy as np
import pandas as pd
import streamlit as st

# 1. TELEMETRY & LOGGER CONFIGURATION
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. GLOBAL STYLING FRAMEWORK & DESIGN SYSTEM Injection
st.set_page_config(
    page_title='Enterprise Churn Risk Advisor',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Premium executive dark/light styling rules injection
st.markdown("""
    <style>
        @import url('https://googleapis.com');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .custom-caption {
            font-size: 0.85rem;
            color: #8892b0;
            margin-top: -8px;
            margin-bottom: 24px;
        }
    </style>
""", unsafe_allow_html=True)

# 3. SCHEMA & CONSTANTS SPECIFICATIONS
RAW_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

# 4. ROBUST RESOURCE LAYER (Cached memory optimization)
@st.cache_resource(show_spinner="Bootstrapping secure model pipelines...")
def load_production_artifacts():
    try:
        if not os.path.exists('churn_model.joblib') or not os.path.exists('model_meta.json'):
            logger.error("Missing critical files inside execution workspace environment.")
            return None, None
        model = joblib.load('churn_model.joblib')
        with open('model_meta.json') as f:
            meta = json.load(f)
        logger.info("Successfully mounted prediction engine configuration blocks.")
        return model, meta
    except Exception as e:
        logger.critical(f"Pipeline loading crash trace: {e}")
        return None, None

model, meta = load_production_artifacts()

if model is None or meta is None:
    st.error("### 🚨 System Boot Failure: Missing Production Pipeline Components")
    st.info("Please verify that `churn_model.joblib` and `model_meta.json` sit securely inside the root project repository directory.")
    st.stop()

COLS = meta['feature_columns']

# 5. DATA INFERENCE ENGINE PIPELINES
def prepare_serving_input(raw_data: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Safely handles one-hot encoding alignment transitions for out-of-fold inputs."""
    X = pd.get_dummies(raw_data[RAW_COLS])
    return X.reindex(columns=feature_columns, fill_value=0).astype(float)

def execute_inference(raw_frame: pd.DataFrame) -> np.ndarray:
    try:
        processed_frame = prepare_serving_input(raw_frame, COLS)
        return model.predict_proba(processed_frame)[:, 1]
    except Exception as e:
        st.error(f"Prediction layer interface error: {e}")
        return np.array([0.0])

# ----------------------------------------------------------------------
# 6. ENTERPRISE SIDEBAR DECK INTERFACE
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Dashboard")
    st.markdown("### Profile Parameter Mapping")
    
    tenure = st.slider('Account Lifespan (Tenure Months)', 0, 72, 12, help="Duration of active account with provider.")
    contract = st.selectbox('Subscription Contract Schema', ['Month-to-month', 'One year', 'Two year'])
    monthly = st.number_input('Assigned Monthly Outlay (USD)', 18.0, 120.0, 75.0, step=1.0)
    internet = st.selectbox('Pipeline Internet Distribution Network', ['Fiber optic', 'DSL', 'No'])
    
    services = []
    if internet != 'No':
        services = st.multiselect('Active Integrated Cloud Services Add-ons', SERVICES, default=['OnlineSecurity'])
        
    phone = st.radio('Voice Line Infrastructure Port Allocation', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple Active Secondary Numbers Assigned', ['No', 'Yes'], horizontal=True)
        
    payment = st.selectbox('Active Transaction Clearing Method', PAYMENTS)
    paperless = st.radio('Paperless System Billing Opt-in', ['Yes', 'No'], horizontal=True)
    
    with st.sidebar.expander('👥 Client Demographic Fields'):
        gender = st.radio('Identified Gender Profile', ['Female', 'Male'], horizontal=True)
        senior = st.checkbox('Senior Citizen System Metric Classification Flag')
        partner = st.checkbox('Verified Marital Relationship Partner Status')
        dependents = st.checkbox('Active Family Dependents Coverage Status')
        
    st.divider()
    threshold = st.slider('Risk Calibration Decision Limit (t*)', 0.05, 0.95, float(meta['threshold']), 0.05)

# Transform structural UI controls entries down into execution vectors matrix
customer_dict = {
    'gender': gender, 'SeniorCitizen': int(senior), 'Partner': 'Yes' if partner else 'No',
    'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure, 'PhoneService': phone,
    'MultipleLines': multi, 'InternetService': internet, 'Contract': contract,
    'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': monthly,
    'TotalCharges': tenure * monthly
}

for svc in SERVICES:
    customer_dict[svc] = ('No internet service' if internet == 'No' else 'Yes' if svc in services else 'No')

single_customer_df = pd.DataFrame([customer_dict])

# ----------------------------------------------------------------------
# 7. MAIN INTERFACE FRAMEWORK - LARGE SCALE DATA CONTROL PLATFORM
# ----------------------------------------------------------------------
st.title('🛡️ Enterprise Customer Churn Risk Advisor')
st.markdown(f"<div class='custom-caption'>Production Cluster Framework: <b>{meta['model_name']}</b> Engine Core | Cross-Validation Framework Standard AUC: <b>{meta['cv_auc']:.3f} ± {meta['cv_auc_std']:.3f}</b></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    '📊 Real-Time Diagnostic Node', 
    '📁 Micro-Batch Processing Module', 
    '📘 Platform Architecture & Card'
])

# --- TAB 1: ADVANCED DIAGNOSTICS & SYSTEM WHAT-IF MATRIX ---
with tab1:
    p_churn = float(execute_inference(single_customer_df)[0])
    
    # Establish operational risk band frameworks
    if p_churn >= threshold:
        risk_color, risk_tier, structural_action = "crimson", "🔴 CRITICAL ACCELERATION RISK", "Immediate Outbound Retention Call Matrix Requested"
    elif p_churn >= (threshold / 2):
        risk_color, risk_tier, structural_action = "gold", "🟡 MONITORED WATCH BOUNDARY", "Queue Account for Standard Loyalty Campaign Placement"
    else:
        risk_color, risk_tier, structural_action = "emerald", "🟢 STABLE RETENTION BAND", "Maintain Standard Pipeline Operations Framework"

    # HTML Card layout formatting block elements
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-card'><h4>Calculated Churn Vector</h4><h2 style='color:#61afef;'>{p_churn:.2%}</h2><small>Calculated from profile parameters matrix</small></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><h4>Operational Stratum Status</h4><h3 style='color:{risk_color}; margin-top:12px;'>{risk_tier}</h3></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><h4>Prescriptive Mitigation Matrix</h4><h4 style='color:#98c379; margin-top:8px;'>{structural_action}</h4></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(p_churn)
    
    # ADVANCED SCENARIO OPTIMIZATION ENGINE
    st.subheader('🔮 Automated Operational What-If Minimization Strategy Deck')
    st.markdown("<p style='font-size:0.9rem; color:#8892b0;'>Simulates proactive account structural updates to isolate ideal retention trajectories.</p>", unsafe_allow_html=True)
    
    what_if_scenarios = []
    if contract != 'Two year':
        what_if_scenarios.append(('Contract', 'Two year' if contract == 'Month-to-month' else 'One year'))
    if payment == 'Electronic check':
        what_if_scenarios.append(('PaymentMethod', 'Credit card (automatic)'))
    if internet != 'No' and 'TechSupport' not in services:
        what_if_scenarios.append(('TechSupport', 'Yes'))
        
    optimization_rows = []
    for feature, alternate_value in what_if_scenarios:
        simulated_df = single_customer_df.copy()
        simulated_df[feature] = alternate_value
        p_simulated = float(execute_inference(simulated_df)[0])
        variance_shift_delta = p_simulated - p_churn
        
        optimization_rows.append({
            'Strategic Optimization Step': f"Transition profile dimension [{feature}] ➡️ '{alternate_value}'",
            'Simulated Risk Velocity': f"{p_simulated:.2%}",
            'Risk Reduction Target Variance Delta': f"{variance_shift_delta:+.2%}"
        })
        
    if optimization_rows:
        st.dataframe(pd.DataFrame(optimization_rows), use_container_width=True, hide_index=True)
