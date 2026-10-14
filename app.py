import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. APPLICATION SETUP & LOOK
st.set_page_config(page_title='Churn Risk Advisor', layout='wide', initial_sidebar_state='expanded')

RAW_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

# 2. CACHED INFRASTRUCTURE MANAGEMENT (Optimizes server processing)
@st.cache_resource
def load_production_artifacts():
    model = joblib.load('churn_model.joblib')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_production_artifacts()
    COLS = meta['feature_columns']
except Exception as e:
    st.error(f"Failed to load engine components. Ensure artifacts sit in the app directory. Error: {e}")
    st.stop()

def prepare_serving_input(raw_data, feature_columns):
    """Processes categorical text inputs cleanly without drop_first alignment slips."""
    X = pd.get_dummies(raw_data[RAW_COLS])
    return X.reindex(columns=feature_columns, fill_value=0).astype(float)

def execute_scoring(raw_frame):
    return model.predict_proba(prepare_serving_input(raw_frame, COLS))[:, 1]

# ----------------------------------------------------------------------
# 3. INTERACTIVE SIDEBAR CONTROL DECK (Single Profile Entry)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header('📌 Customer Input Profile')
    
    tenure = st.slider('Tenure (Account Age in Months)', 0, 72, 4)
    contract = st.selectbox('Contract Type Framework', ['Month-to-month', 'One year', 'Two year'])
    monthly = st.number_input('Monthly Charges (USD)', 18.0, 120.0, 85.0, step=0.5)
    internet = st.selectbox('Internet Service Type Architecture', ['Fiber optic', 'DSL', 'No'])
    
    services = []
    if internet != 'No':
        services = st.multiselect('Activated Add-on Services', SERVICES, default=['StreamingTV'])
        
    phone = st.radio('Phone Service Infrastructure', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple Lines Line Setup', ['No', 'Yes'], horizontal=True)
        
    payment = st.selectbox('Configured Payment Method', PAYMENTS)
    paperless = st.radio('Paperless Billing Configuration', ['Yes', 'No'], horizontal=True)
    
    with st.expander('👥 Demographic Attributes'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
        senior = st.checkbox('Senior Citizen Status Indicator')
        partner = st.checkbox('Has Legal Relationship Partner')
        dependents = st.checkbox('Has Living Dependents Support')
        
    st.divider()
    # Pull configured baseline default boundary from json configuration mapping
    threshold = st.slider('Target Decision Risk Threshold', 0.05, 0.95, float(meta['threshold']), 0.05)

# Transform structural entry deck values into structural dataframe row array
customer_row = {
    'gender': gender, 'SeniorCitizen': int(senior),
    'Partner': 'Yes' if partner else 'No', 'Dependents': 'Yes' if dependents else 'No',
    'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multi,
    'InternetService': internet, 'Contract': contract, 'PaperlessBilling': paperless,
    'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': tenure * monthly
}

for service in SERVICES:
    customer_row[service] = ('No internet service' if internet == 'No' 
                             else 'Yes' if service in services else 'No')

single_customer_df = pd.DataFrame([customer_row])

# ----------------------------------------------------------------------
# 4. MAIN CENTRAL INTERFACE PLATFORM
# ----------------------------------------------------------------------
st.title('🛡️ Enterprise Customer Churn Risk Advisor')
st.caption(f"Engine Core: **{meta['model_name']}** | Cross-Validation Standard AUC: **{meta['cv_auc']:.3f} ± {meta['cv_auc_std']:.3f}** | Status: Production Support Mode")

tab1, tab2, tab3 = st.tabs(['📊 Single Profile Diagnosis', '📁 Batch Automated Scoring', '📘 Model Architecture Sheet'])

# TAB 1: SINGLE CUSTOMER EVALUATION
with tab1:
    p_churn = float(execute_scoring(single_customer_df)[0])
    
    # Establish production-grade risk band tiers
    risk_tier = ('🔴 HIGH RISK' if p_churn >= threshold else 
                 '🟡 WATCH LIST' if p_churn >= (threshold / 2) else '🟢 LOW RISK')
    
    # Metric KPI Blocks Layout
    c1, c2, c3 = st.columns(3)
    c1.metric('Calculated Churn Probability', f'{p_churn:.1%}')
    c2.metric('Operational Risk Tier', risk_tier)
    c3.metric('Recommended Action Matrix', 'Initiate Corporate Outreach' if p_churn >= threshold else 'No Action Required')
    
    st.progress(p_churn)
    
    # WHAT-IF SIMULATION DECK (Strategic Sensitivity Matrix Calculation Engine)
    st.subheader('🔮 Automated What-If Optimization Sensitivity Matrix')
    st.markdown("<small>Simulates strategic account optimization scenarios to lower system risk models.</small>", unsafe_values=True)
    
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
        p_simulated = float(execute_scoring(simulated_df)[0])
        optimization_rows.append({
            'Strategic Action Parameter': f"Modify {feature} ➡️ {alternate_value}",
            'Simulated Risk': f"{p_simulated:.1%}",
            'Risk Reduction Shift Delta': f"{p_simulated - p_churn:+.1%}"
        })
        
    if optimization_rows:
        st.dataframe(pd.DataFrame(optimization_rows), use_container_width=True, hide_index=True)
        st.caption('⚠️ *Note: Statistical matrix observations show data correlations, not guaranteed causal downstream revenue behaviors.*')
    else:
        st.info("Account is already optimized to maximum operational retention guidelines.")

# TAB 2: BULK BATCH SCORING PROCESSING MODULE
with tab2:
    st.header('📁 Enterprise Bulk File Execution Processing')
    st.markdown("Upload raw transaction data matrices matching standard Telco schemas to run bulk operations.")
    
    uploaded_file = st.file_uploader("Drop target database CSV frame here", type='csv')
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        
        missing_columns = [col for col in RAW_COLS if col not in batch_data.columns]
        if missing_columns:
            st.error(f"Structural Parsing Failure! Missing expected schema dimensions: {missing_columns}")
            st.stop()
            
        # Repair textual blank types safely across evaluation frames
        batch_data['TotalCharges'] = pd.to_numeric(batch_data['TotalCharges'], errors='coerce').fillna(0)
        
        # Execute vectorized model evaluation scoring loops
        batch_data['Churn_Risk_Probability'] = execute_scoring(batch_data).round(4)
        batch_data['Actionable_Contact_Flag'] = np.where(batch_data['Churn_Risk_Probability'] >= threshold, '🚨 CONTACT NOW', '✅ RETAIN')
        
        flagged_count = (batch_data['Actionable_Contact_Flag'] == '🚨 CONTACT NOW').sum()
        st.warning(f"Analysis complete: Flagged **{flagged_count}** accounts out of **{len(batch_data)}** processed records for immediate mitigation outreach.")
        
        st.dataframe(batch_data.sort_values('Churn_Risk_Probability', ascending=False), use_container_width=True)
        
        # Export processed matrix as downloadable database artifact
        processed_csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored Accounts Database File",
            data=processed_csv,
            file_name="scored_retention_accounts.csv",
            mime="text/csv"
        )

# TAB 3: ENTERPRISE ARCHITECTURE SHEET
with tab3:
    st.header('📘 Technical Model Specification Card')
    st.json({k: v for k, v in meta.items() if k != 'feature_columns'})
    st.markdown("""
    **Core Operational Constraints & Guardrails:**
    - **Dataset Domain:** Model architecture trained completely on historical US telecom transaction profiles. 
    - **Geographic Limits:** Not verified or calibrated for deployment inside Pakistani regional telecommunication frameworks (Jazz, Telenor, Zong) without localized retraining matrices.
    - **Drift Warning:** Predictive capabilities fade over time as corporate data structures change or billing campaigns shift.
    """)
