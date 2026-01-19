import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Concrete Properties Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with the blue color palette
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #EEF6FF 0%, #DAE9FF 100%);
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2C67F2 0%, #1748DE 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(44, 103, 242, 0.3);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .section-header {
        color: #1B368D;
        border-left: 4px solid #5BA3FF;
        padding-left: 15px;
        margin: 20px 0 15px 0;
        font-weight: 600;
    }
    
    .input-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #357EFC 0%, #2C67F2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(53, 126, 252, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(53, 126, 252, 0.6);
    }
    
    .sidebar .stSelectbox label, .sidebar .stNumberInput label {
        color: #1B368D !important;
        font-weight: 500;
    }
    
    h1 {
        color: #1B368D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #BDDAFF 0%, #90C3FF 100%);
    }
    
    .healing-efficiency-card {
        background: linear-gradient(135deg, #5BA3FF 0%, #357EFC 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏗️ Concrete Properties Predictor")
st.markdown("### Predicting Compressive Strength & Slump from Mix Design Parameters")
st.markdown("---")

# Load models (you would need to ensure models are available)
@st.cache_resource
def load_models():
    try:
        with open("xgb_model_strength.pkl", "rb") as f:
            strength_model = pickle.load(f)
        with open("best_model_slump.pkl", "rb") as f:
            slump_model = pickle.load(f)
        with open("slump_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return strength_model, slump_model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the trained models are available.")
        return None, None, None

# Default values for inputs (based on your test data)
default_values = {
    'Fineness of Cement': 398.0,
    'Fineness of Fly ash m3/kg': 398.0,
    'Soundness of Cement': 7.89,
    'Specific Gravity of Aggregates (10mm)': 2.64,
    'Specific Gravity of Aggregates (20mm)': 2.74,
    'Specific Gravity of Aggregates (Crush Sand)': 2.65,
    'Water Absorption of Aggregates (10 mm)': 0.94,
    'Water Absorption of Aggregates (20 mm)': 0.48,
    'Water Absorption of Aggregates (Crush Sand)': 2.14,
    'Aggregate Crushing Value (10 mm)': 16.27,
    'Aggregate Crushing Value (20mm)': 22.78,
    'Zone of Sand': 4,
    'Silt content %': 6.1,
    'pH of water during making of concrete': 8.4,
    'Water/Cement ratio kg/m3': 0.40,
    'Dosage of Admixture (kg/m3)': 5.2,
    'Mix Design SAP_Dosage(%)': 0.10,
    'Cement Content  (kg/m3)': 394.9,
    'Fly Ash Content': 76.5,
    'Coarse aggregates kg/m3': 1103.9,
    'Fine aggregates  (kg/m3)': 683.4,
    'Water  (kg/m3)': 185.8,
    'Curing days': 28.0,
    'Self-Healing Efficiency': 37.36
}

# Sidebar for inputs
st.sidebar.markdown('<div class="section-header"><h2>Input Parameters</h2></div>', unsafe_allow_html=True)

# Group inputs into logical sections
cement_inputs = {}
aggregate_inputs = {}
mix_inputs = {}
material_inputs = {}
special_inputs = {}

# Cement and Fly Ash Properties
with st.sidebar.expander("🏭 Cement & Fly Ash Properties", expanded=True):
    cement_inputs['Fineness of Cement'] = st.number_input(
        "Fineness of Cement", 
        value=default_values['Fineness of Cement'],
        min_value=0.0, max_value=1000.0, step=1.0
    )
    cement_inputs['Fineness of Fly ash m3/kg'] = st.number_input(
        "Fineness of Fly ash (m³/kg)", 
        value=default_values['Fineness of Fly ash m3/kg'],
        min_value=0.0, max_value=1000.0, step=1.0
    )
    cement_inputs['Soundness of Cement'] = st.number_input(
        "Soundness of Cement", 
        value=default_values['Soundness of Cement'],
        min_value=0.0, max_value=20.0, step=0.01
    )

# Aggregate Properties
with st.sidebar.expander("🪨 Aggregate Properties", expanded=False):
    aggregate_inputs['Specific Gravity of Aggregates (10mm)'] = st.number_input(
        "Specific Gravity (10mm)", 
        value=default_values['Specific Gravity of Aggregates (10mm)'],
        min_value=1.0, max_value=4.0, step=0.01
    )
    aggregate_inputs['Specific Gravity of Aggregates (20mm)'] = st.number_input(
        "Specific Gravity (20mm)", 
        value=default_values['Specific Gravity of Aggregates (20mm)'],
        min_value=1.0, max_value=4.0, step=0.01
    )
    aggregate_inputs['Specific Gravity of Aggregates (Crush Sand)'] = st.number_input(
        "Specific Gravity (Crush Sand)", 
        value=default_values['Specific Gravity of Aggregates (Crush Sand)'],
        min_value=1.0, max_value=4.0, step=0.01
    )
    aggregate_inputs['Water Absorption of Aggregates (10 mm)'] = st.number_input(
        "Water Absorption (10mm) %", 
        value=default_values['Water Absorption of Aggregates (10 mm)'],
        min_value=0.0, max_value=10.0, step=0.01
    )
    aggregate_inputs['Water Absorption of Aggregates (20 mm)'] = st.number_input(
        "Water Absorption (20mm) %", 
        value=default_values['Water Absorption of Aggregates (20 mm)'],
        min_value=0.0, max_value=10.0, step=0.01
    )
    aggregate_inputs['Water Absorption of Aggregates (Crush Sand)'] = st.number_input(
        "Water Absorption (Crush Sand) %", 
        value=default_values['Water Absorption of Aggregates (Crush Sand)'],
        min_value=0.0, max_value=10.0, step=0.01
    )
    aggregate_inputs['Aggregate Crushing Value (10 mm)'] = st.number_input(
        "Crushing Value (10mm) %", 
        value=default_values['Aggregate Crushing Value (10 mm)'],
        min_value=0.0, max_value=50.0, step=0.01
    )
    aggregate_inputs['Aggregate Crushing Value (20mm)'] = st.number_input(
        "Crushing Value (20mm) %", 
        value=default_values['Aggregate Crushing Value (20mm)'],
        min_value=0.0, max_value=50.0, step=0.01
    )

# Mix Design Parameters
with st.sidebar.expander("⚗️ Mix Design & Water Properties", expanded=False):
    mix_inputs['Zone of Sand'] = st.selectbox(
        "Zone of Sand", 
        options=[1, 2, 3, 4],
        index=3  # Default to 4
    )
    mix_inputs['Silt content %'] = st.number_input(
        "Silt Content %", 
        value=default_values['Silt content %'],
        min_value=0.0, max_value=20.0, step=0.1
    )
    mix_inputs['pH of water during making of concrete'] = st.number_input(
        "pH of Water", 
        value=default_values['pH of water during making of concrete'],
        min_value=6.0, max_value=9.0, step=0.1
    )
    mix_inputs['Water/Cement ratio kg/m3'] = st.number_input(
        "Water/Cement Ratio", 
        value=default_values['Water/Cement ratio kg/m3'],
        min_value=0.1, max_value=1.0, step=0.01
    )
    mix_inputs['Dosage of Admixture (kg/m3)'] = st.number_input(
        "Admixture Dosage (kg/m³)", 
        value=default_values['Dosage of Admixture (kg/m3)'],
        min_value=0.0, max_value=20.0, step=0.1
    )
    mix_inputs['Mix Design SAP_Dosage(%)'] = st.number_input(
        "SAP Dosage %", 
        value=default_values['Mix Design SAP_Dosage(%)'],
        min_value=0.0, max_value=2.0, step=0.01
    )

# Material Quantities and Curing
with st.sidebar.expander("📊 Material Quantities & Curing", expanded=False):
    material_inputs['Cement Content  (kg/m3)'] = st.number_input(
        "Cement Content (kg/m³)", 
        value=default_values['Cement Content  (kg/m3)'],
        min_value=100.0, max_value=800.0, step=1.0
    )
    material_inputs['Fly Ash Content'] = st.number_input(
        "Fly Ash Content (kg/m³)", 
        value=default_values['Fly Ash Content'],
        min_value=0.0, max_value=200.0, step=1.0
    )
    material_inputs['Coarse aggregates kg/m3'] = st.number_input(
        "Coarse Aggregates (kg/m³)", 
        value=default_values['Coarse aggregates kg/m3'],
        min_value=800.0, max_value=1500.0, step=1.0
    )
    material_inputs['Fine aggregates  (kg/m3)'] = st.number_input(
        "Fine Aggregates (kg/m³)", 
        value=default_values['Fine aggregates  (kg/m3)'],
        min_value=400.0, max_value=900.0, step=1.0
    )
    material_inputs['Water  (kg/m3)'] = st.number_input(
        "Water Content (kg/m³)", 
        value=default_values['Water  (kg/m3)'],
        min_value=100.0, max_value=300.0, step=1.0
    )
    material_inputs['Curing days'] = st.number_input(
        "Curing Days", 
        value=default_values['Curing days'],
        min_value=1.0, max_value=90.0, step=1.0
    )

# Self-Healing Efficiency (now an input parameter)
with st.sidebar.expander("🔧 Special Properties", expanded=True):
    special_inputs['Self-Healing Efficiency'] = st.number_input(
        "Self-Healing Efficiency (%)", 
        value=default_values['Self-Healing Efficiency'],
        min_value=0.0, max_value=100.0, step=0.01,
        help="This is now an input parameter used for prediction"
    )

# Combine all inputs
all_inputs = {**cement_inputs, **aggregate_inputs, **mix_inputs, **material_inputs, **special_inputs}

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="section-header"><h2>🎯 Prediction Results</h2></div>', unsafe_allow_html=True)
    
    if st.button("🔮 Predict Properties", type="primary"):
        # Create DataFrame from inputs
        input_data = pd.DataFrame([all_inputs])
        
        # Load models and scaler
        strength_model, slump_model, scaler = load_models()
        
        if strength_model is not None and slump_model is not None and scaler is not None:
            try:
                # Make predictions
                strength_pred = strength_model.predict(input_data)[0]
                
                # Scale data for slump prediction
                input_data_scaled = scaler.transform(input_data)
                slump_pred = slump_model.predict(input_data_scaled)[0]
                
                # Display results
                col_strength, col_slump = st.columns(2)
                
                with col_strength:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{strength_pred:.2f}</div>
                        <div class="metric-label">Compressive Strength (MPa)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_slump:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{slump_pred:.1f}</div>
                        <div class="metric-label">Slump (mm)</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create a gauge chart for visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "indicator"}, {"type": "indicator"}]],
                    subplot_titles=("Compressive Strength", "Slump")
                )
                
                # Compressive Strength Gauge
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = strength_pred,
                    title = {'text': ""},
                    domain = {'x': [0, 0.5], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 50]},
                        'bar': {'color': "#357EFC"},
                        'steps': [
                            {'range': [0, 20], 'color': "#BDDAFF"},
                            {'range': [20, 30], 'color': "#90C3FF"},
                            {'range': [30, 40], 'color': "#5BA3FF"},
                            {'range': [40, 50], 'color': "#357EFC"}
                        ],
                        'threshold': {
                            'line': {'color': "#1B368D", 'width': 4},
                            'thickness': 0.75,
                            'value': 45
                        }
                    }
                ), row=1, col=1)
                
                # Slump Gauge
                fig.add_trace(go.Indicator(
                    mode = "gauge+number",
                    value = slump_pred,
                    title = {'text': ""},
                    domain = {'x': [0.5, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 200]},
                        'bar': {'color': "#2C67F2"},
                        'steps': [
                            {'range': [0, 50], 'color': "#BDDAFF"},
                            {'range': [50, 100], 'color': "#90C3FF"},
                            {'range': [100, 150], 'color': "#5BA3FF"},
                            {'range': [150, 200], 'color': "#357EFC"}
                        ],
                        'threshold': {
                            'line': {'color': "#1B368D", 'width': 4},
                            'thickness': 0.75,
                            'value': 180
                        }
                    }
                ), row=1, col=2)
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance interpretation
                st.markdown("#### 📋 Performance Interpretation")
                
                # Compressive Strength interpretation
                if strength_pred >= 40:
                    strength_category = "Excellent"
                    strength_color = "#357EFC"
                elif strength_pred >= 35:
                    strength_category = "Good"
                    strength_color = "#5BA3FF"
                elif strength_pred >= 25:
                    strength_category = "Moderate"
                    strength_color = "#90C3FF"
                else:
                    strength_category = "Low"
                    strength_color = "#BDDAFF"
                
                # Slump interpretation
                if slump_pred >= 150:
                    slump_category = "High Workability"
                    slump_color = "#357EFC"
                elif slump_pred >= 100:
                    slump_category = "Medium Workability"
                    slump_color = "#5BA3FF"
                elif slump_pred >= 50:
                    slump_category = "Low Workability"
                    slump_color = "#90C3FF"
                else:
                    slump_category = "Very Low Workability"
                    slump_color = "#BDDAFF"
                
                col_interp1, col_interp2 = st.columns(2)
                
                with col_interp1:
                    st.markdown(f"""
                    <div style="background: {strength_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                        <strong>{strength_category}</strong><br>
                        Compressive Strength
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_interp2:
                    st.markdown(f"""
                    <div style="background: {slump_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;">
                        <strong>{slump_category}</strong><br>
                        Concrete Workability
                    </div>
                    """, unsafe_allow_html=True)
                
                # Store results in session state for history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                st.session_state.prediction_history.append({
                    'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Compressive Strength': strength_pred,
                    'Slump': slump_pred,
                    'Self-Healing Efficiency': all_inputs['Self-Healing Efficiency']
                })
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Models not available. Please ensure the trained models are in the same directory.")

with col2:
    st.markdown('<div class="section-header"><h3>📈 Input Summary</h3></div>', unsafe_allow_html=True)
    
    # Display key parameters
    key_params = {
        'W/C Ratio': f"{all_inputs['Water/Cement ratio kg/m3']:.2f}",
        'Cement Content': f"{all_inputs['Cement Content  (kg/m3)']} kg/m³",
        'Curing Days': f"{int(all_inputs['Curing days'])} days",
        'SAP Dosage': f"{all_inputs['Mix Design SAP_Dosage(%)']}%"
    }
    
    for param, value in key_params.items():
        st.metric(param, value)
    
    # # Display Self-Healing Efficiency as an input parameter
    # st.markdown('<div class="section-header"><h4>🔧 Special Input</h4></div>', unsafe_allow_html=True)
    # st.markdown(f"""
    # <div class="healing-efficiency-card">
    #     <div style="font-size: 1.5rem; font-weight: bold;">{all_inputs['Self-Healing Efficiency']:.2f}%</div>
    #     <div style="opacity: 0.9;">Self-Healing Efficiency</div>
    #     <div style="font-size: 0.8rem; margin-top: 5px;">(Input Parameter)</div>
    # </div>
    # """, unsafe_allow_html=True)

# Prediction History
if 'prediction_history' in st.session_state and st.session_state.prediction_history:
    st.markdown('<div class="section-header"><h2>📊 Prediction History</h2></div>', unsafe_allow_html=True)
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Display recent predictions
    st.dataframe(
        history_df.tail(10).style.format({
            'Compressive Strength': '{:.2f}',
            'Slump': '{:.1f}',
            'Self-Healing Efficiency': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    if len(history_df) > 1:
        # Plot history
        fig_history = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Compressive Strength Trends", "Slump Trends"),
            vertical_spacing=0.12
        )
        
        fig_history.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['Compressive Strength'],
            mode='lines+markers',
            name='Compressive Strength',
            line=dict(color='#357EFC', width=3),
            marker=dict(size=8)
        ), row=1, col=1)
        
        fig_history.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['Slump'],
            mode='lines+markers',
            name='Slump',
            line=dict(color='#2C67F2', width=3),
            marker=dict(size=8)
        ), row=2, col=1)
        
        fig_history.update_xaxes(title_text="Prediction Number", row=2, col=1)
        fig_history.update_yaxes(title_text="Strength (MPa)", row=1, col=1)
        fig_history.update_yaxes(title_text="Slump (mm)", row=2, col=1)
        
        fig_history.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        
        st.plotly_chart(fig_history, use_container_width=True)

# Batch Prediction Feature
st.markdown('<div class="section-header"><h2>📋 Batch Prediction</h2></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=['csv'])
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_data.head())
        
        if st.button("🚀 Run Batch Predictions"):
            strength_model, slump_model, scaler = load_models()
            if strength_model and slump_model and scaler:
                # Make batch predictions
                strength_preds = strength_model.predict(batch_data)
                batch_data_scaled = scaler.transform(batch_data)
                slump_preds = slump_model.predict(batch_data_scaled)
                
                # Create results
                results = batch_data.copy()
                results['Predicted_Compressive_Strength'] = strength_preds
                results['Predicted_Slump'] = slump_preds
                
                st.success("Batch predictions completed!")
                st.dataframe(results)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name='batch_predictions.csv',
                    mime='text/csv'
                )
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #1B368D; padding: 20px;">© Krishna.ai - @consulting.krishnaai@gmail.com</div>', 
    unsafe_allow_html=True
)