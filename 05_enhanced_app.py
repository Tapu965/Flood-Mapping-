"""
ENHANCED STREAMLIT APP with Damage Assessment
Run: streamlit run 05_enhanced_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
import sys
import time
import gc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.enhanced_predictor import EnhancedFloodPredictor

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Enhanced Flood Damage Assessment",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #0D47A1;
        font-size: 1.5rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .damage-high {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .damage-medium {
        color: #ffa500;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .damage-low {
        color: #00cc66;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌊 Enhanced Flood Damage Assessment System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Damage Estimation & Recovery Planning</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the enhanced model"""
    model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'flood_model.h5')
    if os.path.exists(model_path):
        try:
            with st.spinner("Loading enhanced damage assessment model..."):
                predictor = EnhancedFloodPredictor('config.yaml', 'models/trained_models/flood_model.h5')
                return predictor
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# Load the model
predictor = load_model()

if predictor is None:
    st.error("""
    ### ❌ Model not found!
    
    Please run these commands first:
    1. `python 01_complete_setup.py`
    2. `python 02_train.py`
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Assessment Settings")
    
    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher values = stricter flood detection"
    )
    
    land_use_type = st.selectbox(
        "Land Use Type",
        options=['mixed', 'urban', 'agricultural'],
        help="Select the primary land use in the area"
    )
    
    st.markdown("---")
    
    st.header("💰 Damage Parameters")
    
    cost_per_hectare = st.number_input(
        "Cost per Hectare (USD)",
        min_value=1000,
        max_value=500000,
        value=50000,
        step=5000,
        help="Estimated economic value per flooded hectare"
    )
    
    population_density = st.number_input(
        "Population Density (per km²)",
        min_value=10,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of people per square kilometer"
    )
    
    # Update predictor parameters
    predictor.damage_params['cost_per_hectare'] = cost_per_hectare
    predictor.damage_params['population_density'] = population_density
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.markdown("""
    This enhanced system provides:
    - 🗺️ Flood extent mapping
    - 💰 Economic damage estimation
    - 🏘️ Population affected
    - 🌾 Agricultural impact
    - 🏗️ Infrastructure damage
    - 📊 Recovery recommendations
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Satellite Image")
    
    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a satellite image for comprehensive flood damage assessment"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width='stretch')
        
        # Analyze button
        if st.button("🔍 Run Complete Damage Assessment", type="primary"):
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            temp_path = None
            
            try:
                status_text.text("📸 Processing image...")
                progress_bar.progress(10)
                
                # Create temp file
                temp_path = os.path.join(tempfile.gettempdir(), f"flood_{int(time.time())}.png")
                image.save(temp_path)
                progress_bar.progress(30)
                
                status_text.text("🔍 Detecting flood areas...")
                mask, confidence, original_size = predictor.predict(temp_path)
                progress_bar.progress(50)
                
                # Apply confidence threshold
                mask = (confidence > confidence_threshold).astype(np.uint8)
                
                status_text.text("📊 Calculating damage estimates...")
                progress_bar.progress(70)
                
                # Create overlay
                overlay = predictor.overlay(temp_path, mask)
                
                # Generate comprehensive report
                report = predictor.generate_report(mask, temp_path, original_size, land_use_type)
                progress_bar.progress(90)
                
                # Store results
                st.session_state['enhanced_results'] = {
                    'mask': mask,
                    'overlay': overlay,
                    'confidence': confidence,
                    'report': report,
                    'image': image,
                    'timestamp': datetime.now()
                }
                
                status_text.text("✅ Assessment complete!")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Comprehensive damage assessment complete! Scroll down for detailed report.")
                
            except Exception as e:
                st.error(f"❌ Error during assessment: {str(e)}")
                progress_bar.empty()
                status_text.empty()
            
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    for _ in range(3):
                        try:
                            gc.collect()
                            time.sleep(0.5)
                            os.unlink(temp_path)
                            break
                        except:
                            time.sleep(0.5)

with col2:
    if 'enhanced_results' in st.session_state:
        r = st.session_state['enhanced_results']
        report = r['report']
        
        st.subheader("📊 Quick Overview")
        
        # Severity indicator
        severity = report['flood_statistics']['severity']
        if severity in ['Catastrophic', 'Severe']:
            st.markdown(f'<div class="warning-box">⚠️ **Severity Level: {severity}** - Urgent action required</div>', unsafe_allow_html=True)
        elif severity == 'Moderate':
            st.markdown(f'<div class="info-box">📊 **Severity Level: {severity}** - Prepare for response</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">✅ **Severity Level: {severity}** - Monitor situation</div>', unsafe_allow_html=True)
        
        # Key metrics in columns
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Flood Coverage", f"{report['flood_statistics']['coverage_percentage']}%")
        with m2:
            st.metric("Flooded Area", f"{report['damage_assessment']['flooded_area_km2']} km²")
        with m3:
            st.metric("Est. Damage", f"${report['damage_assessment']['estimated_damage_usd']:,.0f}")

# Full results section (spans both columns)
if 'enhanced_results' in st.session_state:
    st.markdown("---")
    st.subheader("📋 Comprehensive Damage Assessment Report")
    
    r = st.session_state['enhanced_results']
    report = r['report']
    
    # Create tabs for different aspects
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Visual Analysis", 
        "💰 Economic Impact", 
        "🏘️ Population & Infrastructure",
        "🌾 Agricultural Impact",
        "📈 Recommendations"
    ])
    
    with tab1:
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("**Original Image**")
            st.image(r['image'], width='stretch')
        
        with col_v2:
            st.markdown("**Flood Analysis Overlay**")
            if r['overlay'] is not None:
                st.image(r['overlay'], width='stretch')
            else:
                st.warning("Overlay not available")
        
        # Confidence map
        st.markdown("**Confidence Map**")
        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(r['confidence'][:, :, 0], cmap='hot', vmin=0, vmax=1)
        ax.set_title("Detection Confidence")
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Confidence')
        st.pyplot(fig)
        plt.close(fig)
    
    with tab2:
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            st.markdown("### 💰 Damage Breakdown")
            
            # Create damage breakdown chart
            damage_data = pd.DataFrame({
                'Category': ['Direct Damage', 'Infrastructure', 'Agricultural', 'Indirect Losses'],
                'Amount (USD)': [
                    report['damage_assessment']['estimated_damage_usd'] * 0.6,
                    report['damage_assessment']['estimated_damage_usd'] * 0.2,
                    report['damage_assessment']['estimated_damage_usd'] * 0.15,
                    report['damage_assessment']['estimated_damage_usd'] * 0.05
                ]
            })
            
            fig = px.pie(damage_data, values='Amount (USD)', names='Category', 
                        title='Damage Distribution',
                        color_discrete_sequence=px.colors.sequential.Reds_r)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_e2:
            st.markdown("### 📊 Damage Statistics")
            
            st.markdown(f"""
            - **Total Estimated Damage:** ${report['damage_assessment']['estimated_damage_usd']:,.0f}
            - **Damage Category:** {report['damage_assessment']['damage_category']}
            - **Flooded Area:** {report['damage_assessment']['flooded_area_km2']} km²
            - **Flooded Hectares:** {report['damage_assessment']['flooded_hectares']} ha
            - **Cost per Hectare:** ${predictor.damage_params['cost_per_hectare']:,.0f}
            """)
            
            # Damage severity gauge - FIXED VERSION
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=report['damage_assessment']['estimated_damage_usd'],
                title={'text': "Damage Severity (USD)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, report['damage_assessment']['estimated_damage_usd'] * 1.5]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 500000], 'color': "lightgray"},
                        {'range': [500000, 2000000], 'color': "gray"},
                        {'range': [2000000, report['damage_assessment']['estimated_damage_usd'] * 1.5], 'color': "darkgray"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("### 👥 Population Impact")
            
            st.markdown(f"""
            - **Population Affected:** {report['damage_assessment']['population_affected']:,} people
            - **Population Density:** {predictor.damage_params['population_density']} people/km²
            - **Buildings Affected:** {report['damage_assessment']['buildings_affected']} structures
            """)
            
            # Population impact chart
            pop_data = pd.DataFrame({
                'Status': ['Affected', 'Unaffected'],
                'People': [
                    report['damage_assessment']['population_affected'],
                    max(0, 10000 - report['damage_assessment']['population_affected'])
                ]
            })
            fig = px.bar(pop_data, x='Status', y='People', color='Status',
                        color_discrete_map={'Affected': 'red', 'Unaffected': 'green'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col_p2:
            st.markdown("### 🏗️ Infrastructure Impact")
            
            infra = report['infrastructure_impact']
            st.markdown(f"""
            - **Total Roads:** {infra['total_roads_km']} km
            - **Flooded Roads:** {infra['flooded_roads_km']} km
            - **Roads Affected:** {infra['roads_affected_percent']}%
            - **Bridges Affected:** {infra['bridges_affected']}
            """)
            
            # Infrastructure damage chart
            infra_data = pd.DataFrame({
                'Type': ['Roads', 'Bridges'],
                'Affected': [infra['flooded_roads_km'], infra['bridges_affected']],
                'Total': [infra['total_roads_km'], infra['bridges_affected'] * 2]
            })
            fig = go.Figure(data=[
                go.Bar(name='Affected', x=infra_data['Type'], y=infra_data['Affected'], marker_color='red'),
                go.Bar(name='Total', x=infra_data['Type'], y=infra_data['Total'], marker_color='lightgray')
            ])
            fig.update_layout(barmode='group', title="Infrastructure Damage")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("### 🌾 Agricultural Impact")
            
            agri = report['agricultural_impact']
            st.markdown(f"""
            - **Agricultural Area:** {agri['agricultural_area_percent']}% of total
            - **Crop Damage:** {agri['crop_damage_percent']}% of crops
            - **Estimated Crop Loss:** ${agri['estimated_crop_loss_usd']:,.0f}
            - **Recovery Time:** {agri['recovery_time_months']} months
            """)
        
        with col_a2:
            # Crop damage gauge - FIXED VERSION
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=agri['crop_damage_percent'],
                title={'text': "Crop Damage %"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 🎯 Recommended Actions")
        
        recommendations = report['recommendations']
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        st.markdown("---")
        
        st.markdown("### 📅 Recovery Timeline")
        
        # Create recovery timeline
        recovery_months = max(
            report['agricultural_impact']['recovery_time_months'],
            3  # Minimum recovery time
        )
        
        timeline_data = pd.DataFrame({
            'Phase': ['Immediate Response', 'Short-term Recovery', 'Long-term Reconstruction'],
            'Duration (months)': [1, 3, recovery_months],
            'Start': [0, 1, 4],
            'End': [1, 4, 4 + recovery_months]
        })
        
        fig = go.Figure()
        for i, row in timeline_data.iterrows():
            fig.add_trace(go.Bar(
                name=row['Phase'],
                x=[row['Duration (months)']],
                y=['Timeline'],
                orientation='h',
                marker=dict(
                    color=['red', 'orange', 'green'][i],
                    line=dict(color='black', width=1)
                ),
                text=row['Phase'],
                textposition='inside',
                hoverinfo='text',
                base=row['Start']
            ))
        
        fig.update_layout(
            title="Recovery Timeline (months)",
            barmode='stack',
            height=200,
            showlegend=True,
            xaxis_title="Months"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Export report button
        st.markdown("---")
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp2:
            if st.button("📥 Download Full Assessment Report"):
                # Create detailed report text
                report_text = f"""
                FLOOD DAMAGE ASSESSMENT REPORT
                ===============================
                Generated: {report['timestamp']}
                Image: {report['image_analyzed']}
                
                FLOOD STATISTICS
                ----------------
                Coverage: {report['flood_statistics']['coverage_percentage']}%
                Severity: {report['flood_statistics']['severity']}
                Flooded Area: {report['damage_assessment']['flooded_area_km2']} km²
                
                DAMAGE ASSESSMENT
                -----------------
                Estimated Damage: ${report['damage_assessment']['estimated_damage_usd']:,.0f}
                Damage Category: {report['damage_assessment']['damage_category']}
                Buildings Affected: {report['damage_assessment']['buildings_affected']}
                Population Affected: {report['damage_assessment']['population_affected']}
                
                INFRASTRUCTURE IMPACT
                ---------------------
                Flooded Roads: {report['infrastructure_impact']['flooded_roads_km']} km
                Roads Affected: {report['infrastructure_impact']['roads_affected_percent']}%
                Bridges Affected: {report['infrastructure_impact']['bridges_affected']}
                
                AGRICULTURAL IMPACT
                -------------------
                Crop Damage: {report['agricultural_impact']['crop_damage_percent']}%
                Crop Loss: ${report['agricultural_impact']['estimated_crop_loss_usd']:,.0f}
                Recovery Time: {report['agricultural_impact']['recovery_time_months']} months
                
                RECOMMENDATIONS
                ---------------
                {chr(10).join(f'- {rec}' for rec in report['recommendations'])}
                """
                
                st.download_button(
                    label="Download Report",
                    data=report_text,
                    file_name=f"flood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>Enhanced Flood Damage Assessment System | AI-Powered | Real-time Analysis</div>",
    unsafe_allow_html=True
)