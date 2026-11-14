"""
BEFORE/AFTER FLOOD ANALYSIS APP
Upload pre-flood and post-flood images to detect NEW flood water
Excludes permanent water bodies (rivers, lakes) from damage calculation
Run: streamlit run 06_before_after_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
import sys
import time
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.change_detector import ChangeDetector
from inference.enhanced_predictor import EnhancedFloodPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(
    page_title="Before/After Flood Analysis",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1E88E5;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #0D47A1;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .info-box {
        background: #e3f2fd;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌊 Before/After Flood Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare pre-flood and post-flood images to detect NEW flood water • Excludes permanent water bodies</p>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    detector = ChangeDetector('config.yaml', 'models/trained_models/flood_model_200.h5')
    predictor = EnhancedFloodPredictor('config.yaml', 'models/trained_models/flood_model_200.h5')
    return detector, predictor

detector, predictor = load_models()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    with st.expander("🔍 Detection Settings", expanded=True):
        water_threshold = st.slider(
            "Water Detection Sensitivity",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Lower = more sensitive water detection"
        )
        
        min_flood_size = st.number_input(
            "Minimum Flood Patch Size (pixels)",
            min_value=10,
            max_value=1000,
            value=50,
            step=10,
            help="Remove flood patches smaller than this"
        )
    
    with st.expander("💰 Damage Parameters", expanded=True):
        cost_per_hectare = st.number_input(
            "Cost per Hectare (USD)",
            min_value=1000,
            max_value=500000,
            value=50000,
            step=5000
        )
        
        population_density = st.number_input(
            "Population Density (per km²)",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100
        )
    
    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. Upload **BEFORE** flood image
    2. Upload **AFTER** flood image
    3. System detects water in both images
    4. **Permanent water** (rivers, lakes) is identified
    5. **New flood water** is calculated as difference
    6. Damage assessed only on **new flood areas**
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload BEFORE Flood Image")
    before_file = st.file_uploader(
        "Choose pre-flood satellite image",
        type=['jpg', 'jpeg', 'png'],
        key="before_upload"
    )
    
    if before_file:
        before_img = Image.open(before_file)
        st.image(before_img, caption="Before Flood", use_container_width=True)

with col2:
    st.markdown("### 📤 Upload AFTER Flood Image")
    after_file = st.file_uploader(
        "Choose post-flood satellite image",
        type=['jpg', 'jpeg', 'png'],
        key="after_upload"
    )
    
    if after_file:
        after_img = Image.open(after_file)
        st.image(after_img, caption="After Flood", use_container_width=True)

# Analyze button
if before_file and after_file:
    st.markdown("---")
    
    if st.button("🔍 Analyze Flood Change", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.markdown("🔄 **Step 1/4:** Loading images...")
            progress_bar.progress(25)
            
            # Save temp files
            before_path = os.path.join(tempfile.gettempdir(), f"before_{int(time.time())}.png")
            after_path = os.path.join(tempfile.gettempdir(), f"after_{int(time.time())}.png")
            
            before_img.save(before_path)
            after_img.save(after_path)
            
            status_text.markdown("🔄 **Step 2/4:** Detecting water in both images...")
            progress_bar.progress(50)
            
            # Update detector settings
            detector.min_flood_area = min_flood_size
            
            # Detect change
            results =detector.analyze_from_paths(before_path, after_path)
            
            status_text.markdown("🔄 **Step 3/4:** Calculating flood statistics...")
            progress_bar.progress(75)
            
            # Store results
            st.session_state['before_after_results'] = results
            
            status_text.markdown("🔄 **Step 4/4:** Generating report...")
            progress_bar.progress(100)
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Analysis complete! Scroll down for detailed report.")
            
            # Clean up
            os.unlink(before_path)
            os.unlink(after_path)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Display results
if 'before_after_results' in st.session_state:
    results = st.session_state['before_after_results']
    stats = results['statistics']
    
    st.markdown("---")
    st.markdown("## 📊 Flood Change Analysis Report")
    
    # Key metrics
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric(
            "New Flood Area",
            f"{stats['flood_percentage']:.1f}%",
            delta=f"{stats['flood_area_km2']} km²"
        )
    
    with m2:
        st.metric(
            "Permanent Water",
            f"{stats['permanent_water_percentage']:.1f}%",
            delta=f"{stats['permanent_water_area_km2']} km²"
        )
    
    with m3:
        # Calculate damage only on new flood
        flood_hectares = stats['flood_area_km2'] * 100
        damage = flood_hectares * 50000 * 2  # mixed land use
        st.metric(
            "Estimated Damage",
            f"${damage:,.0f}",
            delta=f"{flood_hectares:.0f} hectares"
        )
    
    with m4:
        st.metric(
            "Severity",
            stats['severity']
        )
    
    # Warning about permanent water exclusion
    st.markdown(f"""
    <div class="success-box">
    ✅ <b>Permanent Water Excluded:</b> {stats['permanent_water_percentage']:.1f}% of the image 
    ({stats['permanent_water_area_km2']} km²) is permanent water (rivers, lakes) and was 
    <b>NOT counted</b> as flood damage.
    </div>
    """, unsafe_allow_html=True)
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Before/After Comparison",
        "🗺️ Flood Map",
        "📊 Statistics",
        "💰 Damage Assessment"
    ])
    
    with tab1:
        st.markdown("### Before vs After Comparison")
        st.image(
            results['comparison_visualization'],
            caption="Left to Right: Before | After | New Flood (Red) | Permanent Water (Blue)",
            use_container_width=True
        )
        
        st.markdown("""
        **Legend:**
        - 🟥 **Red** = New flood water (counted for damage)
        - 🟦 **Blue** = Permanent water bodies (excluded from damage)
        """)
    
    with tab2:
        st.markdown("### Flood Detection Maps")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**New Flood Mask**")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(results['flood_mask'], cmap='Reds', vmin=0, vmax=1)
            ax.set_title("New Flood Water")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        with col_b:
            st.markdown("**Permanent Water Mask**")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(results['permanent_water'], cmap='Blues', vmin=0, vmax=1)
            ax.set_title("Permanent Water Bodies")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
    
    with tab3:
        st.markdown("### Flood Statistics")
        
        # Statistics table
        stats_df = pd.DataFrame([
            {'Metric': 'Total Image Area', 'Value': f"{stats['total_pixels']:,} pixels"},
            {'Metric': 'New Flood Water', 'Value': f"{stats['flood_pixels']:,} pixels"},
            {'Metric': 'Permanent Water', 'Value': f"{stats['permanent_water_pixels']:,} pixels"},
            {'Metric': 'Flood Percentage', 'Value': f"{stats['flood_percentage']:.2f}%"},
            {'Metric': 'Permanent Water Percentage', 'Value': f"{stats['permanent_water_percentage']:.2f}%"},
            {'Metric': 'Flood Area', 'Value': f"{stats['flood_area_km2']:.2f} km²"},
            {'Metric': 'Permanent Water Area', 'Value': f"{stats['permanent_water_area_km2']:.2f} km²"},
            {'Metric': 'Severity', 'Value': stats['severity']},
        ])
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Pie chart
        fig = px.pie(
            values=[stats['flood_pixels'], stats['permanent_water_pixels'], 
                    stats['total_pixels'] - stats['flood_pixels'] - stats['permanent_water_pixels']],
            names=['New Flood', 'Permanent Water', 'Dry Land'],
            title='Area Distribution',
            color_discrete_sequence=['#ff4b4b', '#1E88E5', '#28a745']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Damage Assessment (New Flood Only)")
        
        # Calculate damage based on new flood only
        flood_hectares = stats['flood_area_km2'] * 100
        
        # Damage multipliers by land type (simplified)
        urban_damage = flood_hectares * 0.4 * 50000 * 5
        agri_damage = flood_hectares * 0.3 * 50000 * 1.5
        forest_damage = flood_hectares * 0.2 * 50000 * 0.3
        other_damage = flood_hectares * 0.1 * 50000 * 0.5
        
        total_damage = urban_damage + agri_damage + forest_damage + other_damage
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            st.metric("Total Damage", f"${total_damage:,.0f}")
        with col_d2:
            population = stats['flood_area_km2'] * 1000
            st.metric("Population Affected", f"{population:,.0f}")
        with col_d3:
            buildings = stats['flood_area_km2'] * 50
            st.metric("Buildings Damaged", f"{buildings:,.0f}")
        
        # Damage breakdown chart
        damage_data = pd.DataFrame([
            {'Category': 'Urban', 'Damage (USD)': urban_damage},
            {'Category': 'Agriculture', 'Damage (USD)': agri_damage},
            {'Category': 'Forest', 'Damage (USD)': forest_damage},
            {'Category': 'Other', 'Damage (USD)': other_damage},
        ])
        
        fig = px.bar(damage_data, x='Category', y='Damage (USD)',
                    title='Damage by Land Use Category',
                    color='Category',
                    color_discrete_sequence=px.colors.sequential.Reds_r)
        st.plotly_chart(fig, use_container_width=True)
        
        # Important note
        st.markdown(f"""
        <div class="warning-box">
        ⚠️ <b>Important:</b> This damage estimate is based ONLY on NEW flood water 
        ({stats['flood_area_km2']} km²). Permanent water bodies 
        ({stats['permanent_water_area_km2']} km²) are excluded because flooding 
        an existing river or lake does not cause additional damage.
        </div>
        """, unsafe_allow_html=True)

# Instructions
else:
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <h4>📋 How to Use</h4>
    <ol>
        <li>Upload a <b>BEFORE</b> flood satellite image (pre-flood)</li>
        <li>Upload an <b>AFTER</b> flood satellite image (post-flood)</li>
        <li>Click <b>"Analyze Flood Change"</b></li>
        <li>View results showing only NEW flood areas</li>
        <li>Permanent water bodies (rivers, lakes) are automatically excluded</li>
    </ol>
    <p><b>Why Before/After?</b> Single image analysis counts rivers and lakes as floods. 
    Before/After analysis identifies only NEW water that appeared due to flooding.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; padding: 1rem;'>Before/After Flood Analysis | Permanent Water Exclusion | AI-Powered</div>",
    unsafe_allow_html=True
)