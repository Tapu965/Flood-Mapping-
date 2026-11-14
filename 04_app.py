"""
STREAMLIT WEB APP - RUN AFTER TRAINING
Run: streamlit run 04_app.py
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

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.predictor import FloodPredictor

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Flood Mapping System",
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
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
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

st.markdown('<h1 class="main-header">🌊 Flood Mapping System</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = os.path.join(BASE_DIR, 'models', 'trained_models', 'flood_model.h5')
    if os.path.exists(model_path):
        try:
            with st.spinner("Loading model..."):
                predictor = FloodPredictor('config.yaml', 'models/trained_models/flood_model.h5')
                return predictor
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

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
    st.header("ℹ️ About")
    st.markdown("""
    This system uses a U-Net deep learning model to detect flooded areas in satellite images.
    
    **Features:**
    - Flood area detection
    - Coverage percentage
    - Severity classification
    - Visual overlay
    """)
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width='stretch')  # FIXED: use width='stretch'
        
        if st.button("🔍 Analyze Flood Damage", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Saving image...")
                progress_bar.progress(20)
                
                # Create temp file
                temp_path = os.path.join(tempfile.gettempdir(), f"flood_{int(time.time())}.png")
                image.save(temp_path)
                progress_bar.progress(40)
                
                # Make prediction
                status_text.text("Running analysis...")
                mask, confidence = predictor.predict(temp_path)
                mask = (confidence > confidence_threshold).astype(np.uint8)
                progress_bar.progress(70)
                
                # Create overlay
                overlay = predictor.overlay(temp_path, mask)
                progress_bar.progress(90)
                
                # Get statistics
                stats = predictor.get_stats(mask)
                
                # Store results
                st.session_state['results'] = {
                    'mask': mask,
                    'overlay': overlay,
                    'stats': stats,
                    'confidence': confidence,
                    'image': image
                }
                
                status_text.text("Complete!")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                progress_bar.empty()
                status_text.empty()
            
            finally:
                # Clean up
                for _ in range(3):
                    try:
                        if os.path.exists(temp_path):
                            gc.collect()
                            time.sleep(0.5)
                            os.unlink(temp_path)
                            break
                    except:
                        time.sleep(0.5)

with col2:
    if 'results' in st.session_state:
        st.subheader("📊 Results")
        r = st.session_state['results']
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Coverage", f"{r['stats']['percentage']:.1f}%")
        
        with col_m2:
            severity = r['stats']['severity']
            st.metric("Severity", severity)
        
        with col_m3:
            # FIXED: This now works because predictor.py returns flooded_pixels
            st.metric("Flooded Pixels", f"{r['stats']['flooded_pixels']:,}")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Mask", "Confidence", "Overlay"])
        
        with tab1:
            fig, ax = plt.subplots()
            ax.imshow(r['mask'][:, :, 0], cmap='Blues')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            fig, ax = plt.subplots()
            ax.imshow(r['confidence'][:, :, 0], cmap='hot')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        
        with tab3:
            if r['overlay'] is not None:
                st.image(r['overlay'], caption="Flood Areas in Red", width='stretch')  # FIXED
            else:
                st.warning("No overlay")

# Instructions
if 'results' not in st.session_state:
    with col2:
        st.info("👆 Upload an image and click 'Analyze'")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Flood Mapping System | CPU-Optimized</div>",
    unsafe_allow_html=True
)