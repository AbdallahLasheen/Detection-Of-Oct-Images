# ===============================================================
# Project: OCTelligence Pro | Hybrid AI Diagnostic Hub
# Model: DenseNet121 + Swin Transformer (Hybrid)
# Dataset: Kermany2018 (OCT Retinal Scans)
# ===============================================================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd
import plotly.express as px
import time
import os

# 1️⃣ Page Configuration
st.set_page_config(
    page_title="OCTelligence Pro | Hybrid AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2️⃣ Professional Medical UI Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #edf2f7;
        margin-bottom: 20px;
    }
    .res-box {
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: 800;
        text-align: center;
        font-size: 26px;
        margin-top: 10px;
        text-transform: uppercase;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
    }
    [data-testid="stSidebar"] { background-color: #0f172a; }
    [data-testid="stSidebar"] * { color: white; }
    </style>
    """, unsafe_allow_html=True)

# 3️⃣ Define the Hybrid Architecture (Must match training structure)
class OCTelligenceHybrid(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTelligenceHybrid, self).__init__()
        # Branch 1: DenseNet121
        self.dense_backbone = timm.create_model('densenet121', pretrained=False, num_classes=0)
        # Branch 2: Swin Transformer
        self.swin_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
        
        # Fusion Head
        combined_size = 1024 + 1024 # Features from both models
        self.custom_head = nn.Sequential(
            nn.Linear(combined_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.dense_backbone(x)
        f2 = self.swin_backbone(x)
        combined = torch.cat((f1, f2), dim=1) 
        return self.custom_head(combined)

# 4️⃣ Model Loading Logic
@st.cache_resource
def load_hybrid_model():
    model = OCTelligenceHybrid(num_classes=4)
    try:
        # Loading weights to CPU for universal compatibility
        weights_path = "best_hybrid_octelligence.pth"
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Critical Error: Could not load weights. {e}")
        return None

model = load_hybrid_model()
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
CLASS_DESC = {
    'CNV': "Choroidal Neovascularization: Indicates abnormal vessel growth beneath the retina.",
    'DME': "Diabetic Macular Edema: Fluid accumulation caused by diabetic complications.",
    'DRUSEN': "Early indicators of Age-related Macular Degeneration (AMD).",
    'NORMAL': "Healthy Retinal structure detected. No visible pathologies."
}

# 5️⃣ Preprocessing Function
def preprocess_image(image):
    # Ensure image is converted to RGB (3 channels) even if grayscale
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tfms(image).unsqueeze(0)

# 5️⃣ Batch Processing Function
def process_batch_images(images_list):
    """Process multiple images and return results"""
    results = []
    for img in images_list:
        try:
            input_tensor = preprocess_image(img)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)
            
            result = {
                'class': CLASS_NAMES[idx],
                'confidence': conf.item() * 100,
                'probabilities': probs.numpy()
            }
            results.append(result)
        except Exception as e:
            results.append({'error': str(e)})
    
    return results

# 6️⃣ Sidebar - System Stats
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3534/3534280.png", width=80)
    st.title("OCTelligence Pro")
    st.markdown("---")
    st.subheader("📊 Model Architecture")
    st.info("Hybrid System: DenseNet121 + Swin Transformer")
    st.write("🎯 **Target Accuracy:** 98%+")
    st.write("🔬 **Input Size:** 224x224x3")
    st.markdown("---")
    st.caption("AI-Powered Diagnostic Assistance")

# 7️⃣ Main UI Header
st.title("👁️ Hybrid Retinal Diagnostic Hub")
st.markdown("<p style='color: #64748b; font-size: 18px;'>Deep Learning Analysis for Optical Coherence Tomography (OCT)</p>", unsafe_allow_html=True)

# 8️⃣ Tabs for Organization
tab1, tab2, tab3 = st.tabs(["🚀 Real-time Diagnosis", "📊 Analysis Statistics", "📄 System Info"])

with tab1:
    # Input mode selection
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📊 Select Input Source")
    
    input_mode = st.radio(
        "Choose how to provide images:",
        ["📸 Select Images"],
        horizontal=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_up, col_res = st.columns([1, 1.2], gap="large")
    
    image_names = []
    uploaded_files = []
    images_found = False
    webcam_image = None
    
    if input_mode == "📸 Select Images":
        with col_up:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("📸 Select Multiple Images")
            
            # File uploader for multiple images
            uploaded_file_list = st.file_uploader(
                "Upload OCT Scan Images (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="multiple_images_uploader"
            )
            
            if uploaded_file_list and len(uploaded_file_list) > 0:
                st.success(f"✅ Selected {len(uploaded_file_list)} images")
                
                # Preview grid
                st.subheader("Image Preview")
                preview_cols = st.columns(min(3, len(uploaded_file_list)))
                for idx, uploaded_file in enumerate(uploaded_file_list[:3]):
                    with preview_cols[idx]:
                        img = Image.open(uploaded_file).convert("RGB")
                        st.image(img, caption=uploaded_file.name, use_container_width=True)
                
                if len(uploaded_file_list) > 3:
                    st.caption(f"...and {len(uploaded_file_list) - 3} more images")
                
                # Store for processing
                image_names = [f.name for f in uploaded_file_list]
                uploaded_files = [Image.open(f).convert("RGB") for f in uploaded_file_list]
                images_found = True
            
            st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        if uploaded_files and len(uploaded_files) > 0:
            if st.button("Execute Hybrid Diagnosis", use_container_width=True, key="diagnose_btn"):
                with st.spinner('Processing images...'):
                    # Load images from folder paths or webcam
                    images = []
                    for img_path in uploaded_files:
                        try:
                            # Check if it's already a PIL Image (from webcam)
                            if isinstance(img_path, Image.Image):
                                images.append(img_path)
                            else:
                                # Load from file path
                                img = Image.open(img_path).convert("RGB")
                                images.append(img)
                        except Exception as e:
                            if isinstance(img_path, str):
                                st.error(f"Error loading {os.path.basename(img_path)}: {e}")
                            else:
                                st.error(f"Error processing image: {e}")
                    
                    if images:
                        results = process_batch_images(images)
                        st.session_state['batch_results'] = results
                        st.session_state['image_names'] = image_names
                        
                        # Display results summary
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.subheader("Diagnostic Results")
                        
                        # Summary statistics
                        condition_counts = {}
                        for result in results:
                            if 'error' not in result:
                                condition = result['class']
                                condition_counts[condition] = condition_counts.get(condition, 0) + 1
                        
                        col1, col2, col3, col4 = st.columns(4)
                        for idx, condition in enumerate(CLASS_NAMES):
                            count = condition_counts.get(condition, 0)
                            with [col1, col2, col3, col4][idx]:
                                st.metric(condition, count)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display results with images
                        st.subheader("Detailed Analysis")
                        
                        # Create grid for displaying images with results
                        cols_per_row = 3
                        result_cols = st.columns(cols_per_row)
                        
                        for i, (result, img_source) in enumerate(zip(results, uploaded_files)):
                            col_idx = i % cols_per_row
                            
                            with result_cols[col_idx]:
                                if 'error' not in result:
                                    # Display image
                                    if isinstance(img_source, Image.Image):
                                        img = img_source
                                    else:
                                        img = Image.open(img_source).convert("RGB")
                                    
                                    st.image(img, use_container_width=True)
                                    
                                    # Display diagnosis
                                    diagnosis = result['class']
                                    confidence = result['confidence']
                                    
                                    # Color coding based on diagnosis
                                    if diagnosis == "NORMAL":
                                        status_color = "#10b981"
                                        status_emoji = "✅"
                                    elif diagnosis == "CNV":
                                        status_color = "#f59e0b"
                                        status_emoji = "⚠️"
                                    elif diagnosis == "DME":
                                        status_color = "#3b82f6"
                                        status_emoji = "⚠️"
                                    else:  # DRUSEN
                                        status_color = "#ef4444"
                                        status_emoji = "⚠️"
                                    
                                    # Display diagnosis box
                                    st.markdown(f"""
                                        <div style='background-color: {status_color}; padding: 12px; border-radius: 8px; text-align: center; color: white;'>
                                            <p style='margin: 0; font-size: 14px;'><b>Diagnosis</b></p>
                                            <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold;'>{status_emoji} {diagnosis}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display confidence
                                    st.metric("Confidence", f"{confidence:.2f}%")
                                    
                                    # Display description
                                    st.caption(f"📄 {CLASS_DESC[diagnosis]}")
                                else:
                                    st.error(f"❌ Error processing image: {result['error']}")
                            
                            # Create new row after every cols_per_row items
                            if (i + 1) % cols_per_row == 0 and i + 1 < len(results):
                                result_cols = st.columns(cols_per_row)
        else:
            st.warning("Select a folder or capture from webcam to start diagnosis")

with tab2:
    if 'batch_results' in st.session_state:
        results = st.session_state['batch_results']
        image_names = st.session_state.get('image_names', [])
        
        st.subheader("Batch Analysis Statistics")
        
        # Summary statistics
        total_images = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        errors = total_images - successful
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", total_images)
        col2.metric("Successfully Processed", successful)
        col3.metric("Errors", errors)
        
        # Aggregate probability analysis
        if successful > 0:
            st.subheader("Overall Probability Distribution")
            aggregated_probs = [0] * len(CLASS_NAMES)
            
            for result in results:
                if 'error' not in result:
                    for i, prob in enumerate(result['probabilities']):
                        aggregated_probs[i] += prob
            
            aggregated_probs = [p / successful for p in aggregated_probs]
            
            df_chart = pd.DataFrame({
                'Condition': CLASS_NAMES,
                'Average Probability (%)': [p * 100 for p in aggregated_probs]
            })
            
            fig = px.bar(df_chart, x='Condition', y='Average Probability (%)', 
                         text_auto='.2f', color='Condition',
                         color_discrete_map={'NORMAL': '#10b981', 'CNV': '#f59e0b', 'DME': '#3b82f6', 'DRUSEN': '#ef4444'},
                         template='plotly_white')
            
            fig.update_layout(showlegend=False, height=500, title_x=0.5)
            st.plotly_chart(fig, use_container_width=True, key="aggregate_chart")
            
            # Per-image breakdown
            st.subheader("Individual Image Analysis")
            for idx, (result, img_name) in enumerate(zip(results, image_names)):
                if 'error' not in result:
                    with st.expander(f"📊 {img_name} - {result['class']} ({result['confidence']:.2f}%)"):
                        df_probs = pd.DataFrame({
                            'Condition': CLASS_NAMES,
                            'Probability (%)': [p * 100 for p in result['probabilities']]
                        })
                        fig_ind = px.bar(df_probs, x='Condition', y='Probability (%)', text_auto='.2f')
                        st.plotly_chart(fig_ind, use_container_width=True, key=f"chart_image_{idx}")
    else:
        st.info("Once a batch diagnosis is performed, statistical data will appear here.")

with tab3:
    st.markdown("""
    ### Technical Specification
    This dashboard utilizes a **Hybrid Neural Network** that leverages two distinct architectural strengths:
    
    1.  **DenseNet121 (CNN Branch):** Excels at identifying local pathological features (small lesions, fluid pockets).
    2.  **Swin Transformer (ViT Branch):** Captures long-range dependencies and global contextual information of the retinal layers.
    
    **System Metrics:**
    * **Loss Function:** Weighted Cross Entropy (to handle clinical class imbalance).
    * **Preprocessing:** Standardized to 224x224 pixels with ImageNet normalization.
    * **Performance:** High Sensitivity/Recall for CNV and DME detection.
    """)

# 9️⃣ Unified Footer
st.markdown("---")
f1, f2 = st.columns([4, 1])
with f1:
    st.caption("© 2025 OCTelligence AI Systems | Graduation Project | Medical Imaging Division")
with f2:
    st.caption("Status: 🔌 Hybrid Engine Ready")
