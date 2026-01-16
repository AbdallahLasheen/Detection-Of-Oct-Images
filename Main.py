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

# 5️⃣ Image Preprocessing Function
def preprocess_image(pil_image):
    """Preprocess PIL image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)  # Add batch dimension

# 6️⃣ Batch Processing Function
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
    col_up, col_res = st.columns([1, 1.2], gap="large")
    
    with col_up:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("📤 Select Multiple Scans")
        uploaded_files = st.file_uploader(
            "Upload Multiple Retinal Scans (JPG/PNG)", 
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ Selected {len(uploaded_files)} images")
            # Preview grid
            st.subheader("Preview")
            preview_cols = st.columns(min(3, len(uploaded_files)))
            for idx, uploaded_file in enumerate(uploaded_files[:3]):
                with preview_cols[idx]:
                    img = Image.open(uploaded_file).convert("RGB")
                    st.image(img, caption=uploaded_file.name, use_container_width=True)
            
            if len(uploaded_files) > 3:
                st.caption(f"...and {len(uploaded_files) - 3} more images")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        if uploaded_files:
            if st.button("EXECUTE HYBRID DIAGNOSIS", use_container_width=True):
                with st.spinner('Processing images...'):
                    # Load all images
                    images = [Image.open(f).convert("RGB") for f in uploaded_files]
                    results = process_batch_images(images)
                    st.session_state['batch_results'] = results
                    st.session_state['image_files'] = uploaded_files
                    
                    # Display detailed results
                    st.subheader("🔍 Diagnostic Results")
                    cols_per_row = 3
                    result_cols = st.columns(cols_per_row)
                    
                    for i, (result, uploaded_file) in enumerate(zip(results, uploaded_files)):
                        col_idx = i % cols_per_row
                        
                        with result_cols[col_idx]:
                            if 'error' not in result:
                                img = Image.open(uploaded_file).convert("RGB")
                                st.image(img, use_container_width=True)
                                
                                diagnosis = result['class']
                                confidence = result['confidence']
                                probabilities = result['probabilities']
                                status_color = "#10b981" if diagnosis == "NORMAL" else "#ef4444"
                                icon = "✅" if diagnosis == "NORMAL" else "⚠️"
                                
                                # Diagnosis Box
                                st.markdown(f"""
                                    <div style='background-color: {status_color}; padding: 16px; border-radius: 10px; text-align: center; color: white; margin-bottom: 12px;'>
                                        <p style='margin: 0; font-size: 12px; opacity: 0.9;'>🩺 DIAGNOSIS</p>
                                        <p style='margin: 8px 0 0 0; font-size: 20px; font-weight: bold;'>{icon} {diagnosis}</p>
                                        <p style='margin: 8px 0 0 0; font-size: 13px;'>Confidence: <b>{confidence:.1f}%</b></p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Clinical Description
                                st.markdown(f"""
                                    <div style='background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 12px; border-radius: 6px; margin-bottom: 12px;'>
                                        <p style='margin: 0; font-size: 12px; color: #065f46;'><b>📋 Clinical Notes</b></p>
                                        <p style='margin: 8px 0 0 0; font-size: 11px; color: #047857; line-height: 1.5;'>{CLASS_DESC[diagnosis]}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error(f"❌ Error: {result['error']}")
                        
                        if (i + 1) % cols_per_row == 0 and i + 1 < len(results):
                            result_cols = st.columns(cols_per_row)
        else:
            st.warning("Please upload images to begin diagnosis.")

with tab2:
    if 'batch_results' in st.session_state:
        results = st.session_state['batch_results']
        uploaded_files = st.session_state['image_files']
        st.subheader("📊 Detailed Analysis & Statistics")
        
        total_images = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", total_images)
        col2.metric("Successfully Processed", successful)
        col3.metric("Errors", total_images - successful)
        
        st.markdown("---")
        
        # Overall probability distribution
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
        
        st.markdown("---")
        st.subheader("🔍 Individual Image Analysis")
        
        # Display each image with its detailed analysis
        for i, (result, uploaded_file) in enumerate(zip(results, uploaded_files)):
            if 'error' not in result:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    img = Image.open(uploaded_file).convert("RGB")
                    st.image(img, use_container_width=True)
                    
                with col2:
                    diagnosis = result['class']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    status_color = "#10b981" if diagnosis == "NORMAL" else "#ef4444"
                    icon = "✅" if diagnosis == "NORMAL" else "⚠️"
                    
                    # Diagnosis Box
                    st.markdown(f"""
                        <div style='background-color: {status_color}; padding: 16px; border-radius: 10px; text-align: center; color: white; margin-bottom: 12px;'>
                            <p style='margin: 0; font-size: 12px; opacity: 0.9;'>🩺 DIAGNOSIS</p>
                            <p style='margin: 8px 0 0 0; font-size: 20px; font-weight: bold;'>{icon} {diagnosis}</p>
                            <p style='margin: 8px 0 0 0; font-size: 13px;'>Confidence: <b>{confidence:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Clinical Description
                    st.markdown(f"""
                        <div style='background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 12px; border-radius: 6px; margin-bottom: 12px;'>
                            <p style='margin: 0; font-size: 12px; color: #065f46;'><b>📋 Clinical Notes</b></p>
                            <p style='margin: 8px 0 0 0; font-size: 11px; color: #047857; line-height: 1.5;'>{CLASS_DESC[diagnosis]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability Table
                    prob_data = {
                        'Condition': CLASS_NAMES,
                        'Probability': [f"{p*100:.2f}%" for p in probabilities]
                    }
                    df_prob = pd.DataFrame(prob_data)
                    st.dataframe(df_prob, use_container_width=True, hide_index=True)
                
                # Bar Chart
                fig = px.bar(y=probabilities*100, x=CLASS_NAMES, 
                            orientation='v', 
                            color=CLASS_NAMES,
                            color_discrete_sequence=['#ef4444', '#ef4444', '#ef4444', '#10b981'],
                            labels={'y': 'Probability (%)', 'x': 'Condition'})
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True, key=f"analysis_chart_{i}")
                st.markdown("---")
    else:
        st.info("Once batch diagnosis is performed, statistical data will appear here.")

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
