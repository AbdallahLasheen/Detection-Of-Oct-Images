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
        st.subheader("📤 Scan Upload")
        uploaded_file = st.file_uploader("Upload Retinal Scan (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Patient OCT Scan", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        if uploaded_file:
            if st.button("EXECUTE HYBRID DIAGNOSIS"):
                with st.spinner('Fusing CNN & Transformer Features...'):
                    input_tensor = preprocess_image(img)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        conf, idx = torch.max(probs, 0)
                    
                    result = CLASS_NAMES[idx]
                    score = conf.item() * 100
                    
                    # Store for analysis tab
                    st.session_state['hybrid_probs'] = probs.numpy()
                    
                    # Colors based on health status
                    status_color = "#10b981" if result == "NORMAL" else "#ef4444"
                    
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #1e293b;'>Diagnostic Result</h3>
                            <div class='res-box' style='background-color: {status_color};'>
                                {result}
                            </div>
                            <div style='margin-top: 20px; border-left: 4px solid {status_color}; padding-left: 15px;'>
                                <p style='font-size: 14px; color: #475569;'><b>Clinical Description:</b><br>{CLASS_DESC[result]}</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Detection Confidence", f"{score:.2f}%", delta="Hybrid Confirmed")
        else:
            st.warning("Please upload a retinal scan to begin the diagnostic process.")

with tab2:
    if 'hybrid_probs' in st.session_state:
        st.subheader("Diagnostic Probability Analysis")
        current_probs = st.session_state['hybrid_probs']
        df_chart = pd.DataFrame({
            'Condition': CLASS_NAMES,
            'Probability (%)': current_probs * 100
        })
        
        fig = px.bar(df_chart, x='Condition', y='Probability (%)', 
                     text_auto='.2f', color='Condition',
                     color_discrete_map={'NORMAL': '#10b981', 'CNV': '#f59e0b', 'DME': '#3b82f6', 'DRUSEN': '#ef4444'},
                     template='plotly_white')
        
        fig.update_layout(showlegend=False, height=500, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Once a diagnosis is performed, statistical data will appear here.")

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
