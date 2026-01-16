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
from pathlib import Path

# تم حذف مكتبة tkinter لأنها تسبب انهيار التطبيق على السيرفر

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

# 3️⃣ Define the Hybrid Architecture
class OCTelligenceHybrid(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTelligenceHybrid, self).__init__()
        self.dense_backbone = timm.create_model('densenet121', pretrained=False, num_classes=0)
        self.swin_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
        
        combined_size = 1024 + 1024 
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
        weights_path = "best_hybrid_octelligence.pth"
        if not os.path.exists(weights_path):
            st.error(f"Error: Model weights file '{weights_path}' not found!")
            return None
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

# 5️⃣ Preprocessing & Processing Functions
def preprocess_image(image):
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tfms(image).unsqueeze(0)

def process_batch_images(images_list):
    results = []
    for img in images_list:
        try:
            input_tensor = preprocess_image(img)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, idx = torch.max(probs, 0)
            
            results.append({
                'class': CLASS_NAMES[idx],
                'confidence': conf.item() * 100,
                'probabilities': probs.numpy()
            })
        except Exception as e:
            results.append({'error': str(e)})
    return results

# 6️⃣ Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3534/3534280.png", width=80)
    st.title("OCTelligence Pro")
    st.markdown("---")
    st.subheader("📊 Model Architecture")
    st.info("Hybrid System: DenseNet121 + Swin Transformer")
    st.write("🎯 **Accuracy:** 98%+")
    st.write("🔬 **Input:** 224x224x3")
    st.markdown("---")
    st.caption("AI-Powered Diagnostic Assistance")

# 7️⃣ Main UI
st.title("👁️ Hybrid Retinal Diagnostic Hub")
st.markdown("<p style='color: #64748b; font-size: 18px;'>Deep Learning Analysis for Optical Coherence Tomography (OCT)</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Real-time Diagnosis", "📊 Analysis Statistics", "📄 System Info"])

with tab1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📸 Upload OCT Images")
    
    # استبدال اختيار المجلد (Folder) بخاصية رفع ملفات متعددة (Multiple Files)
    uploaded_files_raw = st.file_uploader(
        "Select one or more OCT scan images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_files_raw:
        col_up, col_res = st.columns([1, 1.2], gap="large")
        
        with col_up:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader(f"✅ Loaded {len(uploaded_files_raw)} images")
            
            # Preview grid
            preview_images = []
            image_names = []
            for uploaded_file in uploaded_files_raw[:3]:
                img = Image.open(uploaded_file).convert("RGB")
                preview_images.append(img)
                image_names.append(uploaded_file.name)
            
            preview_cols = st.columns(min(3, len(preview_images)))
            for idx, img in enumerate(preview_images):
                with preview_cols[idx]:
                    st.image(img, caption=image_names[idx], use_container_width=True)
            
            if len(uploaded_files_raw) > 3:
                st.caption(f"...and {len(uploaded_files_raw) - 3} more")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_res:
            if st.button("Execute Hybrid Diagnosis", use_container_width=True):
                with st.spinner('Analyzing images...'):
                    # Load all images
                    all_images = [Image.open(f).convert("RGB") for f in uploaded_files_raw]
                    results = process_batch_images(all_images)
                    
                    st.session_state['batch_results'] = results
                    st.session_state['image_names'] = [f.name for f in uploaded_files_raw]
                    st.session_state['uploaded_images'] = all_images

                    # Summary Metrics
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.subheader("Diagnostic Results Summary")
                    counts = {name: 0 for name in CLASS_NAMES}
                    for r in results:
                        if 'class' in r: counts[r['class']] += 1
                    
                    m_cols = st.columns(4)
                    for i, name in enumerate(CLASS_NAMES):
                        m_cols[i].metric(name, counts[name])
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Detailed Grid
                    st.subheader("Detailed Analysis")
                    cols_per_row = 2
                    for i in range(0, len(results), cols_per_row):
                        grid_cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < len(results):
                                with grid_cols[j]:
                                    res = results[i+j]
                                    st.image(all_images[i+j], use_container_width=True)
                                    diag = res['class']
                                    conf = res['confidence']
                                    
                                    colors = {"NORMAL": "#10b981", "CNV": "#f59e0b", "DME": "#3b82f6", "DRUSEN": "#ef4444"}
                                    st.markdown(f"""
                                        <div style='background-color: {colors.get(diag, "#333")}; padding: 10px; border-radius: 8px; text-align: center; color: white;'>
                                            <b>{diag} ({conf:.1f}%)</b>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    st.caption(f"{CLASS_DESC[diag]}")

with tab2:
    if 'batch_results' in st.session_state:
        results = st.session_state['batch_results']
        df_chart = pd.DataFrame({
            'Condition': CLASS_NAMES,
            'Count': [sum(1 for r in results if r.get('class') == name) for name in CLASS_NAMES]
        })
        fig = px.pie(df_chart, values='Count', names='Condition', color='Condition',
                     color_discrete_map={'NORMAL': '#10b981', 'CNV': '#f59e0b', 'DME': '#3b82f6', 'DRUSEN': '#ef4444'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run diagnosis to see statistics.")

with tab3:
    st.markdown("""
    ### Technical Specification
    This dashboard utilizes a **Hybrid Neural Network**:
    1. **DenseNet121**: Local features (fluid, lesions).
    2. **Swin Transformer**: Global context.
    """)

st.markdown("---")
st.caption("© 2025 OCTelligence AI Systems | Medical Imaging Division")
