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

# 1️⃣ Page Configuration
st.set_page_config(
    page_title="OCTelligence Pro | Hybrid AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2️⃣ Professional Medical UI Styling (نفس الستايل الخاص بك)
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
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return model
        return None
    except: return None

model = load_hybrid_model()
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
CLASS_DESC = {
    'CNV': "Choroidal Neovascularization: Indicates abnormal vessel growth beneath the retina.",
    'DME': "Diabetic Macular Edema: Fluid accumulation caused by diabetic complications.",
    'DRUSEN': "Early indicators of Age-related Macular Degeneration (AMD).",
    'NORMAL': "Healthy Retinal structure detected. No visible pathologies."
}

# 5️⃣ Preprocessing & Prediction Functions
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
            results.append({'class': CLASS_NAMES[idx], 'confidence': conf.item() * 100})
        except: results.append({'error': 'Failed'})
    return results

# 6️⃣ UI Header
st.title("👁️ Hybrid Retinal Diagnostic Hub")
st.markdown("<p style='color: #64748b; font-size: 18px;'>Deep Learning Analysis for Optical Coherence Tomography (OCT)</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🚀 Real-time Diagnosis", "📊 Analysis Statistics", "📄 System Info"])

with tab1:
    # --- قسم الرفع (مثل الصورة الثانية تماماً) ---
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    
    # صف علوي يحتوي على العنوان وزر التشغيل
    top_col1, top_col2 = st.columns([2, 1])
    with top_col1:
        st.subheader("📸 Upload OCT Images")
        uploaded_files = st.file_uploader("Select one or more OCT scan images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    with top_col2:
        st.write("##") # موازنة المسافة
        execute_btn = st.button("Execute Hybrid Diagnosis")
    
    # رسالة الحالة عند الرفع
    if uploaded_files:
        st.success(f"✅ Loaded {len(uploaded_files)} images")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- قسم عرض النتائج ---
    if uploaded_files and execute_btn:
        with st.spinner('Analyzing images...'):
            pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]
            results = process_batch_images(pil_images)
            st.session_state['results'] = results

            # 1. إحصائيات سريعة (Diagnostic Results)
            st.subheader("Diagnostic Results")
            res_cols = st.columns(4)
            for i, name in enumerate(CLASS_NAMES):
                count = sum(1 for r in results if r.get('class') == name)
                res_cols[i].metric(name, count)

            st.divider()

            # 2. تحليل مفصل (Detailed Analysis) - شبكة منظمة من 3 أعمدة
            st.subheader("Detailed Analysis")
            # لتنظيم الصور، نستخدم 3 أعمدة بدلاً من 2 لتكون أصغر وأكثر ترتيباً
            cols_per_row = 3 
            for i in range(0, len(results), cols_per_row):
                grid = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(results):
                        with grid[j]:
                            res = results[idx]
                            st.image(pil_images[idx], use_container_width=True)
                            
                            diag = res['class']
                            conf = res['confidence']
                            color = "#10b981" if diag == "NORMAL" else "#f59e0b"
                            if diag == "DRUSEN": color = "#ef4444"

                            st.markdown(f"""
                                <div style='background-color: {color}; padding: 10px; border-radius: 8px; text-align: center; color: white;'>
                                    <span style='font-weight: bold;'>{diag}</span><br>
                                    <span style='font-size: 0.9em;'>Confidence: {conf:.1f}%</span>
                                </div>
                            """, unsafe_allow_html=True)
                            st.caption(f"_{CLASS_DESC[diag]}_")
    elif not uploaded_files:
        st.info("Please upload images to start the analysis.")

with tab2:
    if 'results' in st.session_state:
        st.subheader("Analysis Statistics")
        df = pd.DataFrame(st.session_state['results'])
        fig = px.pie(df, names='class', title="Diagnosis Distribution", hole=0.4,
                     color='class', color_discrete_map={'NORMAL': '#10b981', 'CNV': '#f59e0b', 'DME': '#3b82f6', 'DRUSEN': '#ef4444'})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.info("Hybrid System: DenseNet121 + Swin Transformer. Designed for high-accuracy medical diagnostics.")

# Footer
st.markdown("---")
st.caption("© 2025 OCTelligence AI Systems | Medical Imaging Division")
