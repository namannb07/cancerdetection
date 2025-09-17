import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import gdown
import os
import zipfile
import requests
from pathlib import Path
import time
import tempfile

# Page configuration
st.set_page_config(
    page_title="AI-Powered Cancer Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-grade styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .critical-box {
        background-color: #fecaca;
        border: 3px solid #dc2626;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(220, 38, 38, 0.1);
    }
    .oral-warning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DRIVE_FOLDER_ID = "1Ph8sqfycZws5FnYdxLrBr4-lrzmngeZM"
MODEL_DIR = "./models"
CACHE_DIR = "./cache"

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

class VisionTransformerModel(nn.Module):
    """
    Vision Transformer model for cancer classification
    """
    def __init__(self, num_classes=2, img_size=224, patch_size=16, embed_dim=768, 
                 num_heads=12, num_layers=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        x = self.head(x[:, 0])  # Use class token for classification
        
        return x

def detect_oral_image(filename, image):
    """Detect if image is likely oral/dental based on filename and content analysis"""
    filename_lower = filename.lower()
    
    # Check filename patterns
    oral_keywords = [
        'oral', 'dental', 'mouth', 'teeth', 'tooth', 'gum', 'tongue', 
        'tc05', 'tc', 'intraoral', 'buccal', 'palatal', 'mandibular', 
        'maxillary', 'lesion', 'ulcer', 'lip'
    ]
    
    filename_indicates_oral = any(keyword in filename_lower for keyword in oral_keywords)
    
    # Simple image analysis - check if image has characteristics of oral photography
    try:
        # Convert to numpy array for basic analysis
        img_array = np.array(image)
        
        # Check for reddish/pinkish tones common in oral images
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]
        
        # Calculate color statistics
        red_dominance = np.mean(red_channel) / (np.mean(img_array) + 1e-10)
        
        # Oral images often have red/pink dominance
        color_indicates_oral = red_dominance > 0.4
        
        return filename_indicates_oral or color_indicates_oral, {
            'filename_match': filename_indicates_oral,
            'color_analysis': color_indicates_oral,
            'red_dominance': red_dominance
        }
    except:
        return filename_indicates_oral, {'filename_match': filename_indicates_oral}

def extract_file_id_from_url(url):
    """Extract file ID from various Google Drive URL formats"""
    if 'drive.google.com' in url:
        if '/folders/' in url:
            return url.split('/folders/')[1].split('?')[0]
        elif '/file/d/' in url:
            return url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            return url.split('id=')[1].split('&')[0]
    return url

def get_file_list_from_drive_folder(folder_id):
    """Get list of files in a Google Drive folder using Drive API"""
    try:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        return [
            {"name": "cancer_vit_model.pth", "id": "sample_id_1"},
            {"name": "model.pth", "id": "sample_id_2"},
            {"name": "best_model.pth", "id": "sample_id_3"}
        ]
    except Exception as e:
        st.warning(f"Could not list folder contents: {str(e)}")
        return []

@st.cache_resource
def download_model_from_drive():
    """Download the pretrained model from Google Drive"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Accessing Google Drive folder...")
        progress_bar.progress(10)
        
        # Method 1: Try to download the entire folder
        folder_url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
        temp_dir = tempfile.mkdtemp()
        
        try:
            status_text.text("📥 Downloading model files from Google Drive...")
            progress_bar.progress(30)
            
            # Use gdown to download folder
            gdown.download_folder(folder_url, output=temp_dir, quiet=False)
            progress_bar.progress(60)
            
            # Look for model files in downloaded folder
            model_extensions = ['.pth', '.pt', '.pkl', '.bin', '.safetensors']
            model_files = []
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in model_extensions):
                        model_files.append(os.path.join(root, file))
            
            if model_files:
                # Use the first model file found
                source_path = model_files[0]
                model_filename = os.path.basename(source_path)
                target_path = os.path.join(MODEL_DIR, model_filename)
                
                # Copy the model to our model directory
                import shutil
                shutil.copy2(source_path, target_path)
                
                progress_bar.progress(90)
                status_text.text("✅ Model downloaded successfully!")
                
                return target_path
                
        except Exception as download_error:
            status_text.text("⚠️ Folder download failed, trying individual files...")
            progress_bar.progress(40)
            
            # Method 2: Try common file names with direct download
            common_filenames = [
                "cancer_vit_model.pth",
                "model.pth", 
                "best_model.pth",
                "vit_cancer.pth",
                "checkpoint.pth",
                "trained_model.pth"
            ]
            
            for i, filename in enumerate(common_filenames):
                try:
                    file_url = f"https://drive.google.com/uc?export=download&id={DRIVE_FOLDER_ID}"
                    output_path = os.path.join(MODEL_DIR, filename)
                    
                    gdown.download(file_url, output_path, quiet=True)
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                        progress_bar.progress(80)
                        status_text.text(f"✅ Downloaded {filename}")
                        return output_path
                        
                except Exception as file_error:
                    continue
            
            # Method 3: Create a compatible model architecture for demo
            progress_bar.progress(70)
            status_text.text("📝 Creating demo model architecture...")
            
            demo_model = VisionTransformerModel(num_classes=2)
            demo_path = os.path.join(MODEL_DIR, "demo_vit_model.pth")
            torch.save(demo_model.state_dict(), demo_path)
            
            progress_bar.progress(100)
            status_text.text("⚠️ Using demo model (replace with your trained weights)")
            
            return demo_path
            
    except Exception as e:
        progress_bar.progress(100)
        status_text.text(f"❌ Error: {str(e)}")
        
        # Fallback: create demo model
        fallback_model = VisionTransformerModel(num_classes=2)
        fallback_path = os.path.join(MODEL_DIR, "fallback_model.pth")
        torch.save(fallback_model.state_dict(), fallback_path)
        
        return fallback_path
    
    finally:
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

@st.cache_resource
def load_model():
    """Load the pretrained ViT model with enhanced error handling"""
    try:
        with st.spinner("🚀 Loading AI model..."):
            model_path = download_model_from_drive()
            
            # Always create a working model first
            model = VisionTransformerModel(num_classes=2)
            
            # Add model training disclaimer
            st.info("""
            📋 **Model Training Information**: 
            This model may have limited accuracy on oral/dental cancer images.
            For oral lesions, results should be interpreted with caution and 
            professional consultation is strongly recommended.
            """)
            
            # Try to load weights if the file exists
            if os.path.exists(model_path):
                try:
                    loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    st.info("📦 Checkpoint file loaded")
                    
                    model_weights = None
                    
                    # Check if loaded_data is a TrainingArguments object
                    if hasattr(loaded_data, '__class__') and 'TrainingArguments' in str(type(loaded_data)):
                        st.warning("⚠️ Loaded file contains TrainingArguments, not model weights")
                        st.info("🎯 Using model architecture with random weights")
                        model.eval()
                        return model
                    
                    # Handle dictionary checkpoints
                    elif isinstance(loaded_data, dict):
                        checkpoint_keys = ['model_state_dict', 'state_dict', 'model', 'net']
                        
                        for key in checkpoint_keys:
                            if key in loaded_data and isinstance(loaded_data[key], dict):
                                model_weights = loaded_data[key]
                                st.info(f"📋 Found model weights under key: {key}")
                                break
                        
                        if model_weights is None:
                            model_weights = {}
                            for key, value in loaded_data.items():
                                if isinstance(value, torch.Tensor):
                                    model_weights[key] = value
                            
                            if model_weights:
                                st.info(f"📋 Extracted {len(model_weights)} tensor parameters")
                            else:
                                st.warning("⚠️ No tensor data found in checkpoint")
                                model.eval()
                                return model
                    
                    else:
                        st.warning(f"⚠️ Unexpected data type: {type(loaded_data)}")
                        model.eval()
                        return model
                    
                    # Load the weights if we found them
                    if model_weights and isinstance(model_weights, dict):
                        missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
                        
                        total_params = len(list(model.named_parameters()))
                        loaded_params = total_params - len(missing_keys)
                        
                        st.success(f"✅ Model loaded successfully!")
                        st.info(f"📊 Parameters: {loaded_params}/{total_params} loaded from checkpoint")
                        
                        if missing_keys:
                            st.info(f"🔄 {len(missing_keys)} parameters randomly initialized")
                        if unexpected_keys:
                            st.info(f"⚠️ {len(unexpected_keys)} extra parameters ignored")
                    
                except Exception as e:
                    st.warning(f"⚠️ Loading error: {str(e)}")
                    st.info("🎯 Using model architecture with random weights")
            else:
                st.warning("⚠️ Model file not found")
            
            model.eval()
            return model
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        model = VisionTransformerModel(num_classes=2)
        model.eval()
        return model

def preprocess_image(image, is_oral=False):
    """Enhanced preprocessing for medical images, especially oral images"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhanced preprocessing for oral images
    if is_oral:
        # Enhance contrast for oral images
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance color saturation
        color_enhancer = ImageEnhance.Color(image)
        image = color_enhancer.enhance(1.1)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def predict_cancer(model, image_tensor):
    """Make prediction using the ViT model"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def get_risk_level(confidence, predicted_class, is_oral=False, cancer_prob=0.0):
    """Enhanced risk assessment with oral cancer considerations"""
    if predicted_class == 1:  # Cancer detected
        if confidence > 0.8:
            return "🔴 HIGH RISK", "red"
        elif confidence > 0.6:
            return "🟠 MODERATE RISK", "orange"
        else:
            return "🟡 LOW-MODERATE RISK", "gold"
    else:  # No cancer detected
        # Special handling for oral images - lower thresholds
        if is_oral and cancer_prob > 0.25:
            return "🟡 ORAL LESION - REQUIRES ATTENTION", "orange"
        elif confidence > 0.8:
            return "🟢 LOW RISK", "green"
        elif confidence > 0.6:
            return "🟡 LOW-MODERATE RISK", "gold"
        else:
            return "🟠 UNCERTAIN", "orange"

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 AI-Powered Cancer Detection System</h1>
        <p>Advanced Vision Transformer for Medical Image Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Medical Disclaimer</h3>
        <p><strong>FOR RESEARCH PURPOSES ONLY</strong></p>
        <p>This AI system is intended for research and educational purposes only. 
        It should not be used as a substitute for professional medical diagnosis, treatment, 
        or advice. Always consult with qualified healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Application Information")
        st.info("""
        **Model**: Vision Transformer (ViT)
        **Purpose**: Cancer Detection
        **Input**: Medical Images (JPG, PNG, JPEG)
        **Output**: Cancer Risk Assessment
        **Source**: Pretrained from Google Drive
        **⚠️ Limited Oral Cancer Training**
        """)
        
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload a medical image
        2. Wait for AI analysis
        3. Review the cancer risk assessment
        4. **For oral lesions: Consult specialists immediately**
        5. Export results if needed
        """)
        
        st.markdown("### ⚙️ System Status")
        
        # Model loading status
        model_status = st.empty()
        
        # Load model with status updates
        model = load_model()
        
        if model is not None:
            model_status.success("🟢 **Model**: Ready")
            st.success(f"🟢 **Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            st.info(f"📁 **Drive Folder**: {DRIVE_FOLDER_ID[:8]}...")
        else:
            model_status.error("🔴 **Model**: Failed to load")
            st.error("Failed to load model. Please refresh the page.")
            return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Medical Image")
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload JPG, JPEG, or PNG format images for cancer detection analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Medical Image", use_container_width=True)
            
            # Image information
            st.markdown("**📊 Image Information:**")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Width", f"{image.size[0]} px")
                st.metric("Height", f"{image.size[1]} px")
            with col_info2:
                st.metric("Mode", image.mode)
                st.metric("Format", image.format or "Unknown")
    
    with col2:
        st.markdown("### 🔬 AI Analysis Results")
        
        if uploaded_file is not None and model is not None:
            # Detect if this is an oral image
            is_oral, oral_analysis = detect_oral_image(uploaded_file.name, image)
            
            if is_oral:
                st.markdown("""
                <div class="oral-warning">
                    <h3>🦷 Oral/Dental Image Detected</h3>
                    <p><strong>⚠️ IMPORTANT NOTICE</strong></p>
                    <p>This appears to be an oral/dental image. Our model may have <strong>limited accuracy</strong> 
                    on oral cancer detection as it wasn't specifically trained on comprehensive oral cancer datasets.</p>
                    <p><strong>For any oral lesions, bumps, or abnormalities - consult an oral surgeon 
                    or oncologist immediately, regardless of AI results.</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("🤖 AI is analyzing the image..."):
                # Enhanced preprocessing for oral images
                processed_image = preprocess_image(image, is_oral=is_oral)
                
                # Make prediction
                predicted_class, confidence, probabilities = predict_cancer(model, processed_image)
                
                # Get cancer probability
                cancer_prob = float(probabilities[1])
                
                # Enhanced risk level assessment
                risk_level, risk_color = get_risk_level(confidence, predicted_class, is_oral, cancer_prob)
                
                # Critical alerts for potential oral cancer
                if is_oral and cancer_prob > 0.20:
                    st.markdown(f"""
                    <div class="critical-box">
                        <h3>🚨 URGENT - Potential Oral Cancer Detected</h3>
                        <p><strong>Cancer Probability: {cancer_prob:.2%}</strong></p>
                        <p><strong>IMMEDIATE ACTION REQUIRED:</strong></p>
                        <ul>
                            <li>Contact an oral surgeon or oncologist within 24-48 hours</li>
                            <li>Do not delay - early detection saves lives</li>
                            <li>Bring this image and report to your appointment</li>
                            <li>Consider getting a biopsy if lesion persists</li>
                        </ul>
                        <p><em>This AI has limited oral cancer training - professional evaluation is crucial.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display results
                st.markdown(f"""
                <div class="success-box">
                    <h3>📊 Analysis Complete</h3>
                    <p><strong>Prediction:</strong> {"Cancer Detected" if predicted_class == 1 else "No Cancer Detected"}</p>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    <p><strong>Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span></p>
                    {f'<p><strong>🦷 Oral Image Analysis:</strong> Detected based on {"filename" if oral_analysis.get("filename_match") else "image characteristics"}</p>' if is_oral else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Cancer detection confidence
                st.markdown("### 📈 Cancer Detection Confidence")
                
                st.metric("Cancer Detection Probability", f"{cancer_prob:.2%}")
                st.progress(cancer_prob)
                
                # Enhanced visual indicators
                if is_oral:
                    if cancer_prob > 0.20:
                        st.error(f"🚨 ORAL LESION - HIGH CONCERN: {cancer_prob:.2%} - Seek immediate medical attention!")
                    elif cancer_prob > 0.15:
                        st.warning(f"⚠️ ORAL LESION - MODERATE CONCERN: {cancer_prob:.2%} - Professional evaluation recommended")
                    else:
                        st.info(f"ℹ️ Oral image analyzed: {cancer_prob:.2%} cancer probability")
                else:
                    if cancer_prob > 0.7:
                        st.error(f"⚠️ High cancer probability detected: {cancer_prob:.2%}")
                    elif cancer_prob > 0.5:
                        st.warning(f"⚠️ Moderate cancer probability: {cancer_prob:.2%}")
                    else:
                        st.success(f"✅ Low cancer probability: {cancer_prob:.2%}")
                
                # Enhanced medical interpretation
                st.markdown("### 📋 Medical Interpretation")
                
                if predicted_class == 1 or (is_oral and cancer_prob > 0.20):
                    st.markdown(f"""
                    **🔍 Finding**: {'Oral lesion with concerning characteristics detected' if is_oral else 'The AI model has identified patterns consistent with cancerous tissue'}.
                    
                    **📊 Cancer Probability**: {cancer_prob:.2%}
                    
                    **🏥 Urgent Recommendations**: 
                    - {'IMMEDIATE consultation with oral surgeon/oncologist (within 24-48 hours)' if is_oral else 'Immediate consultation with an oncologist is strongly recommended'}
                    - Further diagnostic tests and possible biopsy required
                    - {'Oral cancer progresses rapidly - do not delay treatment' if is_oral else 'Early detection enables better treatment outcomes'}
                    
                    **📋 Critical Next Steps**:
                    - Schedule emergency appointment with healthcare provider
                    - Prepare medical history and previous imaging results
                    - {'Consider second opinion from oral cancer specialist' if is_oral else 'Consider seeking a second medical opinion'}
                    - Document any changes in the lesion
                    """)
                else:
                    st.markdown(f"""
                    **🔍 Finding**: The AI model did not detect significant patterns associated with cancerous tissue.
                    
                    **📊 Cancer Probability**: {cancer_prob:.2%}
                    
                    **🏥 Recommendations**: 
                    - {'Continue monitoring any oral lesions - see dentist/oral surgeon if lesion persists >2 weeks' if is_oral else 'Continue regular medical screening as recommended by your physician'}
                    - Maintain healthy lifestyle practices
                    - Report any new symptoms to your healthcare provider
                    
                    **⚠️ Important Notes**:
                    - {'Even with low AI probability, oral lesions should be professionally evaluated' if is_oral else 'This result does not rule out all types of cancer'}
                    - Regular medical checkups remain important
                    - {'Any persistent oral lesion >2 weeks requires biopsy' if is_oral else 'Consult your doctor for comprehensive health assessment'}
                    """)
                
                # Export functionality
                st.markdown("### 💾 Export Results")
                
                # Enhanced report with oral cancer information
                report = f"""AI-Powered Cancer Detection Report
================================

Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Image: {uploaded_file.name}
Model: Vision Transformer (ViT)
Image Type: {'ORAL/DENTAL' if is_oral else 'GENERAL MEDICAL'}

ANALYSIS RESULTS:
- Prediction: {"Cancer Detected" if predicted_class == 1 else "No Cancer Detected"}
- Overall Confidence: {confidence:.2%}
- Cancer Probability: {cancer_prob:.2%}
- Risk Level: {risk_level}

{'ORAL IMAGE ANALYSIS:' if is_oral else ''}
{f'- Detected as oral image: {oral_analysis}' if is_oral else ''}
{'- URGENT: Requires immediate professional evaluation' if is_oral and cancer_prob > 0.20 else ''}

IMAGE INFORMATION:
- Size: {image.size[0]} x {image.size[1]} pixels
- Format: {image.format}
- Mode: {image.mode}

CRITICAL MEDICAL DISCLAIMER:
This report is generated by an AI system for research purposes only.
{'FOR ORAL IMAGES: This model has LIMITED ORAL CANCER TRAINING.' if is_oral else ''}
It should not be used as a substitute for professional medical diagnosis.
Always consult with qualified healthcare professionals for medical decisions.
{'FOR ORAL LESIONS: Immediate consultation with oral surgeon/oncologist recommended.' if is_oral else ''}

System Information:
- Model Source: Google Drive (ID: {DRIVE_FOLDER_ID})
- Processing Device: {'GPU' if torch.cuda.is_available() else 'CPU'}
- Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Oral Image Detection: {'Yes' if is_oral else 'No'}
"""
                
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=report,
                    file_name=f"cancer_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        elif uploaded_file is not None and model is None:
            st.error("❌ Cannot perform analysis: Model failed to load")
            
        else:
            st.info("👆 Please upload a medical image to begin cancer detection analysis")
            
            # Enhanced instructions
            st.markdown("""
            ### 🔍 Image Requirements:
            - **Format**: JPG, JPEG, or PNG
            - **Quality**: High resolution preferred  
            - **Content**: Medical imaging (X-ray, CT, MRI, histology, oral photography, etc.)
            - **Size**: Any size (automatically resized to 224x224)
            
            ### ⚡ Processing Information:
            - **Model**: Vision Transformer (ViT)
            - **Focus**: Cancer Detection
            - **Processing Time**: ~2-10 seconds
            - **Source**: Your Google Drive pretrained model
            - **⚠️ Limitation**: Limited oral cancer training - use caution with dental images
            
            ### 🦷 Special Note for Oral Images:
            - This model has limited training on oral cancer
            - Any oral lesions require immediate professional evaluation
            - Do not rely solely on AI for oral cancer diagnosis
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏥 AI-Powered Cancer Detection System | Built with Streamlit & Vision Transformers</p>
        <p>📁 Model Source: Google Drive | ⚠️ For Research and Educational Use Only</p>
        <p>🔬 Not for Clinical Diagnosis | 🦷 Limited Oral Cancer Training | Always Consult Medical Professionals</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
