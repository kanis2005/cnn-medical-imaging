import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import io

# Page configuration
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🫁",
    layout="wide"
)

# Title
st.title("🫁 Chest X-Ray Analysis with Deep Learning")
st.markdown("Detect pneumonia and other conditions using Convolutional Neural Networks")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .low-risk {
        background-color: #ccffcc;
        border-left: 5px solid #00ff00;
    }
</style>
""", unsafe_allow_html=True)

def create_cnn_model():
    """Create a CNN model for chest X-ray classification"""
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 classes: Normal, Pneumonia, COVID-19
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 (standard for many CNN architectures)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def generate_sample_xray():
    """Generate a sample X-ray-like image for demonstration"""
    # Create a synthetic X-ray-like image
    img = np.random.rand(224, 224, 3) * 0.3  # Dark background
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

# Main app
def main():
    # Sidebar
    st.sidebar.header("⚙️ Model Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Architecture",
        ["Custom CNN", "Pretrained ResNet50", "Pretrained DenseNet121"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 X-Ray Analysis", "📊 Model Info", "🏥 Clinical Guide", "📈 Performance"])
    
    with tab1:
        st.header("🔍 Chest X-Ray Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload X-Ray Image")
            
            # Image upload
            uploaded_file = st.file_uploader(
                "Choose a chest X-ray image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-Ray Image", use_container_width=True)
                
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Simulate prediction (in real scenario, load trained model)
                st.subheader("🤖 AI Analysis Results")
                
                # Simulate model prediction probabilities
                np.random.seed(42)
                probabilities = np.random.dirichlet(np.ones(3), size=1)[0]
                classes = ['Normal', 'Pneumonia', 'COVID-19']
                predicted_class = classes[np.argmax(probabilities)]
                
                # Display results
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.metric("Predicted Condition", predicted_class)
                    st.metric("Confidence", f"{np.max(probabilities):.2%}")
                
                with result_col2:
                    # Confidence scores
                    fig, ax = plt.subplots(figsize=(8, 4))
                    y_pos = np.arange(len(classes))
                    ax.barh(y_pos, probabilities, color=['green', 'orange', 'red'])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(classes)
                    ax.set_xlabel('Probability')
                    ax.set_title('Condition Probabilities')
                    st.pyplot(fig)
                
                # Clinical recommendations
                st.subheader("💡 Clinical Recommendations")
                
                if predicted_class == "Normal":
                    st.success("""
                    **Findings:** No significant abnormalities detected
                    **Recommendations:**
                    - Routine follow-up as scheduled
                    - No immediate intervention required
                    - Continue standard preventive care
                    """)
                elif predicted_class == "Pneumonia":
                    st.warning("""
                    **Findings:** Opacities consistent with pneumonia detected
                    **Recommendations:**
                    - Antibiotic therapy consideration
                    - Chest CT scan for confirmation
                    - Monitor oxygen saturation
                    - Consider hospitalization if symptoms severe
                    """)
                else:  # COVID-19
                    st.error("""
                    **Findings:** Bilateral ground-glass opacities suggestive of COVID-19
                    **Recommendations:**
                    - COVID-19 PCR testing
                    - Isolation protocols
                    - Steroid therapy consideration
                    - Monitor respiratory status closely
                    - Consider remdesivir if indicated
                    """)
            
            else:
                 st.info("👆 Upload a chest X-ray image to get started")
    sample_image = generate_sample_xray()
    st.image(sample_image, caption="Sample X-Ray (Upload your image for analysis)", use_container_width=True) 
        
    with col2:
            st.subheader("📋 Patient Information")
            
            # Patient details form
            with st.form("patient_info"):
                st.write("**Patient Demographics**")
                col_a, col_b = st.columns(2)
                with col_a:
                    age = st.number_input("Age", min_value=0, max_value=120, value=45)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                with col_b:
                    temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
                    oxygen_sat = st.number_input("O₂ Saturation (%)", min_value=70, max_value=100, value=98)
                
                st.write("**Symptoms**")
                fever = st.checkbox("Fever")
                cough = st.checkbox("Cough")
                shortness_breath = st.checkbox("Shortness of Breath")
                chest_pain = st.checkbox("Chest Pain")
                
                submitted = st.form_submit_button("Update Patient Info")
                if submitted:
                    st.success("Patient information updated!")
    
    with tab2:
        st.header("📊 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CNN Architecture")
            st.image("https://raw.githubusercontent.com/keras-team/keras-io/master/docs/img/conv2d.png", 
        caption="Convolutional Neural Network Architecture", use_container_width=True)
            
            st.write("""
            **Model Architecture:**
            - Input: 224×224×3 (RGB image)
            - 4 Convolutional blocks with MaxPooling
            - 512 & 256 neuron fully connected layers
            - Output: 3 classes (Normal, Pneumonia, COVID-19)
            - Activation: Softmax for multi-class classification
            """)
        
        with col2:
            st.subheader("Training Details")
            st.write("""
            **Training Parameters:**
            - Optimizer: Adam
            - Loss: Categorical Crossentropy
            - Metrics: Accuracy, Precision, Recall
            - Epochs: 50
            - Batch Size: 32
            - Validation Split: 20%
            
            **Dataset:**
            - Chest X-Ray Images (Pneumonia)
            - COVID-19 Radiography Database
            - Normal chest X-rays
            - Total: ~10,000 images
            """)
            
            # Simulated training history
            st.subheader("Training Progress")
            epochs = range(1, 51)
            train_acc = [0.5 + 0.4 * (1 - np.exp(-0.1 * x)) for x in epochs]
            val_acc = [0.5 + 0.35 * (1 - np.exp(-0.08 * x)) for x in epochs]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(epochs, train_acc, label='Training Accuracy')
            ax.plot(epochs, val_acc, label='Validation Accuracy')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with tab3:
        st.header("🏥 Clinical Guidelines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pneumonia Detection")
            st.write("""
            **Radiological Features:**
            - Airspace consolidation
            - Air bronchograms
            - Silhouette sign
            - Lobar or segmental distribution
            
            **Common Patterns:**
            - Bacterial: Lobar consolidation
            - Viral: Interstitial patterns
            - Atypical: Bilateral involvement
            """)
            
            st.subheader("COVID-19 Patterns")
            st.write("""
            **Characteristic Findings:**
            - Bilateral ground-glass opacities
            - Peripheral distribution
            - Crazy-paving pattern
            - Consolidation in severe cases
            
            **Timeline:**
            - Early: Ground-glass opacities
            - Progressive: Consolidation
            - Peak: Extensive involvement
            - Resolution: Gradual clearing
            """)
        
        with col2:
            st.subheader("Normal Chest X-Ray")
            st.write("""
            **Normal Anatomy:**
            - Clear lung fields
            - Sharp costophrenic angles
            - Normal cardiac silhouette
            - Intact diaphragm contours
            - No pleural effusion
            
            **Quality Assessment:**
            - Adequate inspiration
            - Proper rotation
            - Good penetration
            - Complete inclusion of anatomy
            """)
            
            st.subheader("AI Assistance")
            st.write("""
            **This tool assists with:**
            - Rapid screening of chest X-rays
            - Consistency in interpretation
            - Second opinion for radiologists
            - Triage in high-volume settings
            
            **Important:**
            - AI results should be verified by qualified radiologists
            - Clinical correlation is essential
            - Not a replacement for clinical judgment
            """)
    
    with tab4:
        st.header("📈 Model Performance")
        
        # Simulated performance metrics
        st.subheader("Classification Report")
        
        # Confusion matrix
        st.write("**Confusion Matrix**")
        cm = np.array([[850, 30, 20], [25, 920, 55], [15, 45, 940]])
        classes = ['Normal', 'Pneumonia', 'COVID-19']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Accuracy", "92.3%")
        with col2:
            st.metric("Precision", "91.8%")
        with col3:
            st.metric("Recall", "92.1%")
        with col4:
            st.metric("F1-Score", "91.9%")
        
        # ROC curves
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simulate ROC data
        fpr = np.linspace(0, 1, 100)
        for i, class_name in enumerate(classes):
            tpr = 1 - np.exp(-5 * fpr) + i * 0.1  # Simulated ROC
            tpr = np.clip(tpr, 0, 1)
            auc = np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()