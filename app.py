import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from HeartDiseasePredictor import HeartDiseasePredictor

def main():
    # Page configuration
    st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")
    
    # Header
    st.title("Heart Disease Risk Predictor")
    st.write("Enter your health information to assess your risk of heart disease")
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Initialize feature names and categorical features
    predictor.initialize_features()
    
    # Sidebar for model training/loading
    with st.sidebar:
        st.header("Model Options")
        model_option = st.radio(
            "Choose an option:",
            ["Use existing model", "Train new model"]
        )
        
        if model_option == "Train new model":
            if st.button("Train Model"):
                with st.spinner("Loading dataset..."):
                    data = predictor.load_dataset()
                
                with st.spinner("Preprocessing data..."):
                    X, y = predictor.preprocess_data(data)
                
                with st.spinner("Training model..."):
                    predictor.train_model(X, y)
                
                st.success("Model trained and saved successfully!")
        else:
            if os.path.exists(predictor.model_path):
                predictor.load_model()
                st.success("Existing model loaded successfully!")
            else:
                st.warning("No trained model found. Please train a new model.")
    
    # Main content - User input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographic Information")
        age = st.slider("Age", 18, 100, 50)
        sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        
        st.subheader("Heart Health Metrics")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        thalach = st.slider("Maximum Heart Rate", 70, 220, 150)
    
    with col2:
        st.subheader("Diagnostic Information")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                        format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], 
                             format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
        exang = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], 
                          format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
        
        # Fixed: Using dictionary mapping 
        thal_options = {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}
        thal = st.selectbox("Thalassemia", [1, 2, 3], 
                          format_func=lambda x: thal_options[x])
    
    # Create features dictionary
    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Prediction section
    st.subheader("Prediction")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Predict Risk"):
            if not os.path.exists(predictor.model_path):
                st.error("No trained model found. Please train a model first.")
            else:
                with st.spinner("Calculating risk..."):
                    result = predictor.predict(features)
                
                # Display result
                risk_color = {
                    "Low Risk": "green",
                    "Moderate Risk": "orange",
                    "High Risk": "red"
                }
                
                st.markdown(f"""
                ### Risk Assessment:
                - **Risk Level:** <span style='color:{risk_color[result["risk_level"]]}'>{result["risk_level"]}</span>
                - **Probability:** {result["probability"]:.2f}
                """, unsafe_allow_html=True)
    
    with col2:
        if os.path.exists(predictor.model_path) and predictor.model is not None:
            try:
                importance = predictor.get_feature_importance()
                
                if importance and len(importance) > 0:
                    # Get top 8 features
                    top_features = importance[:8]
                    
                    # Create a dataframe for plotting
                    df_importance = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='Importance', y='Feature', data=df_importance, ax=ax)
                    ax.set_title('Top Features Affecting Heart Disease Risk')
                    st.pyplot(fig)
                else:
                    st.info("Feature importance not available for this model type.")
            except Exception as e:
                st.error(f"Error displaying feature importance: {str(e)}")

if __name__ == "__main__":
    main()