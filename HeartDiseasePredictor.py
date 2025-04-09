import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    def __init__(self, model_path='heart_disease_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.categorical_features = None
        self.initialize_features()
        
    def initialize_features(self):
        # Initialize feature names and categorical features
        self.feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 
                             'fbs', 'restecg', 'thalach', 'exang', 
                             'oldpeak', 'slope', 'ca', 'thal']
        
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        
    def load_dataset(self):
        print("Loading Cleveland dataset...")
        
        url_cleveland = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cleveland_df = pd.read_csv(url_cleveland, header=None)
        
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 
            'fbs', 'restecg', 'thalach', 'exang', 
            'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        cleveland_df.columns = column_names
        
        cleveland_df = cleveland_df.replace('?', np.nan)
        
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for col in numeric_cols:
            cleveland_df[col] = pd.to_numeric(cleveland_df[col], errors='coerce')
        
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'target']
        for col in categorical_cols:
            cleveland_df[col] = pd.to_numeric(cleveland_df[col], errors='coerce')
        
        cleveland_df['target'] = cleveland_df['target'].apply(lambda x: 0 if x == 0 else 1)
        
        print(f"Cleveland dataset shape: {cleveland_df.shape}")
        return cleveland_df
        
    def preprocess_data(self, data):
        print("Preprocessing data...")
        
        # Initialize feature names and categorical features
        self.initialize_features()
        
        # Extract features and target
        X = data[self.feature_names]
        y = data['target']
        
        return X, y
    
    def explore_data(self, data):
        print("Exploring data...")
        
        # Print dataset info
        print(f"Dataset shape: {data.shape}")
        print(f"Missing values per column:\n{data.isnull().sum()}")
        
        # Distribution of target variable
        target_counts = data['target'].value_counts()
        print(f"Target distribution: {target_counts}")
        
        # Data summary statistics
        print("Dataset summary statistics:")
        print(data.describe())
        
        # Correlation between features
        correlation = data.corr()
        print("Top correlations with target:")
        print(correlation['target'].sort_values(ascending=False))
    
    def build_model(self):
        print("Building model pipeline...")
        
        # Define preprocessing for numeric columns
        numeric_features = [col for col in self.feature_names if col not in self.categorical_features]
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Define different classifiers for comparison
        classifiers = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Create a pipeline for each classifier
        pipelines = {}
        for clf_name, clf in classifiers.items():
            pipelines[clf_name] = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
        
        return pipelines
    
    def train_model(self, X, y):
        print("Training model...")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Build model pipelines
        pipelines = self.build_model()
        
        # Train and evaluate each model
        best_accuracy = 0
        best_model_name = None
        
        for name, pipeline in pipelines.items():
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Keep track of the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                self.model = pipeline
        
        print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # Detailed evaluation of the best model
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot ROC curve
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        
        # Save the model
        self.save_model()
    
    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"Model file {self.model_path} not found.")
            return False
    
    def predict(self, features):
        if self.model is None:
            if not self.load_model():
                raise Exception("No trained model available. Train a model first.")
        
        # Arrange features in the correct order
        feature_array = []
        for feature in self.feature_names:
            feature_array.append(features.get(feature, np.nan))
            
        # Convert to DataFrame
        X = pd.DataFrame([feature_array], columns=self.feature_names)
        
        # Make prediction
        probability = self.model.predict_proba(X)[0][1]
        prediction = self.model.predict(X)[0]
        
        return {
            'probability': probability,
            'prediction': int(prediction),
            'risk_level': 'High Risk' if probability > 0.7 else
                          'Moderate Risk' if probability > 0.3 else 'Low Risk'
        }

    def get_feature_importance(self):
        if self.model is None:
            if not self.load_model():
                raise Exception("No trained model available. Train a model first.")
        
        # Check if the model has a feature_importances_ attribute
        if not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            return []
            
        try:
            # Get feature names from the preprocessor
            preprocessor = self.model.named_steps['preprocessor']
            feature_names = []
            
            # Get numeric feature names
            numeric_features = [col for col in self.feature_names if col not in self.categorical_features]
            feature_names.extend(numeric_features)
            
            # Get categorical feature names (after one-hot encoding)
            categorical_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            categorical_features = []
            for i, cat_feature in enumerate(self.categorical_features):
                categories = categorical_encoder.categories_[i]
                for category in categories:
                    categorical_features.append(f"{cat_feature}_{category}")
            
            feature_names.extend(categorical_features)
            
            # Get feature importance
            importances = self.model.named_steps['classifier'].feature_importances_
            
            # Handle case where number of features doesn't match
            if len(importances) != len(feature_names):
                # Fallback to just using importance values with generic names
                feature_importance = [("Feature_" + str(i), imp) for i, imp in enumerate(importances)]
            else:
                feature_importance = dict(zip(feature_names, importances))
                feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
            return feature_importance
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return []

if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    
    data = predictor.load_dataset()
    
    predictor.explore_data(data)
    
    X, y = predictor.preprocess_data(data)
    
    predictor.train_model(X, y)
    
    sample_features = {
        'age': 65,
        'sex': 1,  # Male
        'cp': 3,   # Chest pain type (0-3)
        'trestbps': 140,  # Resting blood pressure
        'chol': 260,  # Cholesterol
        'fbs': 0,  # Fasting blood sugar < 120 mg/dl
        'restecg': 0,  # Resting ECG results
        'thalach': 130,  # Maximum heart rate achieved
        'exang': 1,  # Exercise induced angina
        'oldpeak': 2.5,  # ST depression induced by exercise
        'slope': 1,  # The slope of the peak exercise ST segment
        'ca': 2,  # Number of major vessels
        'thal': 2  # Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
    }
    
    result = predictor.predict(sample_features)
    print(f"Prediction result: {result}")
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    if importance:
        print("\nFeature Importance:")
        for feature, score in importance[:10]:  # Top 10 features
            print(f"{feature}: {score:.4f}")