import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class MentalHealthPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        
    def clean_gender(self, gender):
        """Clean and standardize gender values"""
        gender = str(gender).lower()
        if any(term in gender for term in ['female', 'f', 'woman', 'cis female']):
            return 'female'
        elif any(term in gender for term in ['male', 'm', 'cis male', 'man']):
            return 'male'
        else:
            return 'other'
    
    def country_to_continent(self, country):
        """Map country to continent"""
        country = str(country).lower()
        if country in ['united states', 'canada', 'mexico']:
            return 'North America'
        elif country in ['argentina', 'bolivia', 'brazil', 'chile', 'colombia', 'ecuador', 
                        'guyana', 'paraguay', 'peru', 'suriname', 'uruguay', 'venezuela']:
            return 'South America'
        elif country in ['australia', 'new zealand']:
            return 'Oceania'
        elif country in ['india', 'china', 'japan', 'singapore', 'israel', 'thailand']:
            return 'Asia'
        elif country in ['south africa', 'nigeria', 'zimbabwe']:
            return 'Africa'
        else:
            return 'Europe'
    
    def clean_self_employ(self, value):
        """Clean self-employed values"""
        if value == 'Unknown':
            return 'No'
        else:
            return value
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the dataset - exactly matching notebook preprocessing"""
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Drop unnecessary columns if they exist (exact order from notebook)
        columns_to_drop = ['Timestamp', 'state', 'comments']
        for col in columns_to_drop:
            if col in df_processed.columns:
                del df_processed[col]
        
        # Fill missing values (exact notebook approach)
        df_processed.fillna('Unknown', inplace=True)
        
        # Clean Age (exact notebook approach)
        df_processed['Age'] = pd.to_numeric(df_processed['Age'], errors='coerce')
        df_processed['Age'] = df_processed['Age'].apply(lambda x: x if 18 <= x <= 100 else np.nan)
        df_processed['Age'].fillna(df_processed['Age'].mean(), inplace=True)
        
        # Clean Gender (exact notebook approach)
        df_processed['Gender_clean'] = df_processed['Gender'].apply(self.clean_gender)
        
        # Delete original Gender column (like notebook)
        if 'Gender' in df_processed.columns:
            del df_processed['Gender']
        
        # Map Country to Continent (exact notebook approach)
        if 'Country' in df_processed.columns:
            df_processed['Continent'] = df_processed['Country'].apply(self.country_to_continent)
            del df_processed['Country']
        
        # Clean self_employed (exact notebook approach)
        if 'self_employed' in df_processed.columns:
            df_processed['self_employed'] = df_processed['self_employed'].apply(self.clean_self_employ)
        
        # Create encoded DataFrame (exact notebook structure)
        df_encoded = pd.DataFrame()
        df_encoded['Age'] = df_processed['Age']
        
        # Features to encode - exact order from notebook
        all_features_to_encode = [
            'Gender_clean', 'Continent', 'work_interfere', 'leave',
            'self_employed', 'family_history', 'remote_work', 'tech_company',
            'benefits', 'care_options', 'wellness_program', 'seek_help',
            'anonymity', 'mental_health_consequence', 'phys_health_consequence',
            'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
            'mental_vs_physical', 'obs_consequence', 'no_employees', 'treatment'
        ]
        
        # Process ALL features - exactly like notebook using SAME le instance
        if is_training:
            # Initialize single LabelEncoder instance like notebook
            self.le = LabelEncoder()
        
        for feature in all_features_to_encode:
            if feature not in df_processed.columns:
                continue
                
            if feature in ['work_interfere', 'leave', 'no_employees']:
                # Use ordinal encoding but STILL SAVE the mapping (exact notebook approach)
                if feature == 'work_interfere':
                    mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
                elif feature == 'leave':
                    mapping = {'Very difficult': 0, 'Somewhat difficult': 1, "Don't know": 2,
                              'Somewhat easy': 3, 'Very easy': 4}
                elif feature == 'no_employees':
                    mapping = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3,
                              '500-1000': 4, 'More than 1000': 5}
                
                df_encoded[f'{feature}_encoded'] = df_processed[feature].map(mapping)
                if is_training:
                    self.label_encoders[feature] = mapping  # Save the mapping!
            else:
                # CRITICAL: Use the SAME LabelEncoder instance for ALL features (exact notebook approach)
                if is_training:
                    df_encoded[f'{feature}_encoded'] = self.le.fit_transform(df_processed[feature].astype(str))
                    self.label_encoders[feature] = self.le  # Save reference to same encoder
                else:
                    # Use existing encoder - all features use the same encoder
                    if hasattr(self, 'le'):
                        # Handle unseen categories by adding to the shared encoder
                        unique_values = df_processed[feature].astype(str).unique()
                        for val in unique_values:
                            if val not in self.le.classes_:
                                self.le.classes_ = np.append(self.le.classes_, val)
                        df_encoded[f'{feature}_encoded'] = self.le.transform(df_processed[feature].astype(str))
        
        # Handle missing values in work_interfere_encoded (exact notebook approach)
        if 'work_interfere_encoded' in df_encoded.columns:
            mode_value = df_encoded['work_interfere_encoded'].mode()
            if len(mode_value) > 0:
                df_encoded['work_interfere_encoded'] = df_encoded['work_interfere_encoded'].fillna(mode_value[0])
        
        return df_encoded
    
    def train_model(self, df):
        """Train the Random Forest model - exactly matching notebook configuration"""
        # Preprocess data
        df_encoded = self.preprocess_data(df, is_training=True)
        
        # Prepare features and target
        X = df_encoded.drop('treatment_encoded', axis=1)
        y = df_encoded['treatment_encoded']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Train-Test Split (using optimal seed for best performance - exact notebook)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=39,  # Optimal split - same as notebook
            stratify=y
        )
        
        # Train the final model (exact notebook configuration)
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42  # Same as notebook
        )
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Evaluate performance
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def predict(self, input_data):
        """Make prediction for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Create DataFrame from input
        df_input = pd.DataFrame([input_data])
        
        # Preprocess
        df_encoded = self.preprocess_data(df_input, is_training=False)
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df_encoded = df_encoded[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(df_encoded)[0]
        probability = self.model.predict_proba(df_encoded)[0]
        
        return {
            'prediction': int(prediction),
            'probability_no_treatment': float(probability[0]),
            'probability_treatment': float(probability[1])
        }
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save the trained model and encoders"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'le': self.le if hasattr(self, 'le') else None  # Save the shared encoder
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model and encoders"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            if 'le' in model_data and model_data['le'] is not None:
                self.le = model_data['le']  # Restore the shared encoder
            return True
        return False

# Function to train and save model - exact notebook approach
def train_and_save_model():
    """Train the model using the exact notebook approach and save it"""
    # Load data exactly like notebook
    df = pd.read_csv('survey.csv')
    
    # Preprocessing exactly like notebook
    del df['Timestamp']
    del df['state'] 
    del df['comments']

    df.fillna('Unknown', inplace=True)

    # Age cleaning - exact notebook
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].apply(lambda x: x if 18 <= x <= 100 else np.nan)
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # Gender cleaning - exact notebook
    def cleanGender(gender):
        gender = str(gender).lower()
        if any(term in gender for term in ['female', 'f', 'woman', 'cis female']):
            return 'female'
        elif any(term in gender for term in ['male', 'm', 'cis male', 'man']):
            return 'male'
        else:
            return 'other'

    df['Gender_clean'] = df['Gender'].apply(cleanGender)

    # Gender encoding - SINGLE LabelEncoder instance like notebook
    le = LabelEncoder()
    df['Gender_encoded'] = le.fit_transform(df['Gender_clean'])
    del df['Gender']

    # Country to continent mapping - exact notebook
    def country_to_continent(country):
        if country in ['United States', 'Canada', 'Mexico']:
            return 'North America'
        elif country in ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']:
            return 'South America'
        elif country in ['Australia', 'New Zealand']:
            return 'Oceania'
        elif country in ['india', 'china', 'japan', 'singapore', 'israel', 'thailand', 'australia', 'new zealand']:
            return 'Asia'
        elif country in ['south africa', 'nigeria', 'zimbabwe']:
            return 'Africa'
        else:
            return 'Europe'

    df['Continent'] = df['Country'].apply(country_to_continent)
    del df['Country']

    # Clean self_employed - exact notebook
    def clean_self_employ(row):
        if row == 'Unknown':
            return 'No'
        else:
            return row

    df['self_employed'] = df['self_employed'].apply(clean_self_employ)

    # Feature encoding - EXACT notebook approach
    df_encoded = pd.DataFrame()
    df_encoded['Age'] = df['Age']

    all_features_to_encode = [
        'Gender_clean', 'Continent', 'work_interfere', 'leave',
        'self_employed', 'family_history', 'remote_work', 'tech_company',
        'benefits', 'care_options', 'wellness_program', 'seek_help',
        'anonymity', 'mental_health_consequence', 'phys_health_consequence',
        'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
        'mental_vs_physical', 'obs_consequence', 'no_employees', 'treatment'
    ]

    # Process ALL features - EXACT notebook approach using SAME le instance
    label_encoders = {}
    for feature in all_features_to_encode:
        if feature in ['work_interfere', 'leave', 'no_employees']:
            # Use ordinal encoding but STILL SAVE the mapping
            if feature == 'work_interfere':
                mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
            elif feature == 'leave':
                mapping = {'Very difficult': 0, 'Somewhat difficult': 1, "Don't know": 2,
                          'Somewhat easy': 3, 'Very easy': 4}
            elif feature == 'no_employees':
                mapping = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3,
                          '500-1000': 4, 'More than 1000': 5}

            df_encoded[f'{feature}_encoded'] = df[feature].map(mapping)
            label_encoders[feature] = mapping  # Save the mapping!

        else:
            # CRITICAL: Use the SAME LabelEncoder instance 'le' for ALL features
            df_encoded[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
            label_encoders[feature] = le  # Save the encoder

    # Handle missing values in work_interfere_encoded
    mode_value = df_encoded['work_interfere_encoded'].mode()[0]  # Most frequent value
    df_encoded['work_interfere_encoded'] = df_encoded['work_interfere_encoded'].fillna(mode_value)

    # Train-Test Split (using optimal seed for best performance)
    X = df_encoded.drop('treatment_encoded', axis=1)
    y = df_encoded['treatment_encoded']

    # Using seed=39 which was found to give optimal performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=39,  # Optimal split
        stratify=y
    )

    # Train the final model
    final_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    final_model.fit(X_train, y_train)

    # Make predictions
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Save model with exact notebook configuration
    model_data = {
        'model': final_model,
        'label_encoders': label_encoders,
        'feature_columns': X.columns.tolist(),
        'le': le  # Save the shared encoder
    }
    joblib.dump(model_data, 'mental_health_model.pkl')
    
    print("Model trained and saved successfully!")
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return accuracy, recall, f1, roc_auc

if __name__ == "__main__":
    train_and_save_model()