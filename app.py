import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import MentalHealthPredictor
import os

# Page configuration
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Theme toggle in sidebar
with st.sidebar:
    st.title("Settings")
    # Note: Streamlit doesn't have built-in dark/light toggle, but users can use their browser settings
    st.info("💡 Tip: Use your browser's dark/light mode settings to change the theme")

# Simple CSS for clean styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .positive-result {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .negative-result {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #1b5e20;
    }
    .form-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load or train the model"""
    predictor = MentalHealthPredictor()
    
    # Try to load existing model
    if os.path.exists('mental_health_model.pkl'):
        if predictor.load_model('mental_health_model.pkl'):
            return predictor
    
    # If model doesn't exist, train it
    with st.spinner("Training model... This may take a few minutes."):
        from model import train_and_save_model
        train_and_save_model()
        predictor.load_model('mental_health_model.pkl')
    
    return predictor

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Mental Health Treatment Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p>Answer the questions below to get a prediction about likelihood of seeking mental health treatment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    
    # Create form
    with st.form("prediction_form"):
        st.markdown("### Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age input - simple number input
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Continent selection to match model processing
        continent = st.selectbox("Continent", [
            "North America", "Europe", "Asia", "Oceania", "South America", "Africa"
        ])
        
        st.markdown("### Work Environment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self_employed = st.radio("Are you self-employed?", ["No", "Yes"])
            tech_company = st.radio("Do you work for a tech company?", ["Yes", "No"])
            remote_work = st.radio("Do you work remotely?", ["No", "Yes"])
        
        with col2:
            no_employees = st.selectbox("Company size (number of employees)", [
                "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"
            ])
        
        st.markdown("### Mental Health Background")
        
        col1, col2 = st.columns(2)
        
        with col1:
            family_history = st.radio("Do you have a family history of mental illness?", ["No", "Yes"])
        
        with col2:
            work_interfere = st.selectbox(
                "If you have a mental health condition, how often does it interfere with your work?",
                ["Never", "Rarely", "Sometimes", "Often"]
            )
        
        st.markdown("### Company Mental Health Support")
        
        col1, col2 = st.columns(2)
        
        with col1:
            benefits = st.selectbox("Does your employer provide mental health benefits?", [
                "Yes", "No", "Don't know"
            ])
            care_options = st.selectbox("Do you know the options for mental health care your employer provides?", [
                "Yes", "No", "Not sure"
            ])
            wellness_program = st.selectbox("Has your employer ever discussed mental health as part of an employee wellness program?", [
                "Yes", "No", "Don't know"
            ])
        
        with col2:
            seek_help = st.selectbox("Does your employer provide resources to learn more about mental health issues?", [
                "Yes", "No", "Don't know"
            ])
            anonymity = st.selectbox("Is your anonymity protected if you choose to take advantage of mental health resources?", [
                "Yes", "No", "Don't know"
            ])
            leave = st.selectbox("How easy is it for you to take medical leave for a mental health condition?", [
                "Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"
            ])
        
        st.markdown("### Workplace Attitudes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mental_health_consequence = st.selectbox("Do you think discussing mental health with your employer would have negative consequences?", [
                "No", "Yes", "Maybe"
            ])
            phys_health_consequence = st.selectbox("Do you think discussing physical health with your employer would have negative consequences?", [
                "No", "Yes", "Maybe"
            ])
            coworkers = st.selectbox("Would you be willing to discuss a mental health issue with your coworkers?", [
                "Yes", "No", "Some of them"
            ])
            supervisor = st.selectbox("Would you be willing to discuss a mental health issue with your direct supervisor(s)?", [
                "Yes", "No", "Some of them"
            ])
        
        with col2:
            mental_health_interview = st.selectbox("Would you bring up a mental health issue during an interview?", [
                "No", "Yes", "Maybe"
            ])
            phys_health_interview = st.selectbox("Would you bring up a physical health issue during an interview?", [
                "No", "Yes", "Maybe"
            ])
            mental_vs_physical = st.selectbox("Do you feel that your employer takes mental health as seriously as physical health?", [
                "Yes", "No", "Don't know"
            ])
            obs_consequence = st.selectbox("Have you heard of or observed negative consequences for coworkers with mental health conditions?", [
                "No", "Yes"
            ])
        
        # Submit button
        submitted = st.form_submit_button("🔮 Get Prediction", type="primary", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'Age': age,
                'Gender': gender,
                'Continent': continent,
                'self_employed': self_employed,
                'family_history': family_history,
                'work_interfere': work_interfere,
                'no_employees': no_employees,
                'remote_work': remote_work,
                'tech_company': tech_company,
                'benefits': benefits,
                'care_options': care_options,
                'wellness_program': wellness_program,
                'seek_help': seek_help,
                'anonymity': anonymity,
                'leave': leave,
                'mental_health_consequence': mental_health_consequence,
                'phys_health_consequence': phys_health_consequence,
                'coworkers': coworkers,
                'supervisor': supervisor,
                'mental_health_interview': mental_health_interview,
                'phys_health_interview': phys_health_interview,
                'mental_vs_physical': mental_vs_physical,
                'obs_consequence': obs_consequence
            }
            
            # Make prediction
            try:
                result = predictor.predict(input_data)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                if result['prediction'] == 1:
                    st.markdown(f"""
                    <div class="prediction-card positive-result">
                        <h2>⚠️ Likely to Seek Treatment</h2>
                        <h3>{result['probability_treatment']:.1%} probability</h3>
                        <p>Based on your responses, the model predicts you are likely to seek mental health treatment.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("If you're considering mental health support, that's a positive step. Consider reaching out to mental health professionals, employee assistance programs, or trusted healthcare providers.")
                    
                else:
                    st.markdown(f"""
                    <div class="prediction-card negative-result">
                        <h2 style="color: #1b5e20;">✅ Unlikely to Seek Treatment</h2>
                        <h3 style="color: #2e7d32;">{result['probability_no_treatment']:.1%} probability</h3>
                        <p style="color: #1b5e20;">Based on your responses, the model predicts you are unlikely to seek mental health treatment.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("Remember that seeking mental health support when needed is always beneficial. Don't hesitate to reach out to professionals if you ever feel you could benefit from support.")
                
                # Simple probability chart
                st.markdown("### Confidence Breakdown")
                prob_data = {
                    'Outcome': ['Unlikely to Seek Treatment', 'Likely to Seek Treatment'],
                    'Probability': [result['probability_no_treatment'], result['probability_treatment']],
                    'Color': ['#4caf50', '#f44336']
                }
                
                fig = go.Figure(data=[go.Bar(
                    x=prob_data['Outcome'],
                    y=prob_data['Probability'],
                    marker_color=prob_data['Color'],
                    text=[f"{p:.1%}" for p in prob_data['Probability']],
                    textposition='auto',
                )])
                
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis_title="Probability",
                    xaxis_title="Prediction",
                    yaxis=dict(range=[0, 1], tickformat='.0%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mental Health Improvement Suggestions
                st.markdown("---")
                st.markdown("## 💡 Mental Health Improvement Suggestions")
                
                # Personalized suggestions based on responses
                suggestions = []
                
                # Work-related suggestions
                if work_interfere in ['Sometimes', 'Often']:
                    suggestions.append("**Work-Life Balance**: Consider speaking with your supervisor about workload management or flexible work arrangements.")
                
                if benefits == "Don't know" or care_options in ["No", "Not sure"]:
                    suggestions.append("**Explore Benefits**: Research your employer's mental health benefits and available resources through HR.")
                
                if wellness_program == "No":
                    suggestions.append("**Workplace Wellness**: Suggest implementing mental health wellness programs at your workplace.")
                
                if anonymity == "No":
                    suggestions.append("**Privacy Advocacy**: Advocate for better privacy protections for mental health support at work.")
                
                # Personal suggestions
                if family_history == "Yes":
                    suggestions.append("**Family History Awareness**: Being aware of family mental health history can help you monitor your own mental health proactively.")
                
                if mental_health_consequence == "Yes":
                    suggestions.append("**Stigma Reduction**: Consider joining mental health advocacy groups to help reduce workplace stigma.")
                
                # General suggestions for everyone
                suggestions.extend([
                    "**Regular Self-Care**: Establish daily routines that include physical activity, adequate sleep, and relaxation techniques.",
                    "**Professional Support**: Don't hesitate to reach out to mental health professionals when needed - it's a sign of strength, not weakness.",
                    "**Social Connections**: Maintain strong relationships with friends, family, and colleagues who support your wellbeing.",
                    "**Mindfulness Practice**: Consider meditation, yoga, or other mindfulness practices to manage stress and improve mental clarity.",
                    "**Healthy Boundaries**: Learn to set appropriate boundaries between work and personal life.",
                    "**Education**: Stay informed about mental health topics and resources available to you."
                ])
                
                # Display suggestions in a nice format
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Personalized Tips")
                    for i, suggestion in enumerate(suggestions[:len(suggestions)//2]):
                        st.markdown(f"• {suggestion}")
                
                with col2:
                    st.markdown("### 🌟 General Wellness")
                    for suggestion in suggestions[len(suggestions)//2:]:
                        st.markdown(f"• {suggestion}")
                
                # Disclaimer
                st.markdown("---")
                st.warning("""
                **Important Disclaimer:** This prediction is for educational purposes only and should not be used as a substitute for professional medical advice. 
                Always consult qualified mental health professionals for actual treatment decisions.
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please try again or contact support if the error persists.")

if __name__ == "__main__":
    main()