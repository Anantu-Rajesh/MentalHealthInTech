# Mental Health in Tech Predictor# Mental Health in Tech Predictor# Mental Health in Tech Predictor



A machine learning web application that predicts whether someone may seek mental health treatment based on workplace and personal factors. Built with Python, Streamlit, and Scikit-learn, using a Random Forest Classifier (≈80.9% accuracy).



## Quick StartA machine learning web application that predicts the likelihood of someone seeking mental health treatment based on workplace and personal factors. Built using Random Forest Classifier with 80.95% accuracy.A machine learning web application that predicts whether someone might seek mental health treatment based on workplace and personal factors.



### 1. Clone the repository

```bash

git clone https://github.com/yourusername/mental_health_in_tech.git## 🚀 Quick Start## 🚀 Quick Start

cd mental_health_in_tech

```



### 2. Create and activate virtual environment### Prerequisites### Prerequisites

```bash

python -m venv venv- Python 3.8 or higher- Python 3.8 or higher



# Windows- pip (Python package installer)- pip (Python package installer)

venv\Scripts\activate



# macOS/Linux

source venv/bin/activate### Installation & Setup### Installation & Setup

```



### 3. Install dependencies

```bash1. **Clone or download this project**1. **Clone or download this project**

pip install -r requirements.txt

```   ```bash   ```bash



### 4. Run the app   cd mental_health_in_tech   cd mental_health_in_tech

```bash

streamlit run app.py   ```   ```

```



App will open automatically at http://localhost:8501

2. **Create and activate virtual environment**2. **Create and activate virtual environment**

## Project Overview

   ```bash   ```bash

- Predicts likelihood of seeking mental health treatment using workplace and personal data

- Built on OSMI Mental Health in Tech Survey (2014)   # Create virtual environment   # Create virtual environment

- Features 22 engineered inputs from 1,200+ responses

- Provides probability-based predictions and recommendations   python -m venv venv   python -m venv venv

- Works entirely offline — no data storage or tracking

      

## Project Structure

   # Activate virtual environment   # Activate virtual environment

```

mental_health_in_tech/   # On Windows:   # On Windows:

│

├── app.py                     # Main Streamlit application   venv\Scripts\activate   venv\Scripts\activate

├── model.py                   # ML model and preprocessing logic

├── requirements.txt            # Python dependencies   # On macOS/Linux:   # On macOS/Linux:

├── mental_health_model.pkl     # Trained model (auto-generated)

├── survey.csv                  # Training dataset (1,261 responses)   source venv/bin/activate   source venv/bin/activate

├── run_app.bat                 # Windows start script

├── stop_app.bat                # Windows stop script   ```   ```

├── PROJECT_SUMMARY.md          # Technical documentation

├── mental_health_final.ipynb   # Original analysis notebook

├── venv/                       # Virtual environment

└── README.md                   # Project documentation3. **Install dependencies**3. **Install dependencies**

```

   ```bash   ```bash

## Technical Details

   pip install -r requirements.txt   pip install -r requirements.txt

- **Model**: Random Forest Classifier

- **Accuracy**: ~80.95%   ```   ```

- **Libraries**: Scikit-learn, Pandas, NumPy, Streamlit

- **Visualization**: Plotly, Matplotlib, Seaborn

- **Frontend**: Streamlit UI

- **Training Data**: OSMI Mental Health in Tech Survey (2014)4. **Run the application**4. **Run the application**



See `mental_health_final.ipynb` for detailed analysis and model development.   ```bash   ```bash



## Future Improvements   streamlit run app.py   streamlit run app.py



- Integration of newer survey datasets   ```   ```

- Comparison with advanced models (XGBoost, SVM)

- Enhanced feature engineering and UI design

- Multi-language support

5. **Open your browser**5. **Open your browser**

## Disclaimer

   The app will automatically open at `http://localhost:8501`   The app will automatically open at `http://localhost:8501`

This project is for educational and research purposes only. It does not provide medical advice or professional diagnosis.



Developed to promote awareness of mental health in the tech industry through data-driven insights.
### Windows Users - Quick Start Scripts## 📁 Project Structure

- **Start App**: Double-click `run_app.bat`

- **Stop App**: Double-click `stop_app.bat````

mental_health_in_tech/

## 📁 Project Structure├── app.py                          # Main Streamlit application

├── model.py                        # ML model and preprocessing logic

```├── survey.csv                      # Training dataset

mental_health_in_tech/├── mental_health_final (1).ipynb   # Original Jupyter notebook

├── app.py                          # Main Streamlit application├── requirements.txt                # Python dependencies

├── model.py                        # ML model and preprocessing logic├── mental_health_model.pkl         # Trained model (generated on first run)

├── survey.csv                      # Training dataset (1,261 responses)├── venv/                           # Virtual environment

├── mental_health_final (1).ipynb   # Original analysis notebook└── README.md                       # This file

├── requirements.txt                # Python dependencies```

├── mental_health_model.pkl         # Trained model (auto-generated)

├── run_app.bat                     # Windows start script## 🎯 Features

├── stop_app.bat                    # Windows stop script

├── PROJECT_SUMMARY.md              # Technical documentation### 🔮 Simple Prediction Interface

├── venv/                           # Virtual environment- Clean, user-friendly form with all relevant mental health factors

└── README.md                       # This file- Direct age input (number field for easy typing)

```- Real-time prediction with probability scores  

- Clear, easy-to-understand results display with improved visibility

## 🎯 Features- Personalized mental health improvement suggestions

- Emergency resources and helpful links

### 🔮 Simple Prediction Interface- Professional guidance and disclaimers

- Clean, single-page form with all relevant mental health factors

- Direct age input (number field for easy typing)### 💡 Theme Support

- Real-time prediction with probability scores- Automatic light/dark mode based on browser settings

- Clear, accessible results display with high contrast colors- Clean, accessible design for all users

- Personalized mental health improvement suggestions

- Emergency resources and professional guidance### 📱 Mobile-Friendly

- Medical disclaimers and safety information- Responsive design that works on all devices

- Optimized for both desktop and mobile use

### 💡 Accessibility & Design

- High contrast colors for better visibility## 🤖 Model Details

- Responsive design for all screen sizes

- Simple navigation focused on core functionality- **Algorithm**: Random Forest Classifier

- Professional, clean interface- **Performance**: ~80.95% Accuracy, ~78.91% Recall (matching notebook performance)

- **Features**: 22 engineered features from survey responses

## 🤖 Model Performance- **Training Data**: 1,200+ survey responses from tech industry employees



- **Algorithm**: Random Forest Classifier### Key Predictive Factors

- **Accuracy**: 80.95% (exactly matching original notebook)1. Family history of mental illness

- **Recall**: 78.91%2. Work interference from mental health conditions

- **F1-Score**: 80.80%3. Company mental health benefits

- **ROC-AUC**: 85.21%4. Workplace attitudes and culture

- **Features**: 22 engineered features from survey responses5. Company size and work environment

- **Training Data**: 1,261 survey responses from tech industry employees

## 📊 Dataset

### Key Predictive Factors

1. Family history of mental illnessThe model is trained on a comprehensive survey dataset containing:

2. Work interference from mental health conditions- **Personal factors**: Age, gender, family history

3. Company mental health benefits and resources- **Work environment**: Company size, remote work status, tech company classification

4. Workplace attitudes toward mental health discussions- **Company support**: Mental health benefits, wellness programs, resources

5. Company size and work environment- **Workplace culture**: Attitudes towards mental health discussions, perceived consequences

6. Previous mental health treatment history

## ⚠️ Important Disclaimers

## 📊 Dataset Information

- **Educational Purpose Only**: This application is designed for educational and demonstration purposes

The model is trained on a comprehensive survey dataset containing:- **Not Medical Advice**: Predictions should not be used as a substitute for professional medical advice

- **Consult Professionals**: Always consult qualified mental health professionals for actual treatment decisions

**Personal Factors** (5 features):- **Data Limitations**: Model predictions are based on survey data and may not reflect individual circumstances

- Age, gender, family history of mental illness

- Country of residence, self-employed status## 🛠️ Technical Stack



**Work Environment** (8 features):- **Frontend**: Streamlit (Interactive web interface)

- Company size, tech company classification- **Machine Learning**: Scikit-learn (Random Forest Classifier)

- Remote work availability, number of employees- **Data Processing**: Pandas, NumPy

- Role within company, work location- **Visualization**: Plotly, Matplotlib, Seaborn

- **Model Persistence**: Joblib

**Company Support** (9 features):- **Environment**: Python 3.8+ with virtual environment

- Mental health benefits, wellness programs

- Resources availability, ease of leave## 🔧 Development

- Mental health coverage knowledge

- Anonymity protection, medical leave options### Adding New Features

1. Model improvements can be made in `model.py`

**Workplace Culture** (5 features):2. UI enhancements can be added to `app.py`

- Attitudes toward mental health discussions3. Additional analysis can be included in the data analysis section

- Perceived consequences of disclosure

- Supervisor and coworker support### Model Retraining

- Interview discussion comfort levelsThe model automatically trains on first run. To retrain:

1. Delete `mental_health_model.pkl`

## ⚠️ Important Disclaimers2. Restart the application

3. The model will retrain automatically

- **Educational Purpose Only**: This application is for demonstration and educational purposes

- **Not Medical Advice**: Predictions should never replace professional medical consultation## 📝 Usage Instructions

- **Consult Professionals**: Always seek qualified mental health professionals for treatment decisions

- **Emergency Situations**: If experiencing crisis, contact emergency services immediately1. **Navigate to Prediction Page**: Use the sidebar to access the prediction interface

- **Data Limitations**: Model based on 2014 survey data; individual circumstances vary significantly2. **Fill Out Form**: Complete all fields with relevant information

3. **Get Prediction**: Click the predict button to see results

## 🛠️ Technical Stack4. **Explore Analysis**: Use other pages to explore data insights and model performance

5. **Understand Results**: Review probability scores and feature importance

**Frontend**:

- Streamlit (Interactive web interface)## 🤝 Contributing

- HTML/CSS for styling

This project demonstrates:

**Machine Learning**:- End-to-end machine learning pipeline

- Scikit-learn (Random Forest Classifier)- Interactive web application development

- Pandas, NumPy (Data processing)- Data preprocessing and feature engineering

- Joblib (Model persistence)- Model evaluation and interpretation

- Production-ready deployment practices

**Development**:

- Python 3.8+ with virtual environment## 🔒 Privacy & Security

- Jupyter Notebook for initial analysis

- **No Data Storage**: User inputs are not stored or logged

## 🔧 Development Notes- **Local Processing**: All computations happen locally

- **No External Requests**: Application works entirely offline after setup

### Model Training Process

1. Data preprocessing with feature engineering## 📞 Support

2. Single LabelEncoder instance for all categorical features (critical for accuracy)

3. Train-test split with random_state=39For technical issues or questions:

4. Random Forest with random_state=42 for reproducibility1. Check that all dependencies are installed correctly

5. Model automatically saves as `mental_health_model.pkl`2. Ensure you're using the virtual environment

3. Verify Python version compatibility (3.8+)

### Retraining the Model4. Check the console for detailed error messages

To retrain with new data:

1. Replace `survey.csv` with new dataset---

2. Delete `mental_health_model.pkl`

3. Run `python model.py` or restart the Streamlit app**Built with ❤️ using Python, Streamlit, and Machine Learning**

### Adding New Features
- UI enhancements: Edit `app.py`
- Model improvements: Edit `model.py`
- New analysis: Check `mental_health_final (1).ipynb`

## 🔒 Privacy & Security

- **No Data Storage**: User inputs are processed in memory only
- **Local Processing**: All computations happen locally on your machine
- **No External Requests**: Application works completely offline after setup
- **No Tracking**: No analytics or user behavior tracking
- **Temporary Sessions**: All data cleared when browser session ends

## 📈 Usage Analytics

The application provides:
- Individual prediction with probability scores
- Feature importance understanding
- Personalized recommendations based on risk factors
- Educational resources about mental health in tech
- Professional guidance and emergency contacts

## 🤝 Contributing & Future Enhancements

**Potential Improvements**:
- Additional ML models (SVM, XGBoost)
- More recent survey data integration
- Extended feature engineering
- A/B testing for UI improvements
- Multi-language support

**Research Applications**:
- Workplace mental health policy analysis
- Industry-specific risk factor studies
- Intervention effectiveness measurement
- Longitudinal mental health tracking

## 📞 Support & Troubleshooting

**Common Issues**:
1. **Import Errors**: Ensure virtual environment is activated
2. **Model Not Found**: Run `python model.py` to generate model file
3. **Port Conflicts**: Change port with `streamlit run app.py --server.port 8502`
4. **Performance Issues**: Check Python version (3.8+ recommended)

**For Technical Support**:
- Check console output for detailed error messages
- Verify all dependencies in `requirements.txt` are installed
- Ensure sufficient disk space for model file (~1MB)

---

## 📋 Citation

If using this project for research or educational purposes:

```
Mental Health in Tech Predictor
Machine Learning Application for Workplace Mental Health Assessment
Based on OSMI Mental Health in Tech Survey Data (2014)
Implementation: Python, Scikit-learn, Streamlit
```

---

**Built with ❤️ for mental health awareness in the tech industry**

*This project aims to reduce stigma around mental health discussions in technology workplaces and promote early intervention through data-driven insights.*