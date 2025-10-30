# Technical Documentation: Mental Health in Tech Predictor



## 🎯 Project Overview



A production-ready machine learning web application that predicts mental health treatment likelihood based on workplace factors. Achieves 80.95% accuracy matching the original Jupyter notebook analysis.



## 🏗️ Architecture



### Core Components

1. **`model.py`** - ML pipeline with exact notebook replication

2. **`app.py`** - Streamlit web interface (simplified single-page design)

3. **`survey.csv`** - Training dataset (1,261 tech worker responses)

4. **`mental_health_final (1).ipynb`** - Original research analysis



### Technical Stack

- **Backend**: Python 3.8+, scikit-learn, pandas, numpy

- **Frontend**: Streamlit with custom CSS styling

- **Model**: Random Forest Classifier (n_estimators=100, random_state=42)

- **Environment**: Isolated virtual environment with pip dependencies



### Performance Metrics

``````

Accuracy: 80.95%

Recall: 78.91%

F1-Score: 80.80%

ROC-AUC: 85.21%

``````



### Feature Engineering

- 22 engineered features from 27 original survey columns
- Single LabelEncoder for consistent categorical encoding
- Train-test split: random_state=39 for reproducibility



## 🖥️ User Interface Evolution



### Initial Implementation

- Multi-page Streamlit app with navigation sidebar

- Data analysis, model performance, and prediction pages

- Complex feature exploration and visualization tools



### User-Driven Simplification

- Simplified to single-page prediction interface

- Direct age number input (replaced radio buttons)

- High contrast colors for accessibility (#1b5e20, #2e7d32)

- Personalized mental health improvement suggestions

- Emergency resources and professional guidance



## 🛠️ Development Workflow

  

### Setup & Configuration 

1. Virtual environment creation and dependency management  

2. Model extraction from Jupyter notebook analysis 

3. Streamlit application development and testing  

4. Performance optimization and exact notebook matching

### 5. **Project Documentation** ✓

### Key Problem-Solving Moments- Comprehensive README.md with setup instructions

1. **Performance Discrepancy**: Discovered notebook vs. terminal accuracy difference (80.95% vs 78.97%)- Requirements.txt for dependency management

2. **Root Cause**: Different LabelEncoder usage patterns between implementations- Startup scripts for easy execution

3. **Solution**: Replicated exact notebook preprocessing with single LabelEncoder instance- Project structure documentation

4. **Validation**: Achieved perfect performance match

### 6. **Testing & Integration** ✓

### UI/UX Refinements- Model training tested successfully

1. User feedback: Complex navigation was unnecessary- Streamlit application runs without errors

2. Simplification: Single-page form-focused interface- All features functional and integrated

3. Accessibility: High contrast colors, clear typography- Performance metrics: 78.97% accuracy, 78.12% recall

4. Functionality: Direct age input, personalized suggestions

## 🎯 Key Features Implemented

## 📋 File Structure & Dependencies

### **Prediction System**

### Core Files- Clean, single-page interface with organized form sections

```

├── app.py                          # Streamlit web application- All 22 prediction features with appropriate input types

├── model.py                        # ML pipeline and preprocessing- Real-time mental health treatment prediction

├── survey.csv                      # Training dataset- Clear probability scores and confidence visualization

├── requirements.txt                # Python dependencies- Helpful guidance and professional disclaimers

├── mental_health_model.pkl         # Trained model (auto-generated)

└── mental_health_final (1).ipynb   # Original analysis### **User Experience**

```- Simplified, focused interface (no complex navigation)

- Mobile-responsive design

### Supporting Files- Automatic light/dark theme support via browser settings

```- Clear visual feedback and results presentation

├── README.md                       # Comprehensive documentation- Educational disclaimers and guidance

├── PROJECT_SUMMARY.md              # Technical overview (this file)

├── .gitignore                      # Git version control setup## 🚀 How to Run

├── run_app.bat                     # Windows start script

└── stop_app.bat                    # Windows stop script### **Simple Method (Recommended)**

``````bash

# Double-click on run_app.bat

### Dependencies# OR from command line:

```pythonrun_app.bat

streamlit==1.28.1```

pandas==2.1.3

numpy==1.24.3### **Manual Method**

scikit-learn==1.3.2```bash

plotly==5.17.0# 1. Navigate to project directory

joblib==1.3.2cd "C:\Users\deepa rajesh\OneDrive\Desktop\mental_health_in_tech"

```

# 2. Activate virtual environment
venv\Scripts\activate
# 3. Run Streamlit app

## 🚀 Deployment & Usage



### Local Development

```bashstreamlit run app.py

# Activate virtual environment```

venv\Scripts\activate  # Windows

source venv/bin/activate  # macOS/Linux### **Stop Application**

```bash

# Install dependencies# Use stop script

pip install -r requirements.txtstop_app.bat

# OR press Ctrl+C in terminal

# Run application```

streamlit run app.py

```## 📊 Technical Specifications



### Production Considerations### **Model Details**

- Model file auto-generation on first run- **Algorithm**: Random Forest Classifier (100 estimators)

- No external API dependencies (fully offline)- **Training Data**: 1,200+ survey responses

- Minimal memory footprint (~50MB total)- **Features**: 22 engineered features

- Cross-platform compatibility (Windows, macOS, Linux)- **Performance**: 78.97% accuracy, 78.12% recall

- **Validation**: Train-test split with stratification

## 🔍 Quality Assurance

### **Application Architecture**

### Model Validation- **Frontend**: Streamlit web framework

- ✅ Exact accuracy match with original notebook (80.95%)- **Backend**: Python with scikit-learn

- ✅ Consistent predictions across sessions- **Data Processing**: Pandas, NumPy

- ✅ Proper feature encoding and preprocessing- **Visualization**: Plotly, Matplotlib, Seaborn

- ✅ Robust error handling for edge cases- **Deployment**: Local development server



### User Experience Testing### **Key Components**

- ✅ Intuitive single-page interface1. `app.py` - Main Streamlit application

- ✅ Clear prediction results with probability scores2. `model.py` - ML model and preprocessing

- ✅ Accessible design with high contrast colors3. `survey.csv` - Training dataset

- ✅ Comprehensive mental health resources and disclaimers4. `mental_health_model.pkl` - Trained model file

- ✅ Mobile-responsive design5. `requirements.txt` - Dependencies

6. `README.md` - Documentation

### Code Quality

- ✅ Clean, documented Python code## 🎖️ Achievement Highlights

- ✅ Modular architecture (separate model and UI logic)

- ✅ Professional Git repository structure✅ **Complete ML Pipeline**: From Jupyter notebook to production-ready app

- ✅ Comprehensive README and documentation✅ **Virtual Environment**: Proper isolation and dependency management  

✅ **Interactive Frontend**: Professional web interface with multiple features

## 🎓 Learning Outcomes✅ **Model Performance**: Achieved ~79% accuracy with good recall

✅ **Documentation**: Comprehensive setup and usage instructions

### Technical Skills Demonstrated✅ **Easy Deployment**: One-click startup with batch scripts

1. **ML Pipeline Development**: End-to-end model deployment✅ **Data Visualization**: Rich, interactive charts and analysis

2. **Web Application Development**: Streamlit expertise✅ **User Experience**: Intuitive interface with clear results

3. **Data Science**: Feature engineering and model validation

4. **Software Engineering**: Clean code, documentation, version control## 🔮 Application URL

5. **User Experience**: Iterative design based on feedbackOnce running: **http://localhost:8501**



### Best Practices Applied## 📁 Final Project Structure

1. **Reproducibility**: Fixed random seeds, documented preprocessing```

2. **Maintainability**: Modular code structure, clear documentationmental_health_in_tech/

3. **Usability**: User-centered design, accessibility considerations
├── 📱 app.py                    # Main Streamlit application

4. **Professional Standards**: Git-ready repository, comprehensive README
├── 🤖 model.py                  # ML model and preprocessing

├── 📊 survey.csv                # Training dataset  

---├── 📔 mental_health_final (1).ipynb  # Original notebook

├── 🎯 mental_health_model.pkl   # Trained model

**Project Status**: ✅ **Complete** - Production-ready application with exact notebook performance match and user-friendly interface.
├── 📋 requirements.txt          # Dependencies

├── 📖 README.md                 # Documentation

**Next Steps**: Ready for Git version control, further feature development, or deployment to cloud platforms.├── 🚀 run_app.bat              # Startup script
├── 🛑 stop_app.bat             # Stop script
├── 📄 PROJECT_SUMMARY.md       # This summary
└── 📁 venv/                    # Virtual environment
```

---

