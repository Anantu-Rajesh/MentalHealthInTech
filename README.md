# Mental Health in Tech Predictor

A **machine learning web application** that predicts whether someone may seek mental health treatment based on workplace and personal factors.  
Built with **Python, Streamlit, and Scikit-learn**, using a **Random Forest Classifier** achieving approximately **80.9% accuracy**.

---

## 🚀 Quick Start



### 1. Clone the repository

```bash

git clone https://github.com/yourusername/mental_health_in_tech.git## 🚀 Quick Start## 🚀 Quick Start

cd mental_health_in_tech

```



### 2. Create and activate virtual environment

```bash

python -m venv venv- Python 3.8 or higher- Python 3.8 or higher



# Windows- pip (Python package installer)- pip (Python package installer)

venv\Scripts\activate



# macOS/Linux

source venv/bin/activate### Installation & Setup### Installation & Setup

```



### 3. Install dependencies

```bash

pip install -r requirements.txt

```  



### 4. Run the app   

```bash

streamlit run app.py 

```



App will open automatically at `http://localhost:8501`

---

## 🧩 Project Overview

   ```bash   

- Predicts likelihood of seeking mental health treatment using workplace and personal data

- Built on OSMI Mental Health in Tech Survey (2014)  

- Features 22 engineered inputs from 1,200+ responses

- Provides probability-based predictions and recommendations   

- Works entirely offline — no data storage or tracking

 ```     

---

## 📁 Project Structure

```bash

mental_health_in_tech/   

│

├── app.py                     # Main Streamlit application   

├── model.py                   # ML model and preprocessing logic

├── requirements.txt            # Python dependencies  

├── mental_health_model.pkl     # Trained model (auto-generated)

├── survey.csv                  # Training dataset (1,261 responses)   

├── run_app.bat                 # Windows start script

├── stop_app.bat                # Windows stop script   

├── PROJECT_SUMMARY.md          # Technical documentation

├── mental_health_final.ipynb   # Original analysis notebook

├── venv/                       # Virtual environment

└── README.md                   # Project documentation

```

---   

## ⚙️ Technical Details



- **Model**: Random Forest Classifier

- **Accuracy**: ~80.95%   

- **Libraries**: Scikit-learn, Pandas, NumPy, Streamlit

- **Visualization**: Plotly, Matplotlib, Seaborn

- **Frontend**: Streamlit UI

- **Training Data**: OSMI Mental Health in Tech Survey (2014)4. 



See `mental_health_final.ipynb` for detailed analysis and model development.   

---

## 💡 Features



- Real-time predictions with confidence scores.

- Clean, responsive web interface (using Streamlit)

- Educational insights and recommendations

- Offline functionality  (no API calls or data uploads)

- Lightweight setup, runs locally within minutes

---

## 🔧 Future Improvements   



- Integration of newer survey datasets   

- Enhanced feature engineering and UI design

- Multi-language support

---

## ⚠️ Disclaimer
   


- This project is for educational purposes only. 
- It does not provide medical advice or professional diagnosis.
- Developed to promote awareness of mental health in the tech industry through data-driven insights.



```

