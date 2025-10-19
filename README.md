# Diabetes Hospital Readmission Analysis

## 🏥 Project Overview

This project analyzes hospital readmission patterns for diabetic patients using the Diabetes 130-US hospitals dataset. The goal is to identify factors that contribute to hospital readmissions and build predictive classification models to help healthcare providers reduce readmission rates.

### Key Features
- **Interactive Streamlit Dashboard**: Comprehensive data visualization and exploration interface
- **Exploratory Data Analysis**: In-depth analysis of patient demographics, medical conditions, and treatment patterns
- **Predictive Modeling**: Multiple machine learning models to predict 30-day hospital readmissions
- **Model Comparison**: Various modeling approaches including target encoding and non-target encoding strategies

### Dataset
The project uses the **Diabetes 130-US hospitals dataset** (`diabetic_data.csv`), which contains:
- 101,766 hospital encounters
- 50+ features including patient demographics, diagnoses, medications, and laboratory results
- Data from 130 US hospitals over 10 years (1999-2008)

---

## 📁 Project Structure

```
stage_1/
├── app.py                          # Streamlit dashboard application
├── requirements.txt                 # Python dependencies
├── run.sh                          # Shell script to launch the app
├── LICENSE                         # Project license
├── README.md                       # This file
└── notebooks/
    ├── diabetic_data.csv                                      # Dataset
    ├── healthcare_dataset_eda_v1.ipynb                        # Initial EDA
    ├── heatlhcare_dataset_mileston2_v1_targetenc.ipynb       # Model v1 with target encoding
    ├── heatlhcare_dataset_mileston2_v2_nottargetenc.ipynb    # Model v2 without target encoding
    ├── heatlhcare_dataset_mileston2_final_run1.ipynb         # Final model run 1
    ├── heatlhcare_dataset_mileston2_final_run2.ipynb         # Final model run 2
    ├── heatlhcare_dataset_mileston2_patientlevel_proof.ipynb # Patient-level analysis
    └── heatlhcare_dataset_mileston2_reproduce.ipynb          # Reproducibility verification
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd /folder
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Using the provided shell script
```bash
chmod +x run.sh
./run.sh
```

#### Option 2: Direct streamlit command
```bash
streamlit run app.py
```

#### Option 3: Custom port
```bash
streamlit run app.py --server.port 8080
```

The application will open in your default web browser at `http://localhost:8501`

---

## 📊 Notebooks Overview

### 1. Exploratory Data Analysis
- **`healthcare_dataset_eda_v1.ipynb`**
  - Initial data exploration and visualization
  - Feature distribution analysis
  - Missing value assessment
  - Correlation analysis

### 2. Modeling Notebooks

#### Target Encoding Approach
- **`heatlhcare_dataset_mileston2_v1_targetenc.ipynb`**
  - Machine learning models using target encoding for categorical variables
  - Feature engineering with target statistics
  
#### Non-Target Encoding Approach
- **`heatlhcare_dataset_mileston2_v2_nottargetenc.ipynb`**
  - Alternative encoding strategies (one-hot encoding, label encoding)
  - Comparison with target encoding approach

#### Final Models
- **`heatlhcare_dataset_mileston2_final_run1.ipynb`**
  - First final model iteration
  - Hyperparameter tuning
  - Model evaluation metrics

- **`heatlhcare_dataset_mileston2_final_run2.ipynb`**
  - Second final model iteration
  - Additional model refinements
  - Performance comparison

### 3. Analysis and Validation
- **`heatlhcare_dataset_mileston2_patientlevel_proof.ipynb`**
  - Patient-level analysis to validate model assumptions
  - Ensures proper data splitting at patient level

- **`heatlhcare_dataset_mileston2_reproduce.ipynb`**
  - Reproducibility verification of a research paper
  - Result validation across different runs

---

## 🎯 Key Findings

The analysis explores various factors affecting hospital readmission, including:
- Patient demographics (age, gender, race)
- Medical history (number of diagnoses, previous admissions)
- Treatment patterns (medications, procedures)
- Hospital stay characteristics (length of stay, number of procedures)

---

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning models and preprocessing
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Jupyter Notebook**: Interactive development environment

---

## 📝 Usage

### Streamlit Dashboard
The dashboard provides interactive exploration of:
1. **Data Overview**: Dataset statistics and sample records
2. **Exploratory Analysis**: Feature distributions and relationships
3. **Model Performance**: Prediction results and evaluation metrics
4. **Feature Importance**: Key factors affecting readmissions

### Notebooks
To run the Jupyter notebooks:
```bash
jupyter notebook
```
Navigate to the `notebooks/` folder and open the desired notebook.

---

## 🤝 Contributing

This is an academic project for IT5006 Fundamentals of Data Analytics at NUS. 

---

## 📄 License

See the LICENSE file for details.

---

## 📚 References

- Dataset: Diabetes 130-US hospitals for years 1999-2008
- Source: UCI Machine Learning Repository
