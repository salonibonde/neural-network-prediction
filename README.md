# WiDS Datathon 2024: Predicting Metastatic Diagnosis Period

## Overview
This repository contains the code and data used for the WiDS Datathon 2024 challenge, where the goal is to predict the metastatic diagnosis period (in days) for breast cancer patients using patient characteristics, treatment, geo-demographic, and climate data.

The project utilizes machine learning and time series analysis techniques to address messy data, predict missing values, and model the target outcome based on real-world healthcare data.

## Why Does This Issue Matter?
Metastatic breast cancer is an advanced form of cancer that significantly impacts patient outcomes, treatment strategies, and healthcare resource allocation. Accurately predicting the time period when breast cancer becomes metastatic is crucial for timely interventions, improving treatment plans, and providing personalized patient care.

Early identification of high-risk patients allows healthcare providers to monitor and treat more aggressively, improving the chances of slowing disease progression. Additionally, understanding how factors such as geographic and environmental variables (like climate data) impact metastatic diagnosis can inform broader public health policies, research, and resource distribution.

This challenge not only enhances our ability to address real-world problems in healthcare but also pushes forward the integration of data science in life-saving medical predictions.

## Dataset
The dataset contains healthcare records of patients with metastatic breast cancer, including demographic information, diagnosis codes, and geographic characteristics. Additionally, the dataset is enriched with zip code-level climate data to explore the relation between health outcomes and environmental factors.

### Key Columns
- `patient_id`: Unique identifier for each patient.
- `patient_race`: Race of the patient (e.g., Asian, African American, etc.).
- `payer_type`: Insurance type at the time of metastatic diagnosis.
- `bmi`: Body Mass Index of the patient.
- `metastatic_diagnosis_period`: Target variable, the period in days between breast cancer and metastatic cancer diagnosis.
- **Climate Data**: Zip-code-level climate data, including monthly average temperatures.

### Files
- `train.csv`: Training dataset with features and target (metastatic diagnosis period).
- `test.csv`: Test dataset with features (target withheld).
- `submission.csv`: Example submission file showing the required format.

## Methods and Techniques
- **Data Preprocessing**:
  - Handling missing values through interpolation and imputation.
  - Feature engineering: Processing and transforming raw data (e.g., encoding categorical variables, normalizing numerical features).
  - Time series data handling for climate features.
  
- **Modeling**:
  - **LSTM (Long Short-Term Memory)** neural networks for time series prediction.
  - **ARIMA** models for time-dependent data analysis.
  - **Scikit-learn** regression models (e.g., Linear Regression, Random Forest) for baseline comparisons.
  
- **Evaluation**:
  - Root Mean Squared Error (RMSE) is used to evaluate model performance on the validation data.
  
## Results
The model predicts the metastatic diagnosis period for patients using patient, geographic, and climate data. The predictions are submitted in the required format (`submission.csv`) for evaluation against the leaderboard.

## How to Run the Code

### Prerequisites
You need to install the following dependencies:
- Python 3.x
- pandas, numpy, scikit-learn, tensorflow, statsmodels, matplotlib, seaborn

### Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/wids-metastatic-diagnosis-prediction.git
   cd wids-metastatic-diagnosis-prediction
   ```

2. **Data Preparation**:
   Download and place the `train.csv` and `test.csv` files into the `data/` directory.

3. **Run the Jupyter Notebook**:
   Open the notebook and run the analysis:
   ```bash
   jupyter notebook notebook/metastatic_diagnosis_prediction.ipynb
   ```

4. **Generate Predictions**:
   Once the notebook is executed, the prediction results for the test set will be saved in `submission/submission.csv`.

## Tools and Technologies
- **Python**: Main language used for analysis and modeling.
- **Jupyter Notebook**: For interactive coding and data visualization.
- **Pandas, NumPy**: Data manipulation and cleaning.
- **Scikit-learn**: Machine learning models and evaluation.
- **TensorFlow/Keras**: LSTM neural network implementation.
- **Statsmodels**: ARIMA modeling for time series data.
- **Matplotlib, Seaborn**: Data visualization.

## Acknowledgments
This project was developed as part of the [WiDS Datathon 2024](https://widsdatathon.org). The dataset is provided by Health Verity and enriched with US Zip Code data.
