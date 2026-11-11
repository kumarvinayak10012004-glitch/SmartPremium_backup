\# 💰 SmartPremium – Insurance Premium Prediction Using Machine Learning



\## 📌 Project Overview



SmartPremium is a Machine Learning-based project that predicts insurance premium amounts based on customer information and policy-related details.



The project uses multiple regression algorithms to estimate insurance costs and compares their performance using RMSE (Root Mean Squared Error). The best-performing model is selected for making predictions through a Streamlit web application.



\## 🎯 Problem Statement



Insurance premium calculation depends on several factors, such as age, annual income, health score, smoking status, previous claims, credit score, and policy type.



This project aims to use Machine Learning techniques to analyze these factors and predict the estimated insurance premium.



\## 🧠 Skills Gained



\- Exploratory Data Analysis (EDA)

\- Data Cleaning and Preprocessing

\- Handling Missing Values using SimpleImputer

\- Categorical Encoding using OneHotEncoder

\- Feature Selection

\- Regression Model Training

\- Model Evaluation using RMSE

\- Model Comparison

\- Model Saving using Joblib

\- Streamlit Application Development

\- Git and GitHub



\## 🏷️ Domain



Insurance | Finance | Data Science | Machine Learning



\## 💼 Business Use Cases



\- Estimated insurance premium calculation

\- Customer risk analysis

\- Insurance pricing support

\- Data-driven decision-making

\- Understanding factors affecting insurance costs



\## 📊 Dataset Information



\*\*Dataset:\*\* Kaggle Playground Series – Season 4, Episode 12



\*\*Target Variable:\*\* `Premium Amount`



\*\*Features Used:\*\*

\- Age

\- Gender

\- Annual Income

\- Health Score

\- Smoking Status

\- Exercise Frequency

\- Previous Claims

\- Vehicle Age

\- Credit Score

\- Insurance Duration

\- Policy Type



\## ⚙️ Project Approach



1\. \*\*Data Loading:\*\* Loaded the insurance dataset using Pandas.

2\. \*\*EDA:\*\* Explored the dataset and understood the distribution of features.

3\. \*\*Data Cleaning:\*\* Identified and handled missing values.

4\. \*\*Preprocessing:\*\* Applied numerical and categorical imputation.

5\. \*\*Encoding:\*\* Used OneHotEncoder for categorical features.

6\. \*\*Model Training:\*\* Trained four regression algorithms.

7\. \*\*Model Evaluation:\*\* Compared models using RMSE.

8\. \*\*Best Model Selection:\*\* Selected XGBoost based on the lowest RMSE.

9\. \*\*Model Saving:\*\* Saved the trained model and preprocessing components using Joblib.

10\. \*\*Deployment Interface:\*\* Developed a Streamlit application for insurance premium prediction.



\## 📈 Model Comparison \& Results



Four Machine Learning regression algorithms were evaluated.



| Algorithm | RMSE |

|---|---:|

| Linear Regression | 863.3382 |

| Decision Tree | 1233.3456 |

| Random Forest | 859.1960 |

| XGBoost | \*\*847.8424\*\* |



\*\*Best Performing Model: XGBoost\*\*



XGBoost achieved the lowest RMSE among the four tested models.



\## 📏 Evaluation Metric



\*\*RMSE (Root Mean Squared Error):\*\*



RMSE measures the difference between actual and predicted values. A lower RMSE indicates better prediction performance on the evaluated dataset.



\## 🛠️ Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- XGBoost

\- Joblib

\- Streamlit

\- Matplotlib

\- Seaborn

\- Git \& GitHub



\## 📦 Project Deliverables



\- Data preprocessing pipeline

\- Trained regression models

\- Model comparison results

\- Saved model and preprocessing artifacts

\- Streamlit-based prediction application

\- GitHub project repository



\## ▶️ How to Run the Project



Clone the repository:



```bash

git clone https://github.com/kumarvinayak10012004-glitch/SmartPremium\_backup.git

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the Streamlit application:



```bash

python -m streamlit run app/streamlit\_app.py

```



\## 👨‍💻 Author



\*\*Vinayak Kumar\*\*



GitHub: \[kumarvinayak10012004-glitch](https://github.com/kumarvinayak10012004-glitch)



\---



⭐ If you find this project useful, consider giving it a star!

