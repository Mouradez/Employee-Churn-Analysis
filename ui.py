import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Machine Learning Web Application for predicting Employee Churn")
st.markdown("""
In the past, the focus has primarily been on 'rates' such as attrition and retention rates. HR Managers compute these rates and attempt to predict future rates using data warehousing tools. While these rates provide an aggregate impact of churn, they only tell half the story. Another approach is to focus on individual records in addition to the aggregate.

There are numerous case studies available on customer churn. In customer churn, you can predict who and when a customer will stop buying. Employee churn is similar to customer churn but focuses on employees rather than customers. Here, you can predict who and when an employee will terminate their service. Employee churn is expensive, and even incremental improvements can yield significant results. This helps in designing better retention plans and improving employee satisfaction.
""")
st.image('images/employee.jpg',width=750)

st.header("Prediction")
st.markdown("""
This machine learning model has been trained on HR documents about employees to accurately predict employee churn.
""")

st.title("Predicing Employee churn with Machine learning")
st.subheader("Please fill in the information below to get started")
satisfaction_level = st.slider('Satisfaction level',0.0,1.0,0.01) 
last_evaluation	= st.slider('Last evaluation',0.0,1.0,0.01)
number_project = st.slider('How many numbers of projects assigned to an employee ?',2,7,1)	
average_montly_hours = st.slider('How many average numbers of hours worked by an employee in a month ?',96,310,1)	
time_spend_company = st.slider('The number of years spent by an employee in the company',2,10,1)
Work_accident = st.radio('Had a work accident or not',['Yes','No'])
promotion_last_5years = st.radio('Had a promotion in the last 5 years or not',['Yes','No'])
Departments = st.selectbox('Deparments',['Sales', 'Technical', 'Support', 'IT', 'Product_mng', 'Marketing',
       'RandD', 'Accounting', 'HR', 'Management'])	
salary = st.selectbox('Salary level',['Low','Medium','High'])

department_mapping = {'Sales': 7,'Technical': 9,'Support': 8,'IT': 0,'Product_mng': 6,'Marketing': 5,'RandD': 1,'Accounting': 2,'HR': 3,'Management': 4}   
salary_mapping = {'Low': 1,'Medium': 2,'High': 0}

Work_accident_numeric = 1 if Work_accident == 'Yes' else 0
promotion_last_5years_numeric = 1 if promotion_last_5years == 'Yes' else 0
Departments_numeric = department_mapping[Departments]
salary_numeric = salary_mapping[salary]

st.header("Input")
st.write(f"Work Accident: {Work_accident}")
st.write(f"Promotion in Last 5 Years: {promotion_last_5years}")
st.write(f"Department: {Departments}")
st.write(f"Salary Level: {salary}")
st.write(f"Work_accident_numeric: {Work_accident_numeric}")
st.write(f"promotion_last_5years_numeric: {promotion_last_5years_numeric}")
st.write(f"Departments_numeric: {Departments_numeric}")
st.write(f"salary_numeric: {salary_numeric}")

# Features
features = [satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident_numeric,promotion_last_5years_numeric,Departments_numeric,salary_numeric]
features = np.array([features])
model = pickle.load(open('random_forest_classifier.pkl', 'rb'))

# Prediction
if st.button('Predict'):
    prediction = model.predict(features)
    if prediction[0] == 0:
        st.success('According to our calculations, the employee is likely to stay.')
    elif prediction[0] == 1:
        st.error('According to our calculations, the employee is likely to leave.')


