import streamlit as st
from streamlit_option_menu import option_menu
import requests
from datetime import datetime
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
import mlflow
from employee_attrition_mlops.config import (
    DATABASE_URL_PYMSSQL, MLFLOW_TRACKING_URI, PRODUCTION_MODEL_NAME,
    DB_BATCH_PREDICTION_TABLE, DB_HISTORY_TABLE, EMPLOYEE_ID_COL, SNAPSHOT_DATE_COL
)

# Configure page
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide"
)

# Title of the app
st.title("Employee Attrition Prediction")

# Sidebar navigation
tabs = ["Live Prediction", "Workforce Overview", "Monitoring", "Model Info"]

with st.sidebar:
    selected_tab = option_menu(None, tabs, 
        icons=['lightbulb', 'people', 'graph-up', 'info-circle'], default_index=0)
   
# Live Prediction tab
if selected_tab == "Live Prediction":
    st.header("Live Prediction")

    # Collecting inputs from the user
    with st.form(key='prediction_form'):
        st.subheader("Enter Employee Details:")

        st.write("**PERSONAL INFO**")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            if age >= 18 and age <= 30:
                age_group = '18-30'
            elif age >= 31 and age <= 40:
                age_group = '31-40'
            elif age >= 41 and age <= 50:
                age_group = '41-50'
            elif age >= 51 and age <= 60:
                age_group = '51-60'
            else:
                age_group = '61-'
        with col2:
            gender = st.selectbox("Gender", ['Male', 'Female'])
        with col3:
            maritial_status = st.selectbox("Maritial Status", ['Single', 'Married', 'Divorced'])
        col1, col2 = st.columns(2)
        with col1:
            education = st.selectbox("Education", [1, 2, 3, 4, 5])
        with col2:
            education_field = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
        col1, col2 = st.columns(2)
        with col1:
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=3)
        with col2:
            num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2, step=1)
        col1, col2 = st.columns(2)
        with col1:
            distance_from_home = st.number_input("Distance from Home", min_value=0, max_value=100, value=10)
        with col2:
            stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        
        st.write("**ROLE**")
        col1, col2, col3 = st.columns(3)
        with col1:
            emp_id = st.text_input("5-digit Employee ID", value="98765", max_chars=5)
        with col2:
            department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
        with col3:
            job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
        job_level = st.slider("Job Level", min_value=1, max_value=5, value=3, step=1)
        job_involvement = st.slider("Job Involvement", min_value=1, max_value=5, value=3, step=1)
        performance_rating = st.slider("Performance Rating", min_value=1, max_value=5, value=3, step=1)
        over_time = st.selectbox("Frequently Over Time", ['Yes', 'No'])
        col1, col2, col3 = st.columns(3)
        with col1:
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        with col2:
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=30, value=3, step=1)
        with col3:
            years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=30, value=3, step=1)
        col1, col2, col3 = st.columns(3)
        with col1:
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=3, step=1)
        with col2:
            training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, value=3, step=1)
        with col3:
            business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])

        st.write("**SALARY**")
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=7000, step=100)
        col1, col2, col3 = st.columns(3)
        with col1:
            daily_rate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=1102)
        with col2:
            hourly_rate = st.number_input("Hourly Rate", min_value=10, max_value=120, value=94)
        with col3:
            monthly_rate = st.number_input("Monthly Rate", min_value=1000, max_value=30000, value=21410, step=10)
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=12)

        st.write("**SATISFACTORY LEVEL**")
        satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=5, value=3, step=1)
        environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=5, value=3, step=1)
        relationship_satisfaction = st.slider("Relationship Satisfaction", min_value=1, max_value=5, value=3, step=1)
        work_life_balance = st.slider("Work-Life Balance", min_value=1, max_value=5, value=3, step=1)
        
        # Submit button
        submit_button = st.form_submit_button(label="Predict")

    # When form is submitted
    if submit_button:
        if not emp_id:
            st.error("Employee ID is required!")
        elif not (emp_id.isdigit() and len(emp_id) == 5):
            st.error("Please enter a 5-digit Employee ID.")
        else:
            # Prepare the data for the API call (assuming a predefined API endpoint)
            data = {
                    "EmployeeNumber": int(emp_id),
                    "SnapshotDate": datetime.now().strftime("%Y-%m-%d"),
                    "Age": age,
                    "Gender": gender,
                    "MaritalStatus": maritial_status,
                    "Department": department,
                    "EducationField": education_field,
                    "JobLevel": job_level,
                    "JobRole": job_role,
                    "BusinessTravel": business_travel,
                    "DistanceFromHome": distance_from_home,
                    "Education": education,
                    "DailyRate": daily_rate,
                    "HourlyRate": hourly_rate,
                    "MonthlyIncome": monthly_income,
                    "MonthlyRate": monthly_rate,
                    "PercentSalaryHike": percent_salary_hike,
                    "StockOptionLevel": stock_option_level,
                    "OverTime": over_time,
                    "NumCompaniesWorked": num_companies_worked,
                    "TotalWorkingYears": total_working_years,
                    "TrainingTimesLastYear": training_times_last_year,
                    "YearsAtCompany": years_at_company,
                    "YearsInCurrentRole": years_in_current_role,
                    "YearsSinceLastPromotion": years_since_last_promotion,
                    "YearsWithCurrManager": years_with_curr_manager,
                    "EnvironmentSatisfaction": environment_satisfaction,
                    "JobInvolvement": job_involvement,
                    "JobSatisfaction": satisfaction,
                    "PerformanceRating": performance_rating,
                    "RelationshipSatisfaction": relationship_satisfaction,
                    "WorkLifeBalance": work_life_balance,
                    "AgeGroup": age_group
                    }

            # Make API request to perform prediction (replace with your actual API endpoint)
            try:
                api_url = os.getenv("API_URL", "http://api:8000")
                response = requests.post(f"{api_url}/predict", json=data)
            except Exception as e:
                response = requests.post("http://localhost:8000/predict", json=data)
            
            if response.status_code == 200:
                prediction = response.json().get("prediction", "No prediction available.")
                if prediction == 1:
                    prediction = "Employee will LEAVE."
                else:
                    prediction = "Employee will STAY."
                st.subheader("**Prediction:**")
                st.write(f"#### {prediction}")
            else:
                st.error("Error making prediction. Please try again later.")

# Workforce Overview tab
elif selected_tab == "Workforce Overview":
    st.header("Workforce Overview")
    
    # Initialize database connection
    if not DATABASE_URL_PYMSSQL:
        st.error("Database connection not configured. Please set DATABASE_URL_PYMSSQL in your .env file.")
    else:
        try:
            engine = create_engine(DATABASE_URL_PYMSSQL)
            
            # Fetch latest snapshot date
            with engine.connect() as conn:
                max_date_res = conn.execute(text(f"SELECT MAX({SNAPSHOT_DATE_COL}) as max_date FROM {DB_HISTORY_TABLE}"))
                max_date = max_date_res.scalar()
            
            if not max_date:
                st.warning("No data found in the database.")
            else:
                # Fetch batch prediction results
                query = text(f"""
                    SELECT h.*, b.Prediction
                    FROM {DB_HISTORY_TABLE} h
                    LEFT JOIN {DB_BATCH_PREDICTION_TABLE} b
                    ON h.{EMPLOYEE_ID_COL} = b.{EMPLOYEE_ID_COL}
                    AND h.{SNAPSHOT_DATE_COL} = b.{SNAPSHOT_DATE_COL}
                    WHERE h.{SNAPSHOT_DATE_COL} = :snap
                """)
                df = pd.read_sql(query, engine, params={'snap': max_date})
                
                if df.empty:
                    st.warning("No prediction results found for the latest snapshot.")
                else:
                    # Filters
                    st.subheader("Filters")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        department_filter = st.multiselect("Department", df['Department'].unique())
                    with col2:
                        job_role_filter = st.multiselect("Job Role", df['JobRole'].unique())
                    with col3:
                        prediction_filter = st.multiselect("Prediction", ['Stay', 'Leave'])
                        prediction_filter = ["1" if x == 'Leave' else "0" for x in prediction_filter]
                    
                    # Apply filters
                    if department_filter:
                        df = df[df['Department'].isin(department_filter)]
                    if job_role_filter:
                        df = df[df['JobRole'].isin(job_role_filter)]
                    if prediction_filter:
                        df = df[df['Prediction'].isin(prediction_filter)]
                    
                    # Display metrics
                    st.subheader("Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Employees", len(df))
                    with col2:
                        attrition_rate = len(df[df['Prediction'] == "1"]) / len(df)
                        st.metric("Attrition Rate", f"{attrition_rate:.1%}")
                    with col3:
                        avg_age = df['Age'].mean()
                        st.metric("Average Age", f"{avg_age:.1f}" if isinstance(avg_age, (int, float)) else "N/A")
                    with col4:
                        avg_tenure = df['YearsAtCompany'].mean()
                        st.metric("Average Tenure", f"{avg_tenure:.1f}" if isinstance(avg_tenure, (int, float)) else "N/A")
                    
                    # Visualizations
                    st.subheader("Workforce Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Department distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df['Department'].value_counts().plot(kind='bar', ax=ax)
                        plt.title('Employees by Department')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        # Job Role distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df['JobRole'].value_counts().plot(kind='bar', ax=ax)
                        plt.title('Employees by Job Role')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
                    
                    # Data table
                    st.subheader("Employee Data")
                    st.dataframe(df)
        
        except Exception as e:
            st.error(f"Error connecting to database: {str(e)}")
            st.info("Please ensure your database is running and accessible.")

# Monitoring tab
elif selected_tab == "Monitoring":
    st.header("Model Monitoring")

    # Check for drift reports
    st.subheader("Drift Detection")
    try:
        # Load latest drift report
        drift_report_path = os.path.join("reports", "drift_report.json")
        if os.path.exists(drift_report_path):
            with open(drift_report_path, 'r') as f:
                drift_report = json.load(f)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Drift Detected", "Yes" if drift_report['dataset_drift'] else "No")
            with col2:
                drift_share = drift_report['drift_share']
                st.metric("Drift Share", f"{drift_share:.1%}" if isinstance(drift_share, (int, float)) else str(drift_share))
            with col3:
                st.metric("Drifted Features", drift_report['n_drifted_features'])
            
            # Display drifted features
            if drift_report['drifted_features']:
                st.write("Drifted Features:")
                for feature in drift_report['drifted_features']:
                    st.write(f"- {feature}")
        else:
            st.warning("No drift report found. Run drift detection to generate a report.")
    except Exception as e:
        st.error(f"Error loading drift report: {str(e)}")

# Model Info tab
elif selected_tab == "Model Info":
    st.header("Model Info")

    # Fetch model info from the API
    try:
        api_url = os.getenv("API_URL", "http://api:8000")
        response = requests.get(f"{api_url}/model-info")
    except Exception as e:
        response = requests.get("http://localhost:8000/model-info")

    if response.status_code == 200:        
        model_info = response.json()
        model_ver = model_info['latest_registered_version']

        model = f"mlruns/models/AttritionProductionModel/version-{model_ver}/meta.yaml"
        with open(model, 'r') as f:
            model_yaml = yaml.safe_load(f)
            model_source = model_yaml['source'].split('/')
            best_trial_path = f"mlartifacts/{model_source[1]}/{model_source[2]}/{model_source[3]}"

            with open(f"{best_trial_path}/best_optuna_trial_params.json", 'r') as f:
                best_trial_params = json.load(f)
                model_name = best_trial_params['model_type'].replace('_', ' ').title()
        
        st.write(f"Best Performing Model: **{model_name}**")
        st.write(f"Last Update: {datetime.fromtimestamp(model_info['latest_registered_creation_timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("Failed to fetch model info. Please try again later.")
    
    # Display evaluation metrics and visualizations
    run_id = model_info['latest_registered_run_id']
    
    # Load and display model performance
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        try:
            with open(f"{best_trial_path}/evaluation_reports/confusion_matrix_{run_id}.json", 'r') as f:
                confusion_matrix = json.load(f)
                
                # Create a figure for the confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = np.array(confusion_matrix['matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Stay', 'Leave'],
                        yticklabels=['Stay', 'Leave'])
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)
                plt.close()
        except FileNotFoundError:
            st.warning("Confusion matrix not found for this model version.")
    
    with col2:
        try:
            st.image(f"{best_trial_path}/evaluation_reports/roc_curve_{run_id}.png")
        except FileNotFoundError:
            st.warning("ROC curve plot not found.")
    
    # Show Results Interpretation
    st.subheader("Results Interpretation")
    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image(f"{best_trial_path}/explainability_reports/feature_importance_{run_id}.png", caption="Feature Importance")
        except FileNotFoundError:
            st.warning("Feature importance plot not found.")
    
    with col2:
        try:
            st.image(f"{best_trial_path}/explainability_reports/shap_summary_{run_id}.png", caption="SHAP Summary Plot")
        except FileNotFoundError:
            st.warning("SHAP summary plot not found.")
    
    # Load and display fairness report
    try:
        with open(f"{best_trial_path}/evaluation_reports/fairness_report_{run_id}.json", 'r') as f:
            fairness_report = json.load(f)
            st.subheader("Fairness Metrics")
            
            # Gender-based metrics
            if 'Gender' in fairness_report:
                st.write("#### Gender-based Fairness Metrics")
                gender_data = fairness_report['Gender']
                
                # Create metrics comparison for gender groups
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'selection_rate']
                gender_groups = list(gender_data['by_group']['accuracy'].keys())
                
                # Create a figure with subplots for each metric
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for idx, metric in enumerate(metrics):
                    values = [gender_data['by_group'][metric][group] for group in gender_groups]
                    bars = axes[idx].bar(gender_groups, values)
                    axes[idx].set_title(f'{metric.capitalize()} by Gender')
                    axes[idx].set_ylim(0, 1)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                     f'{height:.2f}',
                                     ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                if 'AgeGroup' not in fairness_report:
                    # Display gender-based differences
                    st.write("##### Gender-based Differences")
                    diff_data = gender_data['difference_overall']
                    diff_df = pd.DataFrame.from_dict(diff_data, orient='index', columns=['Difference'])
                    st.dataframe(diff_df)
            
            # Age Group-based metrics
            if 'AgeGroup' in fairness_report:
                st.write("#### Age Group-based Fairness Metrics")
                age_data = fairness_report['AgeGroup']
                
                # Create metrics comparison for age groups
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'selection_rate']
                age_groups = list(age_data['by_group']['accuracy'].keys())
                
                # Create a figure with subplots for each metric
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for idx, metric in enumerate(metrics):
                    values = [age_data['by_group'][metric][group] for group in age_groups]
                    bars = axes[idx].bar(age_groups, values)
                    axes[idx].set_title(f'{metric.capitalize()} by Age Group')
                    axes[idx].set_ylim(0, 1)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                     f'{height:.2f}',
                                     ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                if 'Gender' not in fairness_report:
                    # Display age group-based differences
                    st.write("##### Age Group-based Differences")
                    diff_data = age_data['difference_overall']
                    diff_df = pd.DataFrame.from_dict(diff_data, orient='index', columns=['Difference'])
                    st.dataframe(diff_df)
            
            if ('AgeGroup' in fairness_report) and ('Gender' in fairness_report):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("##### Gender-based Differences")
                    diff_data = gender_data['difference_overall']
                    diff_df = pd.DataFrame.from_dict(diff_data, orient='index', columns=['Difference'])
                    st.dataframe(diff_df)
                with col2:
                    st.write("##### Age Group-based Differences")
                    diff_data = age_data['difference_overall']
                    diff_df = pd.DataFrame.from_dict(diff_data, orient='index', columns=['Difference'])
                    st.dataframe(diff_df)
            
    except FileNotFoundError:
        st.warning("Fairness report not found for this model version.")
    
