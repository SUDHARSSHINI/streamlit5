import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from responsibleai import RAIInsights, FeatureMetadata
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Define functions for data processing and model training
def create_synthetic_data():
    data = {
        'Age': np.random.randint(20, 80, size=50),
        'Sex': np.random.randint(0, 2, size=50),
        'Cholesterol': np.random.randint(150, 300, size=50),
        'Blood Pressure': np.random.randint(80, 180, size=50),
        'Heart Disease': np.random.randint(0, 2, size=50)
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    X = df.drop(columns=['Heart Disease'])
    y = df['Heart Disease'].astype(int)
    categorical_features = ['Sex']
    X[categorical_features] = X[categorical_features].astype(str)
    if X.isnull().any().any() or y.isnull().any():
        X = X.dropna()
        y = y[X.index]
    return X, y

def create_pipeline():
    categorical_features = ['Sex']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def fairness_metrics(X_test, y_test, pipeline):
    y_pred = pipeline.predict(X_test)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    predicted_df = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(data={'Heart Disease': y_pred}).reset_index(drop=True)], axis=1)
    
    test_data_aif360 = BinaryLabelDataset(
        df=test_df,
        label_names=['Heart Disease'],
        protected_attribute_names=['Sex']
    )
    predicted_data_aif360 = BinaryLabelDataset(
        df=predicted_df,
        label_names=['Heart Disease'],
        protected_attribute_names=['Sex']
    )
    metric = ClassificationMetric(test_data_aif360, predicted_data_aif360,
                                   privileged_groups=[{'Sex': 1}], unprivileged_groups=[{'Sex': 0}])
    
    return {
        'Disparate Impact': metric.disparate_impact(),
        'Statistical Parity Difference': metric.statistical_parity_difference(),
        'Equal Opportunity Difference': metric.equal_opportunity_difference(),
        'Average Odds Difference': metric.average_odds_difference()
    }

def responsible_ai_insights(X_train, y_train, X_test, y_test):
    categorical_features = ['Sex']
    feature_metadata = FeatureMetadata(categorical_features=categorical_features)
    
    rai_insights = RAIInsights(
        model=pipeline,
        train=pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1),
        test=pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
        target_column='Heart Disease',
        task_type='classification',
        feature_metadata=feature_metadata
    )
    rai_insights.explainer.add()
    rai_insights.error_analysis.add()
    rai_insights.counterfactual.add(total_CFs=10, desired_class='opposite')
    treatment_features = ['Cholesterol', 'Blood Pressure']
    rai_insights.causal.add(treatment_features=treatment_features)
    rai_insights.compute()
    
    return {
        'Explainer Insights': rai_insights.explainer.get(),
        'Error Analysis Insights': rai_insights.error_analysis.get(),
        'Counterfactual Insights': rai_insights.counterfactual.get(),
        'Causal Insights': rai_insights.causal.get()
    }

# Streamlit app
st.title("Heart Disease Prediction and Fairness Analysis")

# Create and preprocess data
df = create_synthetic_data()
X, y = preprocess_data(df)
pipeline = create_pipeline()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Display metrics and insights
st.header("Fairness Metrics")
metrics = fairness_metrics(X_test, y_test, pipeline)
for metric, value in metrics.items():
    st.write(f"{metric}: {value}")

st.header("Responsible AI Insights")
insights = responsible_ai_insights(X_train, y_train, X_test, y_test)
for insight_type, insight in insights.items():
    st.write(f"{insight_type}:")
    st.write(insight)

# Predict function
def predict_heart_disease(age, sex, cholesterol, blood_pressure):
    new_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure]
    })
    new_data['Sex'] = new_data['Sex'].astype(str)
    predictions = pipeline.predict(new_data)
    return predictions

# User input
st.header("Predict Heart Disease")
age = st.number_input("Enter age:", min_value=0)
sex = st.selectbox("Select sex:", options=[0, 1])
cholesterol = st.number_input("Enter cholesterol level:", min_value=0)
blood_pressure = st.number_input("Enter blood pressure:", min_value=0)

if st.button("Predict"):
    result = predict_heart_disease(age, sex, cholesterol, blood_pressure)
    st.write(f"Prediction: {result[0]}")
