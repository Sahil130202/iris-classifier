import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load models
models = {
    "Logistic Regression": joblib.load("iris_model_logistic_regression.pkl"),
    "Random Forest": joblib.load("iris_model_random_forest.pkl"),
    "Linear Regression (Simulated)": joblib.load("iris_model_linear_regression.pkl"),
}

# Page config
st.set_page_config(page_title=" Iris Classifier App", layout="wide")

# Title & Intro
st.markdown("##  Iris Flower Classifier")
st.markdown("Use different ML models to predict iris flower species based on petal and sepal measurements.")

# Sidebar - Input Features
st.sidebar.header(" Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Model Selection
model_choice = st.selectbox(" Choose Model", list(models.keys()))
model = models[model_choice]

# Prediction
prediction = model.predict(features)
if model_choice == "Linear Regression (Simulated)":
    prediction = np.clip(np.round(prediction), 0, 2).astype(int)

species_map = {0: "Setosa ", 1: "Versicolor ", 2: "Virginica "}
predicted_species = species_map[prediction[0]]

# Output
st.success(f" Predicted Species: **{predicted_species}** using *{model_choice}*")

# Metric Plots
st.markdown("---")
st.markdown("###  Model Evaluation Metrics")

cols = st.columns(2)
with cols[0]:
    st.image("metrics_plot_accuracy.png", caption="Model Accuracy", use_column_width=True)

# Footer
st.markdown("---")
st.markdown(" *Built with Streamlit by Sahil Nayak*")
