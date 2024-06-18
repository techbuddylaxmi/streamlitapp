import streamlit as st
import pandas as pd
import os
import joblib
import base64
from pycaret.regression import setup, compare_models, save_model, pull, load_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Check if the dataset exists, if not create an empty DataFrame
if not os.path.exists('dataset.csv'):
    df = pd.DataFrame()
else:
    df = pd.read_csv('dataset.csv')

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    choice = st.radio("Choose Option", ["Upload", "Modelling", "Download", "Predict"])

# Define session state variables
if 'chosen_target' not in st.session_state:
    st.session_state.chosen_target = None

if 'best_model_names' not in st.session_state:
    st.session_state.best_model_names = []

if 'setup_df' not in st.session_state:
    st.session_state.setup_df = pd.DataFrame()

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_excel(file, engine='openpyxl')
        df.to_csv('dataset.csv', index=False)
        st.dataframe(df)

# Filtered DataFrame for Modelling
if not df.empty:
    df_filtered = df[(df['GCV Quality'] >= 75) & (df['NCV Quality'] >= 75)]
    df_filtered = df_filtered[["NCV", "GCV", "Volume", "Flowing temp", "Pressure(psi)"]]

if choice == "Modelling":
    st.title("Model Training")
    if df.empty or df_filtered.empty:
        st.warning("Please upload a dataset with suitable values for modeling.")
    else:
        st.session_state.chosen_target = st.selectbox('Choose the Target Column', ["NCV", "GCV"])
        if st.button('Run Modelling'):
            with st.spinner("Running Modelling..."):
                setup_df = setup(data=df_filtered, target=st.session_state.chosen_target, session_id=123)
                st.session_state.setup_df = pull()
                st.dataframe(st.session_state.setup_df)
                best_models = compare_models(n_select=5)
                metrics_list = []
                for i, model in enumerate(best_models):
                    model_name = f'{model.__class__.__name__}_{st.session_state.chosen_target}_{i}'
                    st.session_state.best_model_names.append(model_name)
                    save_model(model, model_name)
                    predictions = model.predict(df_filtered.drop(columns=[st.session_state.chosen_target]))
                    rmse = np.sqrt(mean_squared_error(df_filtered[st.session_state.chosen_target], predictions))
                    r2 = r2_score(df_filtered[st.session_state.chosen_target], predictions)
                    mape = np.mean(np.abs((df_filtered[st.session_state.chosen_target] - predictions) / df_filtered[st.session_state.chosen_target])) * 100
                    metrics_list.append({'model_name': model_name, 'rmse': rmse, 'r2': r2, 'mape': mape})

                metrics_df = pd.DataFrame(metrics_list)
                metrics_df.to_csv('model_metrics.csv', index=False)
                compare_df = pull()
                st.dataframe(compare_df)
                st.success("Models trained successfully.")

if choice == "Download":
    st.title("Download Model")
    if st.session_state.chosen_target:
        for model_name in st.session_state.best_model_names:
            model_filename = f'{model_name}.pkl'
            if os.path.exists(model_filename):
                with open(model_filename, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{model_filename}">Download {model_name}</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning(f"Model {model_name} file does not exist.")
    else:
        st.warning("No models to download. Please train models first.")

if choice == "Predict":
    st.title("Make Predictions")
    if st.session_state.chosen_target and not df_filtered.empty:
        features = {column: st.number_input(f"Enter {column}", value=0.0) for column in df_filtered.columns if column != st.session_state.chosen_target}
        if st.button("Predict"):
            predictions_table = pd.DataFrame(columns=['Model', 'Prediction', 'RMSE', 'R2', 'MAPE'])
            if os.path.exists('model_metrics.csv') and os.path.getsize('model_metrics.csv') > 0:
                metrics_df = pd.read_csv('model_metrics.csv')
                for model_name in st.session_state.best_model_names:
                    model_filename = f'{model_name}.pkl'
                    if os.path.exists(model_filename):
                        model = load_model(model_name)
                        model_metrics = metrics_df[metrics_df['model_name'] == model_name].iloc[0]
                        rmse = model_metrics['rmse']
                        r2 = model_metrics['r2']*100
                        mape = model_metrics['mape']
                        input_features = pd.DataFrame([features], columns=df_filtered.drop(columns=[st.session_state.chosen_target]).columns)
                        prediction = model.predict(input_features)
                        new_row = pd.DataFrame({'Model': [model_name], 'Prediction': [prediction[0]], 'RMSE': [rmse], 'R2': [r2], 'MAPE': [mape]})
                        predictions_table = pd.concat([predictions_table, new_row], ignore_index=True)
                    else:
                        st.warning(f"Model {model_filename} file does not exist.")
                st.dataframe(predictions_table)
                feedback = st.radio("Was the prediction accurate?", ["Yes", "No"])
                st.write("Thank you for your feedback!")
            else:
                st.warning("No model metrics found. Please train models first.")
    else:
        st.warning("Please train a model first.")
