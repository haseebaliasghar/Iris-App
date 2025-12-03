import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier

# Set page configuration
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# Title and description
st.title("🌸 Iris Species Predictor")
st.write("This app uses a Machine Learning model to predict the species of an Iris flower based on its measurements.")

# Function to load the model (with fallback)
@st.cache_resource
def load_model():
    model = None
    # 1. Try to load the user's uploaded pickle file
    if os.path.exists('decision_tree_iris.pkl'):
        try:
            with open('decision_tree_iris.pkl', 'rb') as file:
                model = pickle.load(file)
            print("Success: Loaded local pickle file.")
        except Exception as e:
            st.warning(f"Note: Could not load 'decision_tree_iris.pkl' due to version mismatch ({e}). Retraining a fresh model automatically...")
            model = None
    
    # 2. If pickle failed or missing, Retrain on the fly
    if model is None:
        try:
            from sklearn.datasets import load_iris
            iris = load_iris()
            X = iris.data
            y = iris.target
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X, y)
            print("Success: Retrained model in-app.")
        except Exception as e:
            st.error(f"Critical Error: Could not train model. {e}")
            return None

    return model

model = load_model()

# Sidebar for inputs
st.sidebar.header("Input Features")
st.sidebar.write("Adjust the sliders below:")

def user_input_features():
    # Default values based on average Iris data
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 4.3)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.3)
    
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    return data

if model is not None:
    input_data = user_input_features()

    # Create a "Predict" button
    if st.button('Predict Species'):
        prediction = model.predict(input_data)
        
        # Map prediction to name if it returns a number (0,1,2)
        # The retrained model returns numbers, the pickle might return strings.
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        
        if isinstance(prediction[0], (int, np.integer)):
            species = species_map.get(prediction[0], "Unknown")
        else:
            species = prediction[0]
            # Clean up string if needed (remove 'Iris-')
            species = species.replace('Iris-', '')

        st.subheader("Prediction Result:")
        
        # Dynamic formatting
        if species == 'setosa':
            st.success(f"The flower is: **{species}** 🌿")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa")
        elif species == 'versicolor':
            st.success(f"The flower is: **{species}** 💜")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg", caption="Iris Versicolor")
        elif species == 'virginica':
            st.success(f"The flower is: **{species}** 💙")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/320px-Iris_virginica.jpg", caption="Iris Virginica")
        else:
            st.success(f"The flower is: **{species}**")

    # Show the input data
    with st.expander("See input data"):
        st.write(f"Sepal Length: {input_data[0][0]}")
        st.write(f"Sepal Width: {input_data[0][1]}")
        st.write(f"Petal Length: {input_data[0][2]}")
        st.write(f"Petal Width: {input_data[0][3]}")
