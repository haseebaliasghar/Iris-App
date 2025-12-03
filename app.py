import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered"
)

# Title and description
st.title("🌸 Iris Species Predictor")
st.write("This app uses a Machine Learning model to predict the species of an Iris flower based on its measurements.")

# Function to load the model
@st.cache_resource
def load_model():
    # Check if file exists
    if not os.path.exists('decision_tree_iris.pkl'):
        st.error("Model file 'decision_tree_iris.pkl' not found. Please make sure it is in the same directory as this app.py file.")
        return None
    
    with open('decision_tree_iris.pkl', 'rb') as file:
        model = pickle.load(file)
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

# Only proceed if model loaded successfully
if model is not None:
    input_data = user_input_features()

    # Create a "Predict" button
    if st.button('Predict Species'):
        prediction = model.predict(input_data)
        
        # Display results
        st.subheader("Prediction Result:")
        species = prediction[0]
        
        # dynamic formatting based on result
        if species == 'Iris-setosa' or species == 'setosa':
            st.success(f"The flower is: **{species}** 🌿")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/320px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa")
        elif species == 'Iris-versicolor' or species == 'versicolor':
            st.success(f"The flower is: **{species}** 💜")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg", caption="Iris Versicolor")
        elif species == 'Iris-virginica' or species == 'virginica':
            st.success(f"The flower is: **{species}** 💙")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/320px-Iris_virginica.jpg", caption="Iris Virginica")
        else:
            st.success(f"The flower is: **{species}**")

    # Show the input data for reference
    with st.expander("See input data"):
        st.write(f"Sepal Length: {input_data[0][0]}")
        st.write(f"Sepal Width: {input_data[0][1]}")
        st.write(f"Petal Length: {input_data[0][2]}")
        st.write(f"Petal Width: {input_data[0][3]}")

else:
    st.warning("Please upload the 'decision_tree_iris.pkl' file to your GitHub repository.")