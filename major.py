import streamlit as st
import numpy as np
import tensorflow as tf

st.title("📊 Road Accident Prediction Dashboard")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lstm_model.h5")

model = load_model()

num_features = 5  # change based on your dataset

inputs = []
for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, 1, num_features)
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction)

    st.success(f"Predicted Class: {predicted_class}")