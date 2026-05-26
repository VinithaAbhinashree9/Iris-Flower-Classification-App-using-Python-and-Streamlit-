import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#load dataset
iris = load_iris()
X = iris.data
y = iris.target

#Train the model

model =LogisticRegression(max_iter=200)
model.fit(X,y)

#create user interface
st.title("Iris Flower classification")
st.write("Enter the flower measurements:")

#ADD input widgets
#text imputs
sepal_length = st.text_input("Sepal Length(cm)","5.1")
sepal_width = st.text_input("Sepal Width (cm)", "3.5")
# Selection Widgets (Dropdown / Slider)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.selectbox("Petal Width (cm)", [0.1, 0.2, 0.5, 1.0, 1.5, 2.0])
# Prepare Input Data
input_data = pd.DataFrame({
 "sepal length (cm)": [float(sepal_length)],
 "sepal width (cm)": [float(sepal_width)],
 "petal length (cm)": [petal_length],
 "petal width (cm)": [petal_width]
})
# Make Prediction
prediction = model.predict(input_data)
species = iris.target_names[prediction[0]]
# Display Output
st.subheader("Prediction Result")
st.success(f"The predicted flower species is: {species}")