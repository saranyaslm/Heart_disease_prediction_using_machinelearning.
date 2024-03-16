import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st



heart_data = pd.read_csv("heart_disease_data.csv")
heart_data.head()

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()

model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


st.title("heart disease prediction model")
input_text = st.text_input("provide comma separated values")
separated_input = input_text.split(",")
img = Image.open("heart_image.jpg")
st.image(img,width=200)

try:
    np_df = np.asarray(separated_input,dtype=float)
    reshaped_df = np_df.reshape(1,-1)
    prediction = model.predict(reshaped_df)
    if prediction[0] == 0:
        st.write("This person doesn't have heart disease")
    else:
        st.write("This person have heart disease")
except ValueError:
    st.write("Invalid values. Please try again.")


st.subheader("About data")
st.write(heart_data)
st.subheader("Model performance on Training Data")
st.write(training_data_accuracy)
st.subheader("Model performance on Test Data")
st.write(test_data_accuracy)
