import os
import zipfile
import pickle
import streamlit as st

# Path variables
zip_path = "model.pkl.zip"
model_path = "model.pkl"

# Unzip model if not already unzipped
if not os.path.exists(model_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

# Load the model
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))
else:
    st.error("Model file not found after unzipping.")
    st.stop()


# Load other required assets like cv.pkl
if os.path.exists('cv.pkl'):
    helper.cv = pickle.load(open('cv.pkl', 'rb'))
else:
    st.error("❌ cv.pkl not found. Please upload the vectorizer file.")
    st.stop()

model = pickle.load(open('model.pkl','rb'))

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')

