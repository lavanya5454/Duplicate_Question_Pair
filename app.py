import streamlit as st
import helper
import pickle
import os
if os.path.exists('model.pkl'):
    model = pickle.load(open('model.pkl', 'rb'))
else:
    st.error("❌ model.pkl not found. Please upload the model file.")
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

