import streamlit as st
import helper
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1, q2)
    st.write("Query shape/type:", type(query), query)

    # Optional: reshape if it's a flat list
    if isinstance(query, list):
        query = np.array(query).reshape(1, -1)

    try:
        result = model.predict(query)[0]

        if result:
            st.header('Duplicate')
        else:
            st.header('Not Duplicate')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
