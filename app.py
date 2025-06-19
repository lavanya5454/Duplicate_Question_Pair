import streamlit as st
import helper
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title('Duplicate Question Pairs Checker')

# Input fields
q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

# On button click
if st.button('Find'):
    # Create query vector using helper function
    query = helper.query_point_creator(q1, q2)

    # Debugging output
    st.write("Query shape:", query.shape)
    st.write("Query contents:", query)

    # Ensure it's 2D before prediction
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Make prediction
    try:
        result = model.predict(query)[0]

        if result:
            st.success('🔁 These questions are **Duplicate**.')
        else:
            st.info('❌ These questions are **Not Duplicate**.')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
