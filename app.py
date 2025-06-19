import streamlit as st
import helper
import pickle
import numpy as np

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# App title
st.title('🧠 Duplicate Question Pairs Checker')

# Input fields
q1 = st.text_input('Enter Question 1')
q2 = st.text_input('Enter Question 2')

# Check if both questions are filled
if st.button('Find'):
    if not q1 or not q2:
        st.warning("⚠️ Please enter both questions.")
        st.stop()

    # Generate feature vector
    try:
        query = helper.query_point_creator(q1, q2)
    except Exception as e:
        st.error(f"Failed to create features: {e}")
        st.stop()

    # Debug info (can be removed later)
    st.write("✅ Query shape:", query.shape)
    st.write("🔍 Query preview:", query if query.shape[1] <= 10 else "Feature vector too large to display")

    # Ensure the query is 2D
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Make prediction
    try:
        result = model.predict(query)[0]

        if result:
            st.success('✅ These questions are **Duplicate**.')
        else:
            st.info('❌ These questions are **Not Duplicate**.')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
