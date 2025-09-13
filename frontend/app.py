import streamlit as st

st.title("My First Streamlit App")
st.header("Welcome!")
st.write("This is a simple Streamlit application.")

# Add a slider widget
x = st.slider('Select a value', 0, 100)
st.write('The selected value is', x)

# Add a button
if st.button('Click me'):
    st.write('Button was clicked!')