from __future__ import division
import streamlit as st
from authentication import make_sidebar
import os

make_sidebar()

st.title("Instructions")
st.sidebar.success('Select the ticket data or sales data')

# Read the HTML file content from the parent directory
html_file_path = os.path.join(os.path.dirname(__file__), '.', 'instructions.html')
with open(html_file_path, 'r') as file:
    html_content = file.read()

# Display the HTML content
st.markdown(html_content, unsafe_allow_html=True)
