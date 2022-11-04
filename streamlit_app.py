import streamlit as st
from numpy import load
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

from matplotlib import pyplot as plt
from sklearn import metrics

# Set the page title
st.set_page_config(page_title='PPG2ABP',
                   page_icon=':heartpulse:',
                   layout='wide',
                   initial_sidebar_state='auto')

hide_streamlit_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# App title
st.title("Sleep Classification using PPG")


# Set the page description
st.markdown('''sleep stage binary classification tiny model.''')

# Load data
@st.cache
def load_data(path):
    return glob.glob(os.path.join(path, "*.npz"))

data = None
# take path string input
try:
    data = load_data('./out_resteaze/predict')
except:
    path = st.text_input("Enter path to data",)
    st.write("Please enter a valid path")

if data:
    # Set the slider to select the subject to be evaluated
    subject = st.slider('Select the subject to be evaluated', min_value=0, max_value=len(data), value=0, step=1)
    with st.spinner('Fetching and processing values.....'):
        # Load the data
        data = load(data[subject])
        preds = pd.DataFrame(data=zip(data['y_true'], data['y_pred']),
                             columns=['True', 'Preds'])

        # set two columns to display the plots
        col1, col2 = st.columns([3, 1])

        with col1:
            # Plot using plotly go
            st.subheader("Hypnogram")
            fig = go.Figure()
            fig.add_trace(go.Scatter(name="True", x=list(range(len(preds))), y=preds['True'], marker=dict(
                                                                                                        color='Black',
                                                                                                        size=20,
                                                                                                        line=dict(
                                                                                                            color='MediumPurple',
                                                                                                            width=8
                                                                                                        )
                                                                                                    )))
            fig.add_trace(go.Scatter(name="Preds", x=list(range(len(preds))), y=preds['Preds'], opacity=0.99))
            fig.update_layout(yaxis_title='Sleep Stage', xaxis_title='Epochs', autosize=True, width=1300, height=500)
            st.plotly_chart(fig)

        with col2:
            st.subheader("Confusion Matrix")

            # convert the confusion matrix to a matrix
            conf_mat = metrics.confusion_matrix(preds['True'], preds['Preds'])
            conf_mat = pd.DataFrame(conf_mat, columns=['Wake', 'Sleep'], index=['Wake', 'Sleep'])

            # plot the confusion matrix
            fig1 = px.imshow(conf_mat, labels=dict(x="Predicted", y="Actual", color="Count"), text_auto=True, aspect="auto")
            fig1.update_layout(autosize=True, width=450, height=450)
            st.plotly_chart(fig1)