import pyedflib
import os
import shutil
import glob
import datetime
import time
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from numpy import mean, sqrt, square, arange
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

# streamlit page config
st.set_page_config(
    page_title="Heart Rate Variability",
    page_icon=":heartbeat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# hide streamlit warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# hide streamlit logo
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# streamlit title
st.title("Heart Rate Variability")

# streamlit sidebar
