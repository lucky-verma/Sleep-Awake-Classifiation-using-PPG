import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import butter, sosfilt, sosfreqz


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def get_analysis_data(seg_indices, seg_data, fs=25, epoch_length=30):
    data = []

    for i in range(len(seg_indices)):
        tmp = seg_data[i]
        for i in range(fs*epoch_length):
            data.append(tmp)

    return data


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
st.title("Peaks Analysis")

# open a csv file from the local directory and load it into a pandas dataframe using streamlit file uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    fname = uploaded_file.name.split('-')[0]
    mdf = pd.read_csv(uploaded_file)
    df = mdf[mdf.sleep_state != -1].reset_index()

    df = df[[
        'ledGreen',
        'sleep_state'
    ]].dropna()
    # df['sleep_state'] = df['sleep_state'].mask(lambda col: col == -1, 6)

    # Applying the condition
    df["sleep_state"] = np.where(df["sleep_state"] == 0, 0, 1)


    fs = st.sidebar.slider('Sampling Frequency', 1, 100, 25)
    lowcut = st.sidebar.slider('lowcut', 0.1, 2.0, 0.55)
    highcut = st.sidebar.slider('highcut', 1, 10, 3)
    order = st.sidebar.slider('order', 1, 10, 3)
    epoch_length = st.sidebar.slider('Epoch Length', 1, 300, 30)

    df['ledGreen'] = butter_bandpass_filter(df['ledGreen'],
                                            lowcut,
                                            highcut,
                                            fs,
                                            order=order)

    # Select to run analysis on the entire dataset or a subset
    subset = st.sidebar.selectbox('Subset', ('All', 'Subset'), index=0)

    if subset == 'All':

        # streamlit button to run analysis
        if st.sidebar.button('Run Analysis'):
            st.write('## Heartpy Analysis')

            #run the analysis
            wd, m = hp.process(np.array(df['ledGreen']), sample_rate=fs)

            #call plotter
            st.pyplot(hp.plotter(wd, m))

            # Display measures computed
            st.write('## Measures')
            st.write(m)

            st.write('Number of rejected Peaks', len(wd['removed_beats']))
            st.write('Number of accepted  Peaks', len(wd['peaklist']))

        # Overlay the sleep state on the heart rate and breathing rate
        st.write('## Sleep State Overlay')

        # Calculate the breathing rate for each epoch
        wd_seg, m_seg = hp.process_segmentwise(np.array(df['ledGreen']), fs, segment_width=epoch_length, segment_overlap=0.5)

        # Create a dataframe the respiration rate and bpm and heart rate variability
        df_seg = pd.DataFrame()
        df_seg['respiration_rate'] = get_analysis_data(
            wd_seg['segment_indices'], wd_seg['RR_list'], fs, epoch_length)
        df_seg['bpm'] = get_analysis_data(wd_seg['segment_indices'], m_seg['bpm'], fs, epoch_length)
        df_seg['hrv'] = get_analysis_data(wd_seg['segment_indices'], m_seg['rmssd'], fs, epoch_length)
        df_seg['hr'] = get_analysis_data(wd_seg['segment_indices'], wd_seg['hr'], fs, epoch_length)
        df_seg['breathingrate'] = get_analysis_data(wd_seg['segment_indices'], m_seg['breathingrate'] * 1000, fs, epoch_length)

        # plotting using go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['ledGreen'], name='PPG (led Green))'))
        fig.add_trace(go.Scatter(x=df.index, y=df['sleep_state']*100, name='Sleep State'))
        fig.add_trace(go.Scatter(x=df.index, y=df_seg['bpm'], name='BPM'))
        fig.add_trace(go.Scatter(x=df.index, y=df_seg['hrv'], name='HRV'))
        fig.add_trace(go.Scatter(x=df.index, y=df_seg['breathingrate'], name='Breathing Rate'))
        fig.add_trace(go.Scatter(x=df.index, y=df_seg['respiration_rate']*500, name='Respiration Rate'))
        fig.update_layout(
            title="Sleep State Overlay",
        )
        st.plotly_chart(fig)

    else:
        # Select the start and end of the subset
        start = st.slider('Start', 0, len(df), 10_000)

        # two columns
        col1, col2 = st.columns(2)

        with col1:
            st.write('## Heartpy Analysis')
            #run the analysis
            wd, m = hp.process(np.array(df['ledGreen'][start:start+(fs*epoch_length)]), sample_rate=fs)

            #call plotter
            st.pyplot(hp.plotter(wd, m))

            # plot_breathing
            st.pyplot(hp.plot_breathing(wd, m))

            st.write('Number of rejected Peaks', len(wd['removed_beats']))
            st.write('Number of accepted  Peaks', len(wd['peaklist']))

        with col2:
            st.write('## Sleep State')

            # plot sleep state
            fig = px.line(df[start:start+(fs*epoch_length)], x=df.index[start:start+(fs*epoch_length)], y='sleep_state')
            fig.update_layout(
                title="Sleep State",
            )
            st.plotly_chart(fig)

            # Display measures computed
            st.write('## Measures')
            st.write(m)
