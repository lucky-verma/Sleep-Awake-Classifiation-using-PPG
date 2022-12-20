import sys

import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from hrvanalysis import get_frequency_domain_features
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
import frame_analysis_ibi
from frame_analysis_ibi import find_features

def interpolate_signal_at_constant_sampling_rate(timestamps=[], signal=[], time_unit='second', fs=4):
    '''
    Input: timestamps, 1D time series, time unit, sampling rate
    Output: interpolated time, interpolated signal
    Description: The function takes as input a signal which is variably sampled and output an interpolated uniformly sampled signal
    '''
    if len(timestamps) == 0:
        return [], []
    if len(signal) == 0:
        return [], []
    signal_ = np.asarray(signal, dtype='float64')
    t_array = np.array(timestamps)
    if time_unit == 'second':
        t = t_array * 1000
    t -= t[0]
    t_interpol = np.arange(t[0], t[-1], 1000. / fs)
    f_interpol_r = sp.interpolate.interp1d(t, signal_, 'cubic')
    signal_interpol = f_interpol_r(t_interpol)
    if time_unit == 'second':
        t_interpol /= 1000
    t_interpol += t_array[0]
    return t_interpol, signal_interpol


def findresp(data, timestamps, fs):
    '''
    Input: 1D interpolated IBI series, timestamps for the interpolated IBI series, sampling rate
    Output: Respiration rate in respiration cycles/minute
    Description: The function takes as input a PPG time series and extract respiration. The algorithm used
    calculates the maximum frequency of the IBI series that is equivalent to the respiration rate
    '''
    data = data - np.mean(data)
    data = butter_bandpass_filter(data, 0.2, 0.3, 4, order=5)
    N = int(fs * timestamps[len(timestamps) - 1])
    yf = fft(data)
    xf = fftfreq(N, 1 / fs)
    max = 0
    index = 0
    for i in range(len(yf)):
        if yf[i] > max and xf[i] > 0:
            max = yf[i]
            index = i
    return int(xf[index] * 60.0)


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    import numpy as np

    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."

    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def extractgreenfeatures(data):
    '''
    Input: Green PPG signal
    Output: Heart rate, Heart Rate Variability (SDNN), High Frequency component of the IBI series, Ratio of HF/LF of the IBI series, respiration rate
    Description: Calculate Heart Rate and Heart Rate Variability through a time series peak detection algorithm. Respiration rate is calculated as the high frequency
    component of the IBI series. 
    '''
    fs = 25
    signal = data
    filtered = hp.filter_signal(signal, [0.3, 3.5], sample_rate=fs,
                                order=3, filtertype='bandpass')

    fast = medfilt(filtered, 3)
    # find the R-peaks
    zero_crossings = np.where(np.diff(np.sign(fast)) > 0)[0]
    # find the inter-beat interval
    iwi = np.diff(zero_crossings / fs)
    iwiseries = []
    for j in range(len(iwi)):
        if (iwi[j] < np.median(iwi) + np.std(iwi)) and (iwi[j] > np.median(iwi) - np.std(iwi)):
            iwiseries.append(iwi[j])
    timestamp, signal = interpolate_signal_at_constant_sampling_rate(timestamps=(zero_crossings / fs)[1:], signal=iwi,
                                                                     time_unit='second', fs=4)
    resp = findresp(signal, timestamp, fs=4)
    hr = int(60.0 / np.average(iwiseries))
    hrv = int(np.std(iwiseries) * 1000.0)
    rrint = np.diff(zero_crossings)
    freqdomainfeatures = get_frequency_domain_features(rrint)
    return hr, hrv, freqdomainfeatures['hf'], freqdomainfeatures['lf_hf_ratio'], resp, fast, zero_crossings


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    Input: band for the filter, sampling rate
    Output: filter coefficients
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Input: Data Time series, band for the filter
    Output: Filtered signal
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def performFFT(data, samplerate):
    '''
    Input: time series data, sampling rate
    Output: FFT of the signal
    '''
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / samplerate)
    return xf, np.abs(yf)


def aligned(a, alignment=16):
    '''
    Input: Signal, memory alignment required
    Output: Aligned signal
    '''
    if (a.ctypes.data % alignment) == 0:
        return a
    extra = alignment / a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) / a.itemsize
    aa = buf[ofs:ofs + a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    return aa


# input - df: a Dataframe, chunkSize: the chunk size
# output - a list of DataFrame
# purpose - splits the DataFrame into smaller chunks
def split_dataframe(df, chunk_size=10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def analyze_ppg(df):
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    window_size = 25 * 30
    hr_array = []
    hrv_array = []
    resp_array = []
    timestamps = []

    dict_ppg = {'ts': df['timestamp'], 'green': df['led_green'], 'red': df['led_red'], 'ir': df['led_ir']}
    df_ppg = pd.DataFrame(dict_ppg).dropna()

    chunks = split_dataframe(df_ppg, window_size)

    for i in range(len(chunks)):
        datadf = chunks[i]
        if len(datadf) == window_size:
            hr, hrv, freqdomainfeatures, freqdomainfeatures, resp, fast, zero_crossings = extractgreenfeatures(
                datadf['green'])
            hr_array.append(hr)
            hrv_array.append(hrv)
            resp_array.append(resp)
            timestamps.append(datadf.iloc[0]['ts'])

    return {'timestamp': timestamps, 'heart_rate': hr_array, 'resp_rate': resp_array, "hrv": hrv_array}


def main(inputfile, outputfile, groundtruth):
    datfile = inputfile
    windowsize = 25 * 60
    num_cycles = 6
    hrarray = []
    hrccarray = []
    hrvarray = []
    resparray = []
    timestamps = []

    df = pd.read_csv(datfile)
    dictppg = {'ts': df['unixTimes'], 'green': df['ledGreen'], 'red': df['ledRed'], 'ir': df['ledIR']}
    dfppg = pd.DataFrame(dictppg).dropna()
    chunks = split_dataframe(dfppg, windowsize)

    for i in range(len(chunks)):
        datadf = chunks[i]
        frame_data = []
        frame_datair = []
        frame_datared = []
        frame_peak_inds = []
        frame_peak_indsir = []
        frame_peak_indsred = []
        frame_valley_inds = []
        frame_valley_indsir = []
        frame_valley_indsred = []
        heart_rate = 0
        heart_rateir = 0
        heart_ratered = 0
        avg_amplitude = 0
        avg_amplitudeir = 0
        avg_amplitudered = 0
        half_avg_amplitude = 0
        half_avg_amplitudeir = 0
        half_avg_amplitudered = 0
        quarter_avg_amplitude = 0
        quarter_avg_amplitudeir = 0
        quarter_avg_amplitudered = 0
        avg_fw_50 = 0
        avg_fw_50ir = 0
        avg_fw_50red = 0
        avg_fw_25 = 0
        avg_fw_25ir = 0
        avg_fw_25red = 0
        acg = 0
        acir = 0
        acr = 0
        dcg = 0
        dcir = 0 
        dcr = 0
        auc_total = 0
        auc_totalir = 0
        auc_totalred = 0
        if len(datadf) == windowsize:
            hr, hrv, freqdomainfeatures, freqdomainfeatures, resp, fast, zero_crossings = extractgreenfeatures(datadf['green'])
            frame_data, frame_peak_inds, frame_valley_inds, heart_rate, avg_amplitude, half_avg_amplitude, quarter_avg_amplitude, avg_fw_50, avg_fw_25, acg, dcg, auc_total, heart_rate_var, valid= find_features(datadf['green'], num_cycles)
            if valid == False:
                print(f"Valid is False")
            frame_datared, frame_peak_indsred, frame_valley_indsred, heart_ratered, avg_amplitudered, half_avg_amplitudered, quarter_avg_amplitudered, avg_fw_50red, avg_fw_25red, acr, dcr, auc_totalred, heart_rate_varred, validred= find_features(datadf['red'], num_cycles)
            frame_datair, frame_peak_indsir, frame_valley_indsir, heart_rateir, avg_amplitudeir, half_avg_amplitudeir, quarter_avg_amplitudeir, avg_fw_50ir, avg_fw_25ir, acir, dcir, auc_totalir, heart_rate_varir, validir= find_features(datadf['ir'], num_cycles)
            if valid == True:
                #print(f"{heart_rateir} and {heart_rate} and {hr} and Sp02: {acg} and {dcg} and {acir} and {dcir}")
                if dcg > 0 and dcir > 0:
                    # print(f"SP02: {((acr/dcr)/(acir/dcir))} and {hrv} and {heart_rate_var*1000} and {heart_rate} and {heart_rateir} and {heart_ratered}")
                    print(f"HRV: {hrv} and {heart_rate_var*1000} and {heart_rate} and {heart_rateir} and {heart_ratered}")

                hrccarray.append(heart_rate)
            hrarray.append(hr)
            hrvarray.append(hrv)
            resparray.append(resp)
            timestamps.append(datadf.iloc[0]['ts'])

    if groundtruth != "0":
        filename = groundtruth
        groundtruth = pd.read_csv(filename)
    outputdict = {'Timestamp': timestamps, 'HR': hrarray, 'RespRate': resparray, "HRV": hrvarray}
    outputdf = pd.DataFrame(outputdict)
    outputdf.to_csv(outputfile)
    plt.plot(hrarray, 'bo--')
    plt.plot(hrccarray, 'ro--')
    if groundtruth != "0":
        plt.plot(groundtruth['heartrate'])
    plt.xlabel('Sample number (windows of 60 seconds)')
    plt.ylabel('Heart Rate')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            f"Correct Usage: python analyzePPG.py <input overnight file> <output file with metrics> <FitBIT data optional>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])