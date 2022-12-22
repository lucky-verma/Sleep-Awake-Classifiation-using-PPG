# @author

from itertools import cycle
from logging.config import dictConfig
import pyampd_mod
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from scipy.interpolate import interp1d
import bisect
from scipy.signal import butter, filtfilt

# filtering functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# feature analysis functions
def find_frame(input_data, num_cycles):
    """
    Uses fast_ampd (ampd with a maximum window of 15 points) to find peaks in a sample. Finds 8 frames with num_cycles consecutive cardiac cycles 
    with the lowest IBI stdev, then finds frame with lowest variance on the y-axis (data).
    """

    # apply ampd algorithm to find all peaks and valleys
    peaks, valleys = pyampd_mod.ampd(input_data, scale=15)

    # find 8 frames with lowest IBI stdev
    intervals = np.diff(peaks)
    df = pd.DataFrame(intervals, columns = ['value'])
    stdevs = df.rolling(num_cycles).std()
    sorted_stdevs = stdevs.sort_values(by=['value'])
    sorted_stdevs = sorted_stdevs.head(8)
    frames = []
    for stdev in sorted_stdevs.iterrows():
        stdev_ind = stdev[0]
        temp_frame = []
        for i in range(num_cycles):
            temp_frame.insert(0, int(peaks[stdev_ind - i]))
        frames.append(temp_frame)
    min_stdev = float('inf')

    # find frame with lowest y-stdev
    frame_peak_inds = None
    for frame in frames:
        peaks_data = []
        for index in frame:
            peaks_data.append(input_data[index])
        y_intervals = np.diff(peaks_data)
        stdev = y_intervals.std()
        if stdev < min_stdev:
            min_stdev = stdev
            frame_peak_inds = frame

    # find valley indices for cycles containing frame peaks using bisect
    first_val_ind = bisect.bisect(valleys, frame_peak_inds[0]) - 1
    frame_valley_inds = valleys[first_val_ind:first_val_ind + num_cycles + 1]

    return frame_peak_inds, frame_valley_inds

def find_features(input_data, num_cycles):
    """
    Calculates features of best frame containing NUM_CYCLES cardiac cycles.
    Returns frame's peaks, valleys, heart rate, average amplitude, 50% average amplitude, 25% average amplitude,
    average wavelength width at 50% amplitude, average wavelength width at 25% amplitude, 
    AC average, DC average, and total area under the cycles. 
    """

    valid = True
    input_data_tmp = []
    frame_data = []
    frame_peak_inds = []
    frame_valley_inds = []
    heart_rate = 0
    heart_rate_var = 0
    avg_amplitude = 0
    half_avg_amplitude = 0
    quarter_avg_amplitude = 0 
    avg_fw_50 = 0
    avg_fw_25 = 0
    ac = 0
    dc = 0
    auc_total = 0
    
    try:
        # find peaks and valleys of frame in IR, to find with green, use argmin 
        filt_input_data = input_data.copy() # deep copy of input data
        input_data = filt_input_data
        filt_input_data = butter_bandpass_filter(filt_input_data, 0.75, 10, 25) # apply filter on copy
        frame_peak_inds, frame_valley_inds = find_frame(filt_input_data, num_cycles)
        for index, value in input_data.items():
            input_data_tmp.append(value)
        # calculate features
        try:
            heart_rate = find_heart_rate(frame_peak_inds, num_cycles, 25)
        except:
            print("Exception in finding heart rate.\n")
            heart_rate = 0
        #print(f"Heart Rate: {heart_rate}")
        try:
            heart_rate_var =  find_heart_rate_var(frame_peak_inds, num_cycles, 25)
        except:
            print("Exception in finding heart rate.\n")
            heart_rate_var = 0
        #print(f"Heart Rate: {heart_rate} and Heart Rate Variability: {heart_rate_var}")
        avg_amplitude = 0
        avg_fw_50 = 0
        avg_fw_25 = 0
        auc_total = 0
        ac = 0
        dc = 0
        frame_data = []
        for count in range(num_cycles):
            # get index and value for start, peak, and end of each cycle
            start_ind = frame_valley_inds[count]
            end_ind = frame_valley_inds[count + 1]
            start = input_data_tmp[start_ind]
            end = input_data_tmp[end_ind]
            filt_start = filt_input_data[start_ind]
            filt_end = filt_input_data[end_ind]

            # calculate AC and DC averages
            try:
                ac += find_ac(input_data_tmp, start_ind, end_ind, start, end)
            except:
                print("Exception in finding AC average.\n")
                ac += 0
            try: 
                dc += find_dc(start, end)
            except: 
                print("Exception in finding DC average.\n")
                dc += 0

            # get data in cycle with drift removed
            cycle = get_cycle(filt_input_data, start_ind, end_ind, filt_start, filt_end)
            frame_data.extend(cycle)

            # calculate amplitude (and variations) of the cycle
            try:
                amplitude = max(cycle)
            except:
                print("Exception in finding peak in frame.\n")
                amplitude = 0
            avg_amplitude += amplitude
            if amplitude != 0:
                half_amplitude = amplitude / 2
                quarter_amplitude = amplitude / 4
            else:
                half_amplitude = 0
                quarter_amplitude = 0

            # calculate wave widths at various amplitudes
            try: 
                fw_50 = find_fw(cycle, half_amplitude)
            except:
                print("Exception in finding 50% fw.\n")
                fw_50 = 0
            try:
                fw_25 = find_fw(cycle, quarter_amplitude)
            except:
                print("Exception in finding 25% fw.\n")
                fw_25 = 0
            if fw_50 != 0:
                avg_fw_50 += fw_50
            if fw_25 != 0:
                avg_fw_25 += fw_25

            # calculate area under curve (indices being incrementing values as it is a time series)
            indices = list(range(start_ind, end_ind + 1))
            try:
                area = auc(indices, cycle)
            except:
                print("Exception in finding auc.\n")
                area = 0
            auc_total += area
        if avg_amplitude != 0:
            avg_amplitude = avg_amplitude / num_cycles
            half_avg_amplitude = avg_amplitude / 2
            quarter_avg_amplitude = avg_amplitude / 4

        if ac != 0:
            ac = ac / num_cycles
        if dc != 0:
            dc = dc / num_cycles
        if avg_fw_50 != 0:
            avg_fw_50 = avg_fw_50 / num_cycles
        if avg_fw_25 != 0:
            avg_fw_25 = avg_fw_25 / num_cycles

        # if heart rate is not valid (within 40 <= heart_rate <= 200), set it to 0
        if heart_rate < 40 or heart_rate > 200:
            heart_rate = 0
    except:
        valid = False

    return (frame_data, frame_peak_inds, frame_valley_inds, heart_rate, 
            avg_amplitude, half_avg_amplitude, quarter_avg_amplitude, 
            avg_fw_50, avg_fw_25, ac, dc, auc_total, heart_rate_var, valid)

def find_ac(input_data, start_ind, end_ind, start, end):
    """
    AC of a cardiac cycle is the amplitude difference between its peak and the average of its 2 valleys.
    """
    ac_avg_vall = (start + end) / 2
    ac_pk = max(input_data[start_ind:end_ind + 1])
    ret_ac = ac_pk - ac_avg_vall
    return ret_ac

def find_dc(start, end):
    """
    DC for a cardiac cycle is the average of the amplitudes of its 2 valleys.
    """
    dc_avg_vall = (start + end) / 2
    return dc_avg_vall

def get_cycle(input_data, start_ind, end_ind, start, end):
    """
    Given the start and end indices of a cycle, splices that data from overall waveform and removes drift. 
    Returns cycle data and corresponding indices.
    """
    cycle = input_data[start_ind:end_ind + 1]
    cycle = cycle.copy() # make a deep copy, to ensure we are not mutating input_data

    m = (end - start) / (end_ind - start_ind)
    b = start - (m * start_ind)
    for index in range(end_ind - start_ind + 1):
        adjustment = (m * (start_ind + index)) + b
        cycle[index] -= adjustment

    return cycle


def find_heart_rate_var(frame_peak_inds, numcyles, sample_rate):
    diff = []
    for i in range(len(frame_peak_inds)-1):
        diff.append((frame_peak_inds[i+1]- frame_peak_inds[i])/sample_rate)
    return np.std(diff)

def find_heart_rate(frame_peak_inds, num_cycles, sample_rate):
    """
    Finds average heart rate by dividing average interbeat interval of peaks by the number of cycles in the frame.
    """
    dist = frame_peak_inds[-1] - frame_peak_inds[0]
    avg_ibi = dist/(num_cycles-1)
    heart_rate = (60 * sample_rate)/avg_ibi
    return heart_rate

def find_fw(cycle, amplitude):
    """
    Finds width of wavelength in cycle at given amplitude. Since we are working with a time series, 
    estimate the index at amplitude using interpolation. 
    """
    cycle = cycle.tolist()
    peak_ind = cycle.index(max(cycle))
    before_peak_cycle = cycle[:peak_ind + 1]
    after_peak_cycle = cycle[peak_ind:]
    reverse_after_peak_cycle = list(reversed(after_peak_cycle))
    crossing_one = interpolate_crossing(before_peak_cycle, amplitude)
    crossing_two = interpolate_crossing(reverse_after_peak_cycle, amplitude)
    crossing_two = len(cycle) - crossing_two - 1
    fw = crossing_two - crossing_one
    return fw

def interpolate_crossing(data, amplitude):
    left_ind = bisect.bisect(data, amplitude) - 1
    right_ind = left_ind + 1
    left = data[left_ind]
    right = data[right_ind]
    interp = interp1d([left, right], [left_ind, right_ind])
    crossing = interp(amplitude)
    return crossing