# @author Su-Ann Ho

from collections import deque 
from pyampd.ampd import find_peaks
import pandas as pd
import numpy as np

class CardiacCycle:
    def __init__(self, peak, period, start_index, end_index):
        self.peak = peak
        self.period = period
        self.start_index = start_index
        self.end_index = end_index

def point_on_line(x1, y1, x2, y2, x3):
    """Computes y-value of a point (x3, y3) on a line between (x1, y1) and (x2, y2) when given its x-value"""

    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    y3 = (m * x3) + b
    return y3

# DESCRIPTION OF ARGUMENTS:
# input_data: input data array
# input_length: number of data points in input data array
# num_cycles: number of consecutive cardiac cycles the function should look for and return
# noise_threshold: threshold value differentiating noise and a peak -> PICK A VALUE BETWEEN PEAK FOR NOISE AND PEAK FOR AN ACTUAL CYCLE
# max_peak: maximum acceptable peak value for a cycle to be in the frame
# target_period: length that tolerance for cycles' periods is centered around
# period_tolerance: maximum tolerance for variation from the target period for a valid cardiac cycle
def frame_analysis_peaks(input_data, input_length, num_cycles, noise_threshold, max_peak, target_period, period_tolerance):
    """
    #Returns start index, end index, and array of data points of num_cycles best consecutive cardiac cycles in input_data
    #based on a maximum peak value (max_peak) and maximum period tolerance (target_period +- period_tolerance), finding the cycles
    #with highest average peaks
    """

    # return variables
    start_frame = 0
    end_frame = 0
    avg_peak_frame = 0.0
    avg_period_frame = 200.0
    frame = deque([])

    #loop variables
    counter = 0
    prev_data = 0.0
    curr_index = 0
    curr_data = 0.0
    next_data = 0.0
    peak = 0.0
    peak_index = 0
    period = 0.0
    temp_start_frame = 0
    temp_start_val = 0.0
    cont = 0
    output_data = []

    # iterate through all data points
    for curr_index in range(input_length - 1):
        curr_data = input_data[curr_index]
        next_data = input_data[curr_index + 1]

        # identify start and end of cycles (at minima)
        period = curr_index - temp_start_frame
        if ((prev_data > curr_data) and (next_data > curr_data) and (period > 3)):
            adjustment = point_on_line(temp_start_frame, temp_start_val, curr_index, curr_data, peak_index)
            peak_adj = peak - adjustment
            temp_end_frame = curr_index
            if (peak_adj < noise_threshold):
                prev_data = curr_data
                cont = 1
            else:
                cont = 0
                if ((period < target_period + period_tolerance) and (period > target_period - period_tolerance) and (peak_adj < max_peak)):
                        cycle = CardiacCycle(peak_adj, period, temp_start_frame, temp_end_frame)
                        frame.append(cycle)
                        counter += 1
                        if (counter == num_cycles):
                            temp_avg_peak_frame = 0.0
                            temp_avg_period_frame = 0.0
                            for cycle in frame:
                                temp_avg_peak_frame += cycle.peak
                                temp_avg_period_frame += cycle.period
                            temp_avg_peak_frame /= num_cycles
                            temp_avg_period_frame /= num_cycles
                            if (temp_avg_peak_frame > avg_peak_frame):
                                avg_period_frame = temp_avg_period_frame
                                avg_peak_frame = temp_avg_peak_frame
                                start_frame = frame[0].start_index
                                end_frame = temp_end_frame
                            counter -= 1
                            temp_avg_peak_frame = 0.0
                            temp_avg_period_frame = 0.0
                            frame.popleft()
                else:
                    counter = 0
                    temp_avg_peak_frame = 0.0
                    temp_avg_period_frame = 0.0
                    frame.clear()
                temp_start_frame = curr_index
                temp_start_val = curr_data

        # identify peak (at maxima)
        if ((prev_data < curr_data) and (next_data < curr_data)):
            if (cont == 1 and curr_data < peak):
                continue
            peak = curr_data
            peak_index = curr_index
        prev_data = curr_data

    if (end_frame == 0.0):
        print("No {} valid consecutive cardiac cycles found in sample. \n".format(num_cycles))

    else:
        output_data = input_data[start_frame:end_frame + 1]

        print("Values in stream: \n")
        for element in output_data:
            print(element)

        print("The best {} consecutive cardiac cycles begin at index {} and end at index {}, \nwith an average peak of {} and period of {} \n".format( 
            num_cycles, start_frame, end_frame, avg_peak_frame, avg_period_frame))
    
    return start_frame, end_frame, output_data

    
def detect_peaks_valleys(input_data, input_length, num_cycles, noise_threshold, max_peak, target_period, period_tolerance, pk):
    """
    #Returns array of peak indices and array of valley indices in input_data
    #based on a maximum peak value (max_peak) and maximum period tolerance (target_period +- period_tolerance), finding the cycles
    #with highest average peaks
    """

    # return variables
    start_frame = 0
    end_frame = 0
    avg_peak_frame = 0.0
    avg_period_frame = 200.0
    peaks = []
    valleys = []
    frame = deque([])

    #loop variables
    counter = 0
    prev_data = 0.0
    curr_index = 0
    curr_data = 0.0
    next_data = 0.0
    peak = 0.0
    peak_index = 0
    period = 0.0
    temp_start_frame = 0
    temp_start_val = 0.0
    cont = 0

    # iterate through all data points
    for curr_index in range(input_length - 1):
        curr_data = input_data[curr_index]
        next_data = input_data[curr_index + 1]

        # identify start and end of cycles (at minima)
        period = curr_index - temp_start_frame
        if ((prev_data > curr_data) and (next_data > curr_data) and (period > 3)):
            adjustment = point_on_line(temp_start_frame, temp_start_val, curr_index, curr_data, peak_index)
            peak_adj = peak - adjustment
            temp_end_frame = curr_index
            if (peak_adj < noise_threshold):
                prev_data = curr_data
                cont = 1
            else:
                cont = 0
                peaks.append(peak_index)
                valleys.append(curr_index)
                if ((period < target_period + period_tolerance) and (period > target_period - period_tolerance) and (peak_adj < max_peak)):
                        cycle = CardiacCycle(peak_adj, period, temp_start_frame, temp_end_frame)
                        frame.append(cycle)
                        counter += 1
                        if (counter == num_cycles):
                            temp_avg_peak_frame = 0.0
                            temp_avg_period_frame = 0.0
                            for cycle in frame:
                                temp_avg_peak_frame += cycle.peak
                                temp_avg_period_frame += cycle.period
                            temp_avg_peak_frame /= num_cycles
                            temp_avg_period_frame /= num_cycles
                            if (temp_avg_peak_frame > avg_peak_frame):
                                avg_period_frame = temp_avg_period_frame
                                avg_peak_frame = temp_avg_peak_frame
                                start_frame = frame[0].start_index
                                end_frame = temp_end_frame
                            counter -= 1
                            temp_avg_peak_frame = 0.0
                            temp_avg_period_frame = 0.0
                            frame.popleft()
                else:
                    counter = 0
                    temp_avg_peak_frame = 0.0
                    temp_avg_period_frame = 0.0
                    frame.clear()
                temp_start_frame = curr_index
                temp_start_val = curr_data

        # identify peak (at maxima)
        if ((prev_data < curr_data) and (next_data < curr_data)):
            if (cont == 1 and curr_data < peak):
                continue
            peak = curr_data
            peak_index = curr_index
        prev_data = curr_data

    if (end_frame == 0.0):
        print("No {} valid consecutive cardiac cycles found in sample. \n".format(num_cycles))

    else:
        output_data = input_data[start_frame:end_frame + 1]

        # print("Values in stream: \n")
        # for element in output_data:
        #     print(element)

        print("The best {} consecutive cardiac cycles begin at index {} and end at index {}, \nwith an average peak of {} and period of {} \n".format( 
            num_cycles, start_frame, end_frame, avg_peak_frame, avg_period_frame))
    
    if pk == 1:
        return peaks
    else:
        return valleys


    