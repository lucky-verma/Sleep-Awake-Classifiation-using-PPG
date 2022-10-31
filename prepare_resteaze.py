import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd
import datetime

from sleepstage import resteaze_stage_dict
from logger import get_logger

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./prof_data/resteaze",
                        help="File path to the resteaze dataset.")
    parser.add_argument("--output_dir", type=str, default="./prof_data/resteaze/ppg_ledgreen",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="LedGreen",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = get_logger(args.log_file, level="info")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation from EDF files
    ppg_fnames = glob.glob(os.path.join(args.data_dir, "*.csv"))
    ppg_fnames.sort()
    ppg_fnames = np.asarray(ppg_fnames)

    for i in range(len(ppg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(ppg_fnames[i]))


        df = pd.read_csv(ppg_fnames[i], sep=',')
        ppg_df = df[['unixTimes', 'LedGreen', 'sleep_stage', 'sleep_state']].dropna().reset_index(drop=True)

        start_datetime = datetime.datetime.fromtimestamp(ppg_df['unixTimes'][0] / 1000)
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = datetime.datetime.fromtimestamp((ppg_df['unixTimes'][len(ppg_df)-1] - ppg_df['unixTimes'][0]) / 1000)
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = 30
        logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signal from the selected channel
        ch_samples = len(ppg_df[select_ch])

        sampling_rate = 25
        n_epoch_samples = int(epoch_duration * sampling_rate)

        # apply bandpass filter

        fs = 25
        lowcut = 0.35
        highcut = 5.0

        pro_ppg = butter_bandpass_filter(ppg_df[select_ch], lowcut, highcut, fs, order=5)

        signals = pro_ppg[:-(ppg_df.shape[0] % n_epoch_samples)].reshape(-1, n_epoch_samples)
        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(ch_samples))
        logger.info("Sample rate: {}".format(sampling_rate))

        # Sanity check
        n_epochs = signals.shape[0]

        # Generate labels from onset and duration annotation
        labels = []

        sleep_state = ppg_df['sleep_state'][:-(ppg_df.shape[0] % n_epoch_samples)]
        k = 0
        for j in range(n_epochs):
            tmp = j*750
            labels.append(round(sum(sleep_state[k:tmp] / 750)))
            k = tmp

        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != resteaze_stage_dict["WK"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # # Remove movement and unknown
        # move_idx = np.where(y == stage_dict["MOVE"])[0]
        # unk_idx = np.where(y == stage_dict["UNK"])[0]
        # if len(move_idx) > 0 or len(unk_idx) > 0:
        #     remove_idx = np.union1d(move_idx, unk_idx)
        #     logger.info("Remove irrelavant stages")
        #     logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
        #     logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
        #     logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
        #     logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
        #     select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        #     x = x[select_idx]
        #     y = y[select_idx]
        #     logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(ppg_fnames[i]).replace(".csv", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": len(x),
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()
