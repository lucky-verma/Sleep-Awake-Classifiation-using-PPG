import os
import shutil
import glob
import datetime
import time
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pycaret.classification import *

# Read the Files
data = pd.read_csv("final_processed.csv", index_col=0)
logging.info("Data read successfully")

# select the columns
data = data.interpolate()[[
    'ledGreen', 'accelerometer', 'gyroscope', 'bpm', 'breathingrate', 'hr',
    'sleep_state'
]].dropna()
logging.info("Data preprocessed successfully")


# drop rows where 'hr' column have values starting with '['
data = data[~data['hr'].str.startswith('[', na=False)]
logging.info("Data cleaned successfully")


exp_clf102 = setup(
    data=data,
    target='sleep_state',
    session_id=123,
    use_gpu=True,
    normalize=True,
    transformation=True,
    imputation_type=None,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
)
logging.info("Data setup successfully")

logging.info("Training started")
best = compare_models()

logging.info("Training completed")

# save the model with today's date
save_model(best, 'best_model_' + str(datetime.date.today()))
logging.info("Model saved successfully")
