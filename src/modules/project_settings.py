
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd

# import seaborn as sns
# sns.set_theme(style='darkgrid', context='talk')

import os
from importlib import reload


import src.modules.data_cleaning as dc
import src.modules.data_preprocessing as dp


global DATA_FOLDER, SRC_FOLDER, MODULES_FOLDER, TESTS_FOLDER, OUTPUT_FOLDER, FIGURES_FOLDER

# Define folder paths as global variables
DATA_FOLDER = "data/"
SRC_FOLDER = "src/"
MODULES_FOLDER = "src/modules/"
TESTS_FOLDER = "src/tests/"
OUTPUT_FOLDER = "output/"
FIGURES_FOLDER = "output/figures/"

global DATA_PATH, DATA_ORIGINAL_PATH, DATA_GENERATED_PATH, IMAGE_GENERATED_PATH

DATA_PATH = DATA_FOLDER
DATA_ORIGINAL_PATH = DATA_FOLDER
DATA_GENERATED_PATH = OUTPUT_FOLDER
IMAGE_GENERATED_PATH = FIGURES_FOLDER


global COLORS

COLORS = ['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

global LSTM1_OUT_PATH
LSTM1_OUT_PATH = DATA_GENERATED_PATH + 'LSTM_1/'