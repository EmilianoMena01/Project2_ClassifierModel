# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- data.py : python script for data collection                                                       -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

# Libraries and dependencies
import pandas as pd

# Load the train and test dataframes
train = pd.read_csv('files/train-1.csv',low_memory=False)
test = pd.read_csv('files/test.csv',low_memory=False)