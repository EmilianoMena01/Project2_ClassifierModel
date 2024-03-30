# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- main.py : python script with the main functionality                                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

# Libraries and dependencies
from data import train, test
import functions as f

# Exploratory Data Analysis of the train and test dataframes
EDA_train = f.analyse_data(train)
EDA_test = f.analyse_data(test)