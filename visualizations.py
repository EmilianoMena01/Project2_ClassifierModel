# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- visualizations.py : python script with data visualization functions                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

# Libraries and dependencies
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Confusion matrix
def confusion_matrix(y_original : list, y_predicted : list, labels : list, title : str, save : str):
    '''
    Create a confusion matrix plot

    Parameters
    ----------
        y_original : list
            The true labels of the test
        y_predicted : list
            The labels predicted by the model
        labels : list
            The labels values
        title : str
            Title of the plot, name of the model
        save : str
            Name to save the plot e.g. 'Image.png'
        
    Returns
    -------
    plt
        A heatmap plot
    '''
    plt.figure()
    confusion_matrix = metrics.confusion_matrix(y_original, y_predicted)
    sns.heatmap(confusion_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels).set(
        title=title,xlabel='Original',ylabel='Model')
    plt.savefig(save)
    plt.close()

def correlation(data : pd.DataFrame, save : str):
    '''
    Create a confusion matrix plot

    Parameters
    ----------
        data : pd.DataFrame
            Dataset with all the variables
        save : str
            Name to save the plot e.g. 'Image.png'
        
    Returns
    -------
    plt
        A heatmap plot
    '''
    plt.figure()
    sns.set_theme(style="white")
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(20, 20))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(save)
    plt.close()

def pie(values : list, labels : list, save : str):
    '''
    Create a pie graph of the predicted values of the model

    Parameters
    ----------
        values : list
            List with the total predictions of every class
        labels : list
            List with the classes labels
        save : str
            Name to save the plot e.g. 'Image.png'
        
    Returns
    -------
    plt
        A pie plot
    '''
    plt.figure(figsize=(6,6))
    plt.pie(values,labels=labels,autopct = "%0.2f%%");
    plt.legend(loc=1)
    plt.savefig(save)
    plt.close()