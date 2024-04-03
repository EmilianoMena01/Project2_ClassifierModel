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

# Confusion matrix
def heatmap(original : list, model : list, labels : list):
    '''
    Create a violin plot to see outliers on the data

    Parameters
    ----------
        original : list
            The original data
        model : list
            The predicted data
        labels : list
            The different categories in which they were predicted the data
        
    Returns
    -------
    plt
        A heatmap plot
    '''
    confusion_matrix = metrics.confusion_matrix(original, model)
    sns.heatmap(confusion_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels).set(
        title='Confusion Matrix',xlabel='Original',ylabel='Model')
    plt.show()