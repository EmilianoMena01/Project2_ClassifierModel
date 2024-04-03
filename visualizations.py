# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- visualizations.py : python script with data visualization functions                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

# Libraries and dependencies
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion matrix graph
def confusion_matrix(y, y_pred, labels):
    cm = confusion_matrix(y, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()