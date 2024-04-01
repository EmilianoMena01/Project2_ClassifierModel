# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- functions.py : python script with general functions                                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

## Libraries and dependencies
import pandas as pd
import numpy as np
import string
import warnings

## Data Analysis and Data Cleaning
def analyse_data(data : pd.DataFrame):
    '''
    The function returns information of every column of a dataset (Name of the variable, Type of the data,
    Null values, Present values and Unique values)
    
    Parameters
    ----------
        data : pd.DataFrame
            The dataset that will be analysed
        
    Returns
    -------
    pd.DataFrame
        A dataset with the information of every column of the dataset entered
    '''
    df = pd.DataFrame({
        'Variable_name':list(data.columns.values), # Name of the variable (column)
        'Type':list(data.dtypes), # Type of data
        'NA_values':list(data.isnull().sum()), # Null values
        'Present_values':list(data.count()), # Number of values each variable has
        'Unique_values':[data[x].nunique() for x in data.columns.values], # Unique values of each variable
    }).set_index('Variable_name')
    return df

def fill_info_same_customer(data: pd.DataFrame , id_var: str , fill_var: str):
    '''
    The function fill the null values and replace the modified values with the information of the same
    person
    
    Parameters
    ----------
        data : pd.DataFrame
            The dataset that wants to be filled
        id_var: str
            Variable that identifies the person
        fill_var: str
            Variable that will be filled or replaced
    
    Returns
    -------
    list
        A list with all the values present
    '''
    k = list(data[id_var])
    v = list(data[fill_var].ffill())
    c = {x:y for x,y in zip(k,v)}
    result = [c.get(x,0) for x in k]
    return result

def fill_ceros(data: pd.DataFrame , var: str):
    '''
    The function fills the variables null values with 0
    Parameters
    ----------
        data : pd.DataFrame
            The dataset that wants to be filled
        var: str
            Variable that null values will be replaced with 0
    
    Returns
    -------
    list
        A list with all the values present
    '''
    result = data[var].fillna(0)
    return result

def only_numbers(var: str):
    '''
    The function receives a string that contains numbers and punctuaction or other elements and returns
    only the number on float format

    Parameters
    ----------
        var : str
            String that has a number
    
    Returns
    -------
    float
        A float number
    '''
    try:
        var = ''.join(ch for ch in var if ch in string.digits+'.')
    except:
        pass
    result = [float(0) if var=='' or var=='.' or var=='-' or var=='-.' else float(var)][0]
    return result

def numeric_age(data: pd.DataFrame , var: str):
    '''
    The function converts the age from a format of NN Years to NN Months to numeric months. It receives the
    dataset and the variable that has the age
    
    Parameters
    ----------
        data : pd.DataFrame
            The dataset that contains the variable
        var: str
            Variable of the age that will be converted to numeric
    
    Returns
    -------
    list
        A list with all the ages in numeric (int) type
    '''
    result = [int(xi[0])*12 + int(xi[3]) for xi in [x.split() for x in data[var]]]
    return result

def outliers_replace(data : pd.DataFrame, var : str, num_std : int = 1):
   '''
   This function detects and replaces the outliers with the previous value

    Parameters
    ----------
        data : pd.DataFrame
            The dataset that contains the variable
        var: str
            Variable that has outliers
        num_std: int
            Number of standard desviations that will be considered to count as a outlier. By default takes
            1 standard desviation
    
    Returns
    -------
    list
        A list with the original values and the replaced outliers
   '''
   mean = data[var].mean()
   std = data[var].std()
   low = mean - (num_std * std)
   high = mean + (num_std * std)
   outliers = data[(data[var]<low) | (data[var] > high)][var]
   data[var] = data[var].replace(outliers.values,np.nan)
   result = data[var].ffill()
   return result

## Punctuation Model
def punctuation(scores: list, ranges: list, values: list, condition: int = 0):
    '''
    This function receives the values of the variables selected and gives them a score based on the ranges 
    entered considering the condition selected.  

    Parameters
    ----------
        scores : list
            All the scores that every value can get
        ranges: list
            The ranges on which every score will be assigned
        values: list
            The values of every variable selected
        conditions: int
            A flag used to select the type of if statement to evaluate if is in the range or not, 0 is '<=', 
            1 is '==' and 2 is '>='
    
    Returns
    -------
    list
        A list with all punctuations of a variable
    '''
    if condition == 0:
        result = [max([z if xi <= y else scores[-1] for z,y in zip(scores[:-1],ranges)]) for xi in values]
    elif condition == 1:
        result = [max([z if xi == y else scores[-1] for z,y in zip(scores[:-1],ranges)]) for xi in values]
    elif condition == 2:
        result = [max([z if xi >= y else scores[-1] for z,y in zip(scores[:-1],ranges)]) for xi in values]
    else:
        raise Warning("The condition options are 0 for <=, 1 for ==, 3 for >=!")
    return result

def final_punctuation(punctuations: list):
    '''
    This function receives the punctuations of every variable and gets back the final punctuation (sum all
    of them)

    Parameters
    ----------
        punctuations : list
            List of lists that contain all the variables punctuation
    
    Returns
    -------
    list
        A list with the final punctuation of every person
    '''
    result = [sum(x) for x in zip(*punctuations)]
    return result

def score(range1 : int, range2 : int, punctuation : int):
    '''
    This function assign the final score to every punctuation
    
    Parameters
    ----------
        range1 : int
            The first range is a number that will be compared with the punctuation, if the punctuation is 
            under the range number the score will be Poor, if the punctuation is above the range number the
            score will be Standard
        range2 : int
            The second range is a number that will be compared with the punctuation, if the punctuation is 
            under the range number the score will be Standard, if the punctuation is above the range number 
            the score will be Good
        punctuation : int
            The punctuation that a person got on the model
    
    Returns
    -------
    str
        The score of the given punctuation
    '''
    if punctuation < 600:
        result = 'Poor' 
    elif punctuation < 800:
        result = 'Standard' 
    else:
        result = 'Good'
    return result

def df_results(data: pd.DataFrame, punctuation: list, scores: list):
    '''
    This function returns a dataset that has the original scores, the model scores, the model punctuation
    and some info of every person to know whose score it is
    
    Parameters
    ----------
        data : pd.DataFrame
            Dataset with the all the information
        punctuation : list
            The model punctuation
        scores : list
            The model scores
    
    Returns
    -------
    pd.DataFrame
        A dataset that includes the Customer_ID, Name, Model Punctuation, Model Score and Original Score
    '''
    result =  pd.DataFrame({
        'Customer_ID': data['Customer_ID'],
        #'Name': data['Name'],
        'Model_Punctuation': punctuation,
        'Model_Score': scores,
        'Original_Score': data['Credit_Score']
        })
    return result

def accuracy(scores: list, data: pd.DataFrame):
    '''
    This function shows the model accuracy
    
    Parameters
    ----------
        scores : list
            List of the model scores
        data : pd.DataFrame
            Dataset with the all the information
    
    Returns
    -------
    float
        The accuracy expressed as a percentage (multiplied by 100)
    '''
    result = sum([1 if x==y else 0 for x,y in zip(scores,data['Credit_Score'])])/len(data)
    return result

