# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- functions.py : python script with general functions                                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

# Libraries and dependencies
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

credit_mix_numerical =  lambda x: 0 if x == 'Bad' else 2 if x == 'Good' else 1

credit_score_numerical =  lambda x: 0 if x == 'Poor' else 2 if x == 'Good' else 1

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

def data_cleaning(data : pd.DataFrame, drop_variables : list, age_variable : str, credit_mix : str, 
                  credit_score : str, cero_variables : list, numeric_variables : list, abs_variables : list,
                  null_variables : str):
    '''
    This function cleans the model credit score datasets

    Parameters
    ----------
        data : pd.DataFrame
            The dataset that will be cleaned
        drop_variables: list
            List with the variables that will be droped from the dataset
        age_variable: str
            The variable that contains the age on 'NN Years and NN Months'
        credit_mix_variables: str
            The variable that has string values on credit_mix (Bad, Standard and Good) 
        credit_score_variables: str
            The variable that has string values on credit_score (Poor, Standard and Good)
        cero_variables: list
            List with the variables that have null values that will be filled with 0
        numeric_variables: list
            List with the variables that has string values that are numbers
        abs_variables: list
            List with the variables that have negative values they are really positive
        null_variables : list
            List of the variables that have null values that will be replaced with the info of the last record
    
    Returns
    -------
    pd.DataFrame
        A dataframe with the data cleaned
    '''
    df = data.drop(columns = drop_variables, axis = 1)
    for var in cero_variables:
        df[var] = fill_ceros(data = df, var = var)
    for var in null_variables:
        df[var] = df[var].bfill()
    df[age_variable] = numeric_age(df,age_variable)
    df[credit_mix] = [credit_mix_numerical(x) for x in df[credit_mix]]
    try:
        df[credit_score] = [credit_score_numerical(x) for x in df[credit_score]]
    except:
        pass
    for var in numeric_variables:
        df[var] = [only_numbers(x) for x in df[var]]
    for var in abs_variables:
        df[var] = df[var].abs()     
    return df

score =  lambda x: 'Poor' if x == 0 else 'Good' if x == 2 else 'Standard'