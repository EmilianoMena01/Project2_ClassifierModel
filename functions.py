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

def data_cleaning(data : pd.DataFrame, drop_variables : list, customer_variables : list, cero_variables : list, 
                  numeric_variables : list, outliers_variables : list):
   '''
   This function cleans the model credit score datasets

    Parameters
    ----------
        data : pd.DataFrame
            The dataset that contains the variable
        drop_variables: list
            List with the variables that will be droped from the dataset
        customer_variables: list
            List with the variables that will use the fill_info_same_customer function
        cero_variables: list
            List with the variables that will use the fill_ceros
        numeric_variables: list
            List with the variables that will use the only_numbers
        outliers_variables: list
            List with the variables that will use the outliers_replace
    
    Returns
    -------
    pd.DataFrame
        A dataframe with the data cleaned
   '''
   df = data.drop(columns = drop_variables, axis = 1) # Drop the variables that won´t be used
   for var in customer_variables: # Fill the variables with the info of the same Customer
      df[var] = fill_info_same_customer(data = df, id_var = 'Customer_ID', fill_var = var)
   for var in cero_variables: # Fill the variables with 0
      df[var] = fill_ceros(data = df, var = var)
   for var in numeric_variables: # Convert string numbers to float numbers
      df[var] = [only_numbers(x) for x in df[var]]
   df['Credit_History_Age'] = numeric_age(data = df, var = 'Credit_History_Age')# Get the number of months
   for var in outliers_variables: # Replace the outliers
      df[var] = outliers_replace(df,var,1)
   return df