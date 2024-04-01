# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- main.py : python script with the main functionality                                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

## Libraries and dependencies
from data import train, test
import functions as f
import pandas as pd 
from sklearn.model_selection import train_test_split

## Data cleaning
data_info = f.analyse_data(train) # Analyse the data
drop_variables = ['ID','Month','Age','SSN','Occupation','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan',
             'Type_of_Loan','Payment_of_Min_Amount','Payment_Behaviour'] # Variables to drop
df2 = train.drop(columns = drop_variables, axis = 1) # Drop the variables that won´t be used
customer_variables = ['Name','Annual_Income','Monthly_Inhand_Salary','Interest_Rate','Credit_Mix',
                      'Credit_History_Age']
for var in customer_variables: # Fill the variables with the info of the same Customer
   df2[var] = f.fill_info_same_customer(data = df2, id_var = 'Customer_ID', fill_var = var)
cero_variables = ['Num_of_Delayed_Payment','Num_Credit_Inquiries','Amount_invested_monthly','Monthly_Balance']
for var in cero_variables: # Fill the variables with 0
   df2[var] = f.fill_ceros(data = df2, var = var)
numeric_variables = ['Annual_Income','Num_of_Delayed_Payment','Changed_Credit_Limit','Outstanding_Debt',
            'Amount_invested_monthly','Monthly_Balance']
for var in numeric_variables: # Convert string numbers to float numbers
   df2[var] = [f.only_numbers(x) for x in df2[var]]
df2['Credit_History_Age'] = f.numeric_age(data = df2, var = 'Credit_History_Age')# Get the number of months
outliers_variables = ['Annual_Income','Interest_Rate','Num_of_Delayed_Payment','Num_Credit_Inquiries',
                      'Total_EMI_per_month','Amount_invested_monthly','Monthly_Balance']
data_stadistics = df2.describe()
df3 = df2.copy()
for var in outliers_variables: # Replace the outliers
   df3[var] = f.outliers_replace(df3,var,1)

#Se tienen muchas filas de datos sobre un mismo ID, algunas filas del mismo ID tienen datos diversos pero tienen algo en común, ya sea su credit score, su credit mix, entonces
#podemos juntar (promediar) aquellas filas de un mismo ID que cumplan ciertas condiciones por ejemplo _Credit mix bueno y Credit score bueno, entonces se promedian esas filas y se hacen una 
#Entonces así reducimos filas de datos pero manteniendo la información relevante
# -- ------------------------------------------------------------------------------------------------- -- #
   #Data filtered and mean values
#Credit Mix is Good 
good_mix_sc_good = df3[(df3["Credit_Mix"] == "Good") & (df3["Credit_Score"] == "Good")]
good_mix_sc_good=good_mix_sc_good.set_index("Customer_ID")
good_mix_sc_good_numeric = good_mix_sc_good.select_dtypes(include='number')
good_mix_sc_good = good_mix_sc_good_numeric.groupby("Customer_ID").mean().reset_index()
good_mix_underscore = ["Good"] * len(good_mix_sc_good)
good_mix_sc_good['Credit_Score'] = pd.Series(['Good'] * len(good_mix_sc_good), index=good_mix_sc_good.index)

good_mix_sc_st = df3[(df3["Credit_Mix"] == "Good") & (df3["Credit_Score"] == "Standard")]   
good_mix_sc_st = good_mix_sc_st.set_index("Customer_ID")
good_mix_sc_st_numeric = good_mix_sc_st.select_dtypes(include='number')
good_mix_sc_st = good_mix_sc_st_numeric.groupby("Customer_ID").mean().reset_index()
st_mix_good = ["Standard"] * len(good_mix_sc_st)
good_mix_sc_st['Credit_Score'] = pd.Series(['Standard'] * len(good_mix_sc_st), index=good_mix_sc_st.index)

good_mix_sc_bad = df3[(df3["Credit_Mix"] == "Good") & (df3["Credit_Score"] == "Poor")]
good_mix_sc_bad = good_mix_sc_bad.set_index("Customer_ID")
good_mix_sc_bad_numeric = good_mix_sc_bad.select_dtypes(include='number')
good_mix_sc_bad = good_mix_sc_bad_numeric.groupby("Customer_ID").mean().reset_index()
bad_mix_poor = ["Poor"] * len(good_mix_sc_bad)
good_mix_sc_bad['Credit_Score'] = pd.Series(['Poor'] * len(good_mix_sc_bad), index=good_mix_sc_bad.index)

good_mix = pd.concat([good_mix_sc_good, good_mix_sc_st, good_mix_sc_bad])
good_column=["Good"]*len(good_mix)    
good_mix["Credit_Mix"]=pd.Series(["Good"]*len(good_mix), index=good_mix.index)


#Credit Mix is Standard
standard_mix_sc_good = df3[(df3["Credit_Mix"] == "Standard") & (df3["Credit_Score"] == "Good")]
standard_mix_sc_good=standard_mix_sc_good.set_index("Customer_ID")
standard_mix_sc_good_numeric = standard_mix_sc_good.select_dtypes(include='number')
standard_mix_sc_good= standard_mix_sc_good_numeric.groupby("Customer_ID").mean().reset_index()
good_mix_standard = ["Good"] * len(standard_mix_sc_good)
standard_mix_sc_good['Credit_Score'] = pd.Series(['Good'] * len(standard_mix_sc_good), index=standard_mix_sc_good.index)

standard_mix_sc_st = df3[(df3["Credit_Mix"] == "Standard") & (df3["Credit_Score"] == "Standard")]   
standard_mix_sc_st = standard_mix_sc_st.set_index("Customer_ID")
standard_mix_sc_st_numeric = standard_mix_sc_st.select_dtypes(include='number')
standard_mix_sc_st = standard_mix_sc_st_numeric.groupby("Customer_ID").mean().reset_index()
st_mix_standard = ["Standard"] * len(standard_mix_sc_st)
standard_mix_sc_st['Credit_Score'] = pd.Series(['Standard'] * len(standard_mix_sc_st), index=standard_mix_sc_st.index)

standard_mix_sc_bad = df3[(df3["Credit_Mix"] == "Standard") & (df3["Credit_Score"] == "Poor")]
standard_mix_sc_bad = standard_mix_sc_bad.set_index("Customer_ID")
standard_mix_sc_bad_numeric = standard_mix_sc_bad.select_dtypes(include='number')
standard_mix_sc_bad = standard_mix_sc_bad_numeric.groupby("Customer_ID").mean().reset_index()
bad_mix_standard = ["Poor"] * len(standard_mix_sc_bad)
standard_mix_sc_bad['Credit_Score'] = pd.Series(['Poor'] * len(standard_mix_sc_bad), index=standard_mix_sc_bad.index)

standard_mix = pd.concat([standard_mix_sc_good, standard_mix_sc_st, standard_mix_sc_bad])
standard_column=["Standard"]*len(standard_mix)    
standard_mix["Credit_Mix"]=pd.Series(["Standard"]*len(standard_mix), index=standard_mix.index)


#Credit Mix is _ (not exist)
underscore_mix_sc_good = df3[(df3["Credit_Mix"] == "_") & (df3["Credit_Score"] == "Good")]
underscore_mix_sc_good=underscore_mix_sc_good.set_index("Customer_ID")
underscore_mix_sc_good_numeric = underscore_mix_sc_good.select_dtypes(include='number')
underscore_mix_sc_good=underscore_mix_sc_good_numeric.groupby("Customer_ID").mean().reset_index()   
good_mix_underscore=["Good"]*len(underscore_mix_sc_good)
underscore_mix_sc_good['Credit_Score'] = pd.Series(['Good'] * len(underscore_mix_sc_good), index=underscore_mix_sc_good.index)
underscore_mix_sc_good

underscore_mix_sc_st = df3[(df3["Credit_Mix"] == "_") & (df3["Credit_Score"] == "Standard")]   
underscore_mix_sc_st=underscore_mix_sc_st.set_index("Customer_ID")
underscore_mix_sc_st_numeric = underscore_mix_sc_st.select_dtypes(include='number')
underscore_mix_sc_st=underscore_mix_sc_st_numeric.groupby("Customer_ID").mean().reset_index()
st_mix_underscore=["Standard"]*len(underscore_mix_sc_st)
underscore_mix_sc_st['Credit_Score'] = pd.Series(['Good'] * len(underscore_mix_sc_st), index=underscore_mix_sc_st.index)
underscore_mix_sc_st

underscore_mix_sc_bad = df3[(df3["Credit_Mix"] == "_") & (df3["Credit_Score"] == "Poor")]
underscore_mix_sc_bad=underscore_mix_sc_bad.set_index("Customer_ID")
underscore_mix_sc_bad_numeric = underscore_mix_sc_bad.select_dtypes(include='number')
underscore_mix_sc_bad=underscore_mix_sc_bad_numeric.groupby("Customer_ID").mean().reset_index()
bad_mix_underscore=["Poor"]*len(underscore_mix_sc_bad)
underscore_mix_sc_bad['Credit_Score'] = pd.Series(['Poor'] * len(underscore_mix_sc_bad), index=underscore_mix_sc_bad.index)
underscore_mix_sc_bad

underscore_mix=pd.concat([underscore_mix_sc_good, underscore_mix_sc_st, underscore_mix_sc_bad])
underscore_column=["Bad"]*len(underscore_mix)    
underscore_mix["Credit_Mix"]=pd.Series(["_"]*len(underscore_mix), index=underscore_mix.index)


#Credit Mix is Bad
poor_mix_sc_good = pd.DataFrame(df3[(df3["Credit_Mix"] == "Bad") & (df3["Credit_Score"] == "Good")])
poor_mix_sc_good = poor_mix_sc_good.set_index("Customer_ID")
poor_mix_sc_good_numeric = poor_mix_sc_good.select_dtypes(include='number')
poor_mix_sc_good = poor_mix_sc_good_numeric.groupby("Customer_ID").mean().reset_index()  # Evitar que el índice se convierta en columna
good_mix_poor = ["Good"] * len(poor_mix_sc_good)
poor_mix_sc_good['Credit_Score'] = pd.Series(['Good'] * len(poor_mix_sc_good))


poor_mix_sc_st = pd.DataFrame(df3[(df3["Credit_Mix"] == "Bad") & (df3["Credit_Score"] == "Standard")]   )
poor_mix_sc_st = poor_mix_sc_st.set_index("Customer_ID")
poor_mix_sc_st_numeric = poor_mix_sc_st.select_dtypes(include='number')
poor_mix_sc_st = poor_mix_sc_st_numeric.groupby("Customer_ID").mean().reset_index()
st_mix_poor = ["Standard"] * len(poor_mix_sc_st)
poor_mix_sc_st['Credit_Score'] = pd.Series(['Standard'] * len(poor_mix_sc_st), index=poor_mix_sc_st.index)


poor_mix_sc_bad = pd.DataFrame(df3[(df3["Credit_Mix"] == "Bad") & (df3["Credit_Score"] == "Poor")])
poor_mix_sc_bad = poor_mix_sc_bad.set_index("Customer_ID")
poor_mix_sc_bad_numeric = poor_mix_sc_bad.select_dtypes(include='number')
poor_mix_sc_bad = poor_mix_sc_bad_numeric.groupby("Customer_ID").mean().reset_index()
bad_mix_poor = ["Poor"] * len(poor_mix_sc_bad)
poor_mix_sc_bad['Credit_Score'] = pd.Series(['Poor'] * len(poor_mix_sc_bad), index=poor_mix_sc_bad.index)
poor_mix_sc_bad

poor_mix = pd.concat([poor_mix_sc_good, poor_mix_sc_st, poor_mix_sc_bad])
bad_column=["Bad"]*len(poor_mix)    
poor_mix["Credit_Mix"]=pd.Series(["Bad"]*len(poor_mix), index=poor_mix.index)

#Datos comprobados, en data frame original se tienen 2 tipos de credit mix, y en este también solo salen 2, la diferencia es que está agrupados y promediados, disminuyendo la cantidad de datos  
#poor_mix.loc[poor_mix["Customer_ID"]=="CUS_0x1140"]
#m.df3.loc[m.df3['Customer_ID'] == 'CUS_0x1140']

#Concatenate the three dataframes
df4=pd.concat([good_mix, standard_mix,underscore_mix,poor_mix])
df4=pd.DataFrame(df4)

## Punctuation
model_variables = ['Delay_from_due_date','Num_of_Delayed_Payment', # Payment history 35%
                   'Outstanding_Debt', # Amounts owed 30%
                   'Credit_History_Age', # Lengh of credit history 10%
                   'Credit_Mix', # Credit mix 15%
                   'Num_Credit_Inquiries'] # New credit 10%
scores = [[200,150,100,75,50,25],[150,120,90,60,30],[300,275,250,150,75,50],[10,30,50,70,85,100],
          [150,75,25],[100,85,60,45,30]]
ranges = [[3,5,15,30,60],[0,7,13,19],[500,1000,1500,2000,2500],[24,48,72,96,120],['Good','Standard'],
          [3,6,10,20]]
conditions = [0,0,0,0,1,0]
model_data = [df4[v].values for v in model_variables]
variables_punctuation = [f.punctuation(s,r,m,c) for s,r,m,c in zip(scores,ranges,model_data,conditions)]
model_punctuation = f.final_punctuation(variables_punctuation)
model_score = [f.score(600,800,x) for x in model_punctuation]
results = f.df_results(df4, model_punctuation, model_score)
accuracy = f.accuracy(model_score, df4)

#Train data
X = df4.drop(columns = ["Credit_Score","Customer_ID"],axis = 1)
y = df4["Credit_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)