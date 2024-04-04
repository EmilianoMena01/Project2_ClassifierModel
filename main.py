# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- main.py : python script with the main functionality                                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

# Libraries and dependencies
from data import train, test
from visualizations import confusion_matrix, correlation, pie
import functions as f
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Data
## List the variables of every group
drop_variables = ['ID','Month','Name','Age','SSN','Occupation','Type_of_Loan','Payment_of_Min_Amount',
                  'Payment_Behaviour']
cero_variables = ['Num_of_Delayed_Payment','Num_Credit_Inquiries','Amount_invested_monthly','Monthly_Balance']
null_variables = ['Monthly_Inhand_Salary','Credit_History_Age']
age_variable = 'Credit_History_Age' 
credit_mix = 'Credit_Mix' 
credit_score = 'Credit_Score'
numeric_variables = ['Annual_Income','Num_of_Loan','Num_of_Delayed_Payment','Changed_Credit_Limit',
                     'Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']
abs_variables = ['Num_Bank_Accounts','Delay_from_due_date'] 
model_variables = ['Monthly_Inhand_Salary','Delay_from_due_date','Num_of_Delayed_Payment','Credit_History_Age',
                   'Num_Credit_Inquiries','Credit_Mix','Outstanding_Debt','Credit_Utilization_Ratio']

## Explore the data
train_info = f.analyse_data(train)
test_info = f.analyse_data(test)
train_describe = train.describe()
test_describe = test.describe()

## Clean the data and make all numeric values for the classifiers
df_train =  f.data_cleaning(train, drop_variables, age_variable, credit_mix, credit_score, cero_variables, 
                            numeric_variables, abs_variables, null_variables)
df_test =  f.data_cleaning(test, drop_variables, age_variable, credit_mix, credit_score, cero_variables, 
                            numeric_variables, abs_variables, null_variables)

## Get the x and y of the train, validation and test datasets
features = df_train[model_variables]
labels = df_train['Credit_Score']
x_train_or, x_validation_or, y_train, y_validation = train_test_split(features, labels, test_size=0.25, random_state=42) 
x_test_or = df_test[model_variables]
ct = ColumnTransformer([('norm1', Normalizer(norm='l1'), model_variables)], remainder='passthrough')
x_train = ct.fit_transform(x_train_or)
x_validation = ct.fit_transform(x_validation_or)
x_test = ct.transform(x_test_or)

# Models
## K-Neighbors Classifier
clf1 = KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1)
clf1.fit(x_train, y_train)
y_pred1 = clf1.predict(x_validation)
accuracy1 = accuracy_score(y_validation, y_pred1)
cv1 = cross_val_score(clf1, x_validation, y_validation)

## Random Forest Classifier
clf2 = RandomForestClassifier(n_jobs=-1)
clf2.fit(x_train, y_train)
y_pred2 = clf2.predict(x_validation)
accuracy2 = accuracy_score(y_validation, y_pred2)
cv2 = cross_val_score(clf2, x_validation, y_validation)

## Extremely Randomized Trees Classifier
clf3 = ExtraTreesClassifier(n_jobs=-1)
clf3.fit(x_train, y_train)
y_pred3 = clf3.predict(x_validation)
accuracy3 = accuracy_score(y_validation, y_pred3)
cv3 = cross_val_score(clf3, x_validation, y_validation)

## Gradient Boosting Classifier
clf4 = HistGradientBoostingClassifier()
clf4.fit(x_train, y_train)
y_pred4 = clf3.predict(x_validation)
accuracy4 = accuracy_score(y_validation, y_pred4)
cv4 = cross_val_score(clf4, x_validation, y_validation)

## Stacking (K-Neighbors, Random Forest, Extremely Randomized Trees and Gradient Boosting) Classifier
estimators = [('K', KNeighborsClassifier(algorithm='ball_tree', n_jobs=-1)),
              ('rf', RandomForestClassifier(n_jobs=-1)),
              ('et', ExtraTreesClassifier(n_jobs=-1))]
clf5 = StackingClassifier(estimators=estimators, final_estimator=HistGradientBoostingClassifier(), n_jobs=-1)
clf5.fit(x_train, y_train)
y_pred5 = clf5.predict(x_validation)
accuracy5 = accuracy_score(y_validation, y_pred5)
cv5 = cross_val_score(clf5, x_validation, y_validation)

## Final results of every model
models = ['K-Neighbors', 'Random Forest', 'Extremely Randomized Trees', 'Gradient Boosting',
          'Stacking (K-Neighbors, Random Forest, Extremely Randomized Trees and Gradient Boosting)']
accuracy = [round(accuracy1*100,2), round(accuracy2*100,2), round(accuracy3*100,2), round(accuracy4*100,2),
            round(accuracy5*100,2)]
cv_accuracy = [round(cv1.mean()*100,2), round(cv2.mean()*100,2), round(cv3.mean()*100,2),
               round(cv4.mean()*100,2), round(cv5.mean()*100,2)]
cv_std = [round(cv1.std()*100,2), round(cv2.std()*100,2), round(cv3.std()*100,2),
               round(cv4.std()*100,2), round(cv5.std()*100,2)]
metrics = pd.DataFrame({'Model' : models, 'Traditional accuracy' : accuracy, 
                        'Cross validation accuracy' : cv_accuracy, 'Standard deviation' : cv_std})

## Predict the test with the best model
y_test_predicted = clf5.predict(x_test)
test['Credit_Score'] = [f.score(x) for x in y_test_predicted]
predictions = test.groupby('Credit_Score')['Credit_Score'].count()
percentage = [round(x/len(test)*100,2) for x in predictions]
test_results = pd.DataFrame({'Number of Predicted Labels' : predictions, 'Percentage' : percentage})

## Train and Validation labels
x_train_or['Credit_Score'] = [f.score(x) for x in y_train]
train_labels = x_train_or.groupby('Credit_Score')['Credit_Score'].count()
tpercentage = [round(x/len(x_train_or)*100,2) for x in train_labels]
train_results = pd.DataFrame({'Number of Predicted Labels' : train_labels, 'Percentage' : tpercentage})

x_validation_or['Credit_Score'] = [f.score(x) for x in y_validation]
validation_labels = x_validation_or.groupby('Credit_Score')['Credit_Score'].count()
vpercentage = [round(x/len(x_validation_or)*100,2) for x in validation_labels]
validation_results = pd.DataFrame({'Number of Predicted Labels' : validation_labels, 'Percentage' : vpercentage})

# Visualizations
labels = ['Poor','Good','Standard']
cm1 = confusion_matrix(y_validation, y_pred1, labels, 'K-Neighbors', 'Model_1.png')
cm2 = confusion_matrix(y_validation, y_pred2, labels, 'Random Forest', 'Model_2.png')
cm3 = confusion_matrix(y_validation, y_pred3, labels, 'Extremely Randomized Trees', 'Model_3.png')
cm4 = confusion_matrix(y_validation, y_pred4, labels, 'Gradient Boosting', 'Model_4.png')
cm5 = confusion_matrix(y_validation, y_pred5, labels, 'Stacking', 'Model_5.png')
corr = correlation(pd.DataFrame(x_train, columns = model_variables), 'Correlation.png')
gp1 = pie(train_labels.values, train_labels.index, 'Train.png')
gp2 = pie(validation_labels.values, validation_labels.index, 'Validation.png')
gp3 = pie(predictions.values, predictions.index, 'Test.png')