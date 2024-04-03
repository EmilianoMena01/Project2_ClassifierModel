# -- ------------------------------------------------------------------------------------------------- -- #
# -- Project2_ClassifierModel                                                                          -- #
# -- main.py : python script with the main functionality                                               -- #
# -- authors: Emiliano Mena González and Jorge Alberto Hernández Hernández                             -- #
# -- repository: https://github.com/EmilianoMena01/Project2_ClassifierModel                            -- #
# -- ------------------------------------------------------------------------------------------------- -- #

# Libraries and dependencies
from data import train, test
import functions as f
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, StackingClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Prepare the data
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
model_variables = ['Delay_from_due_date','Num_of_Delayed_Payment','Num_Credit_Inquiries','Credit_Mix',
                   'Outstanding_Debt','Credit_History_Age']
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
x_train, x_validation, y_train, y_validation = train_test_split(features, labels, test_size=0.25, random_state=42) 
x_test = df_test[model_variables]
ct = ColumnTransformer([('norm1', Normalizer(norm='l1'), model_variables)], remainder='passthrough')
x_train = ct.fit_transform(x_train)
x_validation = ct.fit_transform(x_validation)
x_test = ct.transform(x_test)

# Stacking (Decision Tree, Random Forest and Gradient Boosting)
estimators = [('GB',GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, criterion='squared_error')),
              ('rf',RandomForestClassifier(criterion='entropy'))]
clf1 = StackingClassifier(estimators=estimators, 
                         final_estimator=DecisionTreeClassifier())
clf1.fit(x_train, y_train)
y_pred1 = clf1.predict(x_validation)
accuracy1 = accuracy_score(y_validation, y_pred1)
cv1 = cross_val_score(clf1, x_validation, y_validation)

# Decision Tree Classifier
clf2 = DecisionTreeClassifier()
clf2.fit(x_train, y_train)
y_pred2 = clf2.predict(x_validation)
accuracy2 = accuracy_score(y_validation, y_pred2)
cv2 = cross_val_score(clf2, x_validation, y_validation)

# Gradient Boosting Classifier
clf3 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, criterion='squared_error')
clf3.fit(x_train, y_train)
y_pred3 = clf3.predict(x_validation)
accuracy3 = accuracy_score(y_validation, y_pred3)
cv3 = cross_val_score(clf3, x_validation, y_validation)

# Random Forest Classifier
clf4 = RandomForestClassifier(criterion='entropy')
clf4.fit(x_train, y_train)
y_pred4 = clf4.predict(x_validation)
accuracy4 = accuracy_score(y_validation, y_pred4)
cv4 = cross_val_score(clf4, x_validation, y_validation)

# AdaBoost Classifier
clf5 = AdaBoostClassifier(n_estimators=100)
clf5.fit(x_train, y_train)
y_pred5 = clf5.predict(x_validation)
accuracy5 = accuracy_score(y_validation, y_pred5)
cv5 = cross_val_score(clf5, x_validation, y_validation)

# Neural Network (Multi Layer Perceptron) Classifier
clf6 = MLPClassifier(hidden_layer_sizes=(10, 5), learning_rate_init=0.05, early_stopping=True)
clf6.fit(x_train, y_train)
y_pred6 = clf6.predict(x_validation)
accuracy6 = accuracy_score(y_validation, y_pred6)
cv6 = cross_val_score(clf6, x_validation, y_validation)

# Final results of every model
models = ['Stack (Decision Tree, Random Forest and Gradient Boosting)','Decision Tree', 'Gradient Boosting',
          'Random Forest', 'Adaboost','Neural Network (Multi Layer Perceptron)']
accuracy = [accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6]
cv_accuracy = [cv1.mean(),cv2.mean(),cv3.mean(),cv4.mean(),cv5.mean(),cv6.mean()]
cv_std = [cv1.std(),cv2.std(),cv3.std(),cv4.std(),cv5.std(),cv6.std()]
metrics = pd.DataFrame({'Model' : models, 'Traditional accuracy' : accuracy, 'Cross validation accuracy' : cv_accuracy,
              'Standard deviation' : cv_std})

# Predict the test with the best model
test['Credit_Score'] = [f.score(x) for x in clf1.predict(x_test)]