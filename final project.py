# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:36:20 2019

@author: jinsith
"""

#importing required libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#Loading data
loan_df=pd.read_csv(r'E:\python code\project1.txt',delimiter="\t")

pd.set_option('display.max_columns',None)
loan_df

#creating copy of dataset
loan_df_rev=pd.DataFrame.copy(loan_df)

#finding the missing values
print(loan_df_rev.isnull().sum())

#finding percentage of missing values
loan_df_rev_null=pd.DataFrame({'Count':loan_df_rev.isnull().sum(),'Percent':100*loan_df_rev.isnull().sum()/len(loan_df_rev)})

loan_df_rev_null[loan_df_rev_null['Count']>0]

#eliminating missing values more than 75%
loan_df_rev=loan_df_rev.dropna(axis=1, thresh=int(0.75*len(loan_df_rev)))

#finding the missing values
print(loan_df_rev.isnull().sum())

#eliminating not req columns
loan_df_rev.drop(['application_type','emp_title','collection_recovery_fee','collections_12_mths_ex_med','dti','title','id','member_id','last_pymnt_d','last_credit_pull_d','zip_code'],axis=1,inplace=True)

#plotting heatmap

plt.figure(figsize=(20,20)) 
sns.set_context("paper", font_scale=1) 
##finding the correllation matrix and changing the categorical data to category for the plot. 
sns.heatmap(loan_df_rev.assign(grade=loan_df_rev.grade.astype('category').cat.codes, 
                         sub_grade=loan_df_rev.sub_grade.astype('category').cat.codes, 
                         term=loan_df_rev.term.astype('category').cat.codes, 
                         emp_length=loan_df_rev.emp_length.astype('category').cat.codes, 
                         verification =loan_df_rev.verification_status.astype('category').cat.codes, 
                         home=loan_df_rev.home_ownership.astype('category').cat.codes, 
                         purpose=loan_df_rev.purpose.astype('category').cat.codes).corr(), 
            annot=True, cmap='bwr',vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.show()

#removing linearity value
loan_df_rev.drop(['installment','revol_bal','out_prncp','grade','sub_grade','total_rec_prncp'],axis=1,inplace=True)

loan_df_rev['emp_length']=loan_df_rev['emp_length'].str.strip('years')
loan_df_rev["emp_length"]=loan_df_rev["emp_length"].replace('< 1 ',1)
loan_df_rev["emp_length"]=loan_df_rev["emp_length"].replace('10+ ',10)


loan_df_rev['term']=loan_df_rev['term'].str.strip('months')


loan_df_rev.drop(['pymnt_plan','purpose','tot_coll_amt','tot_cur_bal','total_rev_hi_lim'],axis=1,inplace=True)





#boxplot
import seaborn as sns
sns.boxplot(x=loan_df_rev['int_rate'])


def win_tech(data ,cols):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3-q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    for i in cols:
        data.loc[data[i]> upper[i], i] = upper[i]
        data.loc[data[i]< lower[i], i] = lower[i]
#new_df.columns

columns1 = ['loan_amnt','int_rate','annual_inc']

win_tech(loan_df_rev,columns1)

sns.boxplot(x=loan_df_rev['int_rate'])


#one hot encoding
loan_df_rev.columns
cols=['term','emp_length','home_ownership','verification_status']
loan_df_rev= pd.get_dummies(loan_df_rev, drop_first=True,columns=cols)


loan_df_rev.drop(['initial_list_status'],axis=1,inplace=True)


loan_df_rev.issue_d=pd.to_datetime(loan_df_rev.issue_d,infer_datetime_format=True)
col_name='issue_d'
print(loan_df_rev[col_name].dtype)


loan_df_rev.drop(['earliest_cr_line'],axis=1,inplace=True)
loan_df_rev.drop(['addr_state'],axis=1,inplace=True)

#traansformation
from scipy.stats import boxcox

numerical = loan_df_rev.columns[loan_df_rev.dtypes == 'float64'] 
for i in numerical: 
    if loan_df_rev[i].min() > 0: 
        transformed, lamb = boxcox(loan_df_rev.loc[loan_df[i].notnull(), i]) 
        if np.abs(1 - lamb) > 0.02: 
            loan_df_rev.loc[loan_df[i].notnull(), i] = transformed

print(loan_df_rev.isnull().sum())

loan_df_rev['revol_util']=loan_df_rev['revol_util'].fillna(loan_df_rev['revol_util'].mean())

#splitting data into test and train
split_data="2015-06-01"
loan_training=loan_df_rev[loan_df_rev['issue_d']<split_data]
loan_training=loan_training.drop(['issue_d'],axis=1)
loan_training.shape

loan_test=loan_df_rev[loan_df_rev['issue_d']>=split_data]
loan_test=loan_test.drop(['issue_d'],axis=1)
loan_test.shape

#selecting X and Y

X_train=loan_training.drop('default_ind',axis=1)
Y_train=loan_training['default_ind']
Y_train=Y_train.astype(int)
print(Y_train)


X_test=loan_test.drop('default_ind',axis=1)
Y_test=loan_test['default_ind']
Y_test=Y_test.astype(int)
print(Y_test)

#run model
from sklearn.linear_model import LogisticRegression

#create a model
classifier=LogisticRegression()

#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

print(classifier.coef_)
print(classifier.intercept_)

#accuracy

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)



#%%
#ADJUSTING THRESHOLD
#%%
#store predicted probabilities
y_pred_prob=classifier.predict_proba(X_test)
print(y_pred_prob)


#testing data calculate prob belong to class 0 and class1

#%%
#move threshold bit higher and predict what can be good threshold
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.94:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

print(y_pred_class)


for a in np.arange(0.1,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :",
          cfm[1,0]," , type 1 error:", cfm[0,1])
    

#%%
#GENERATE ROC CURVE
from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
#to generate fpr and tpr
print(auc)


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
#give title,name roc came into being from transmitter reciever
plt.plot(fpr, tpr, 'b', label = auc)
#want blue colored line
plt.legend(loc = 'lower right')
#where legend to place,describe for graph
plt.plot([0, 1], [0, 1],'r--')
#coordinates in a list ,r-- is red colored dash dash
plt.xlim([0, 1])
#scale of x axis and y axis
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()






#%%

#predicting using the Gradient_Boosting_Classifier
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier()
#model_GradientBoosting=DecisionTreeClassifier()

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)

#checking result
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)