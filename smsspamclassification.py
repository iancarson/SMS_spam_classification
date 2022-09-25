# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:10:08 2019

@author: jdk450
"""

import os
import pandas  as pd
from sklearn import model_selection, linear_model, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
os.getcwd()
matplotlib.use("TkAgg")
sns.u
#os.chdir('Python') # this depends on where you placed the dataset

# load the dataset
data = pd.read_csv('SMSSpamCollection.csv')

# create  dataframes using texts and labels
texts = data.iloc[:, 1]
labels = data.iloc[:, 0]

#take a look
texts
labels

# split the dataset into training and validation datasets 
X_train, X_test, y_train, y_test = train_test_split(texts,labels, test_size=0.20,random_state=0)#80:20
vectorizer = TfidfVectorizer()
X_train= vectorizer.fit_transform(X_train)
X_test= vectorizer.transform(X_test)
logReg = LogisticRegression()
logReg.fit(X_train, y_train)

#predict
y_predict = logReg.predict(X_test)

#get confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_predict)

#show confusion matrix
print(cnf_matrix)

#Calculate performance measures:
print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

#if you don't include pos_label='sham' you get this error:
#ValueError: pos_label=1 is not a valid label: array(['ham', 'spam'], dtype='<U4')
print("Precision:", metrics.precision_score(y_test, y_predict, pos_label='spam'))
print("Recall:",metrics.recall_score(y_test, y_predict, pos_label = 'spam'))
print(metrics.f1_score(y_test, y_predict, pos_label='spam'))

#from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve ,recall_score,classification_report
print(metrics.classification_report(y_test, y_predict))
#Create decision tree classifier object.
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("For decision Tree: ", cm )
print(classification_report(y_test,y_pred))
print("Decision Tree Accuracy Score: ",metrics.accuracy_score(y_test, y_pred))
#Display the heatmap
sns.heatmap(cm)
plt.show()

