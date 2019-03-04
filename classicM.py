from DataProcess import DataProcess
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
# from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)

# get and process data
data = DataProcess()
x_train, y_train, x_test, y_test = data.return_processed_cicids_data_multiclass()

# reshape input to be [samples, timesteps, features]
# x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
# x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
# x_test_21 = x_test_21.reshape(x_test_21.shape[0], 1, x_test_21.shape[1])

# multiclass
# y_train.ravel()=np_utils.to_categorical(y_train.ravel())
# y_test.ravel()=np_utils.to_categorical(y_test.ravel())
# y_test.ravel()_21=np_utils.to_categorical(y_test.ravel()_21)

model = LogisticRegression()
model.fit(x_train, y_train.ravel())


model = LogisticRegression()
model.fit(x_train, y_train.ravel())


# make predictions
# make predictions
expected = y_test.ravel()
predicted = model.predict(x_test)
accuracy = accuracy_score(expected, predicted)
print("******** LogisticRegression ***********\n")
cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)




# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(x_train, y_train.ravel())
print("********  Naive Bayes ***********\n")

# print(model)
# make predictions
expected = y_test.ravel()
predicted = model.predict(x_test)

cm = metrics.confusion_matrix(expected, predicted)
print(expected.shape)
print(predicted.shape)
expected = np.array(expected)
predicted = np.array(predicted)
cm.stats()

# print(cm)
print(expected.shape)
print(predicted.shape)
cm.stats()




# # fit a k-nearest neighbor model to the data
# model = KNeighborsClassifier()
# model.fit(x_train, y_train.ravel())
# print(model)
# # make predictions
# expected = y_test.ravel()
# predicted = model.predict(x_test)
# # summarize the fit of the model

# cm = metrics.confusion_matrix(expected, predicted)
# print("******** k-nearest neighbor ***********\n")

# # print(cm)
# tpr = float(cm[0][0])/np.sum(cm[0])
# fpr = float(cm[1][1])/np.sum(cm[1])
# print("%.3f" %tpr)
# print("%.3f" %fpr)
# print("Accuracy")
# print("%.3f" %accuracy)
# print("fpr")
# print("%.3f" %fpr)
# print("tpr")
# print("%.3f" %tpr)




model = DecisionTreeClassifier()
model.fit(x_train, y_train.ravel())
print(model)
# make predictions
expected = y_test.ravel()
predicted = model.predict(x_test)
# summarize the fit of the model

cm = metrics.confusion_matrix(expected, predicted)
print("******** DecisionTree ***********\n")

print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)

print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)







model = AdaBoostClassifier(n_estimators=100)
model.fit(x_train, y_train.ravel())

# make predictions
expected = y_test.ravel()
predicted = model.predict(x_test)
# summarize the fit of the model

cm = metrics.confusion_matrix(expected, predicted)
print("******** AdaBoost ***********\n")

print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)





model = RandomForestClassifier(n_estimators=100)
model = model.fit(x_train, y_train.ravel())

# make predictions
expected = y_test.ravel()
predicted = model.predict(x_test)
# summarize the fit of the model

cm = metrics.confusion_matrix(expected, predicted)
print("******** RandomForest ***********\n")

print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)
