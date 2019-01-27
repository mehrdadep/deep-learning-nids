from DataProccess import DataProccess
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
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


# get and proccess data
data = DataProccess()
x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_proccessed_data_multiclass()

# reshape input to be [samples, timesteps, features]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
x_test_21 = x_test_21.reshape(x_test_21.shape[0], 1, x_test_21.shape[1])

# multiclass
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
y_test_21=np_utils.to_categorical(y_test_21)

model = LogisticRegression()
model.fit(x_train, y_train)


model = LogisticRegression()
model.fit(x_train, y_train)


# make predictions
expected = y_test
predicted = model.predict(x_test)


print("***************************************************************")



# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(x_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(x_test)

cm = ConfusionMatrix(expected, predicted)
print(expected.shape)
print(predicted.shape)
expected = np.array(expected)
predicted = np.array(predicted)
cm.print_stats()

np.savetxt('expected.txt', expected, fmt='%01d')
np.savetxt('predicted.txt',predicted , fmt='%01d')

print(cm)
print(expected.shape)
print(predicted.shape)
cm.stats()
print("***************************************************************")



# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()
model.fit(x_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)
print("***************************************************************")



model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)
print("***************************************************************")






model = AdaBoostClassifier(n_estimators=100)
model.fit(x_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)
print("***************************************************************")




model = RandomForestClassifier(n_estimators=100)
model = clf.fit(x_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(x_test)
# summarize the fit of the model

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)
print("***************************************************************")