import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from services.process import Processor


class Classic:

    @classmethod
    def run(cls, run_type, dataset):
        x_train, y_train, x_test, y_test = Processor.get_data(
            run_type,
            dataset,
        )

        y_train = y_train.ravel()
        y_test = y_test.ravel()

        print("\r\n******** LogisticRegression ***********\r\n")
        model = LogisticRegression()
        model.fit(x_train, y_train)
        # make predictions
        expected = y_test
        predicted = model.predict(x_test)
        accuracy = accuracy_score(expected, predicted)
        if run_type == 0:
            recall = recall_score(expected, predicted, average="binary")
            precision = precision_score(expected, predicted, average="binary")
            f1 = f1_score(expected, predicted, average="binary")
            print("precision")
            print("%.3f" % precision)
            print("recall")
            print("%.3f" % recall)
            print("f-score")
            print("%.3f" % f1)

        cm = metrics.confusion_matrix(expected, predicted)
        print(cm)
        tpr = float(cm[0][0]) / np.sum(cm[0])
        fpr = float(cm[1][1]) / np.sum(cm[1])
        print("%.3f" % tpr)
        print("%.3f" % fpr)
        print("Accuracy")
        print("%.3f" % accuracy)
        print("fpr")
        print("%.3f" % fpr)
        print("tpr")
        print("%.3f" % tpr)

        # fit a Naive Bayes model to the data
        print("\r\n******** Naive Bayes ***********\r\n")
        model = GaussianNB()
        model.fit(x_train, y_train)
        # make predictions
        expected = y_test
        predicted = model.predict(x_test)
        accuracy = accuracy_score(expected, predicted)
        if run_type == 0:
            recall = recall_score(expected, predicted, average="binary")
            precision = precision_score(expected, predicted, average="binary")
            f1 = f1_score(expected, predicted, average="binary")
            print("precision")
            print("%.3f" % precision)
            print("recall")
            print("%.3f" % recall)
            print("f-score")
            print("%.3f" % f1)

        cm = metrics.confusion_matrix(expected, predicted)
        print(cm)
        tpr = float(cm[0][0]) / np.sum(cm[0])
        fpr = float(cm[1][1]) / np.sum(cm[1])
        print("%.3f" % tpr)
        print("%.3f" % fpr)
        print("Accuracy")
        print("%.3f" % accuracy)
        print("fpr")
        print("%.3f" % fpr)
        print("tpr")
        print("%.3f" % tpr)

        print("\r\n******** Decision Tree ***********\r\n")
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        # make predictions
        expected = y_test
        predicted = model.predict(x_test)
        # summarize the fit of the model
        accuracy = accuracy_score(expected, predicted)
        if run_type == 0:
            recall = recall_score(expected, predicted, average="binary")
            precision = precision_score(expected, predicted, average="binary")
            f1 = f1_score(expected, predicted, average="binary")
            print("precision")
            print("%.3f" % precision)
            print("recall")
            print("%.3f" % recall)
            print("f-score")
            print("%.3f" % f1)

        cm = metrics.confusion_matrix(expected, predicted)
        print(cm)
        tpr = float(cm[0][0]) / np.sum(cm[0])
        fpr = float(cm[1][1]) / np.sum(cm[1])
        print("%.3f" % tpr)
        print("%.3f" % fpr)
        print("Accuracy")
        print("%.3f" % accuracy)
        print("precision")
        print("fpr")
        print("%.3f" % fpr)
        print("tpr")
        print("%.3f" % tpr)

        print("\r\n******** Ada Boost ***********\r\n")
        model = AdaBoostClassifier(n_estimators=100)
        model.fit(x_train, y_train)
        # make predictions
        expected = y_test
        predicted = model.predict(x_test)
        # summarize the fit of the model
        accuracy = accuracy_score(expected, predicted)
        if run_type == 0:
            recall = recall_score(expected, predicted, average="binary")
            precision = precision_score(expected, predicted, average="binary")
            f1 = f1_score(expected, predicted, average="binary")
            print("precision")
            print("%.3f" % precision)
            print("recall")
            print("%.3f" % recall)
            print("f-score")
            print("%.3f" % f1)

        cm = metrics.confusion_matrix(expected, predicted)

        print(cm)

        tpr = float(cm[0][0]) / np.sum(cm[0])
        fpr = float(cm[1][1]) / np.sum(cm[1])
        print("%.3f" % tpr)
        print("%.3f" % fpr)
        print("Accuracy")
        print("%.3f" % accuracy)
        print("fpr")
        print("%.3f" % fpr)
        print("tpr")
        print("%.3f" % tpr)

        print("\r\n******** RandomForest ***********\r\n")
        model = RandomForestClassifier(n_estimators=100)
        model = model.fit(x_train, y_train)
        # make predictions
        expected = y_test
        predicted = model.predict(x_test)
        # summarize the fit of the model
        accuracy = accuracy_score(expected, predicted)
        if run_type == 0:
            recall = recall_score(expected, predicted, average="binary")
            precision = precision_score(expected, predicted, average="binary")
            f1 = f1_score(expected, predicted, average="binary")
            print("precision")
            print("%.3f" % precision)
            print("recall")
            print("%.3f" % recall)
            print("f-score")
            print("%.3f" % f1)

        cm = metrics.confusion_matrix(expected, predicted)
        print(cm)
        tpr = float(cm[0][0]) / np.sum(cm[0])
        fpr = float(cm[1][1]) / np.sum(cm[1])
        print("%.3f" % tpr)
        print("%.3f" % fpr)
        print("Accuracy")
        print("%.3f" % accuracy)
        print("precision")
        print("fpr")
        print("%.3f" % fpr)
        print("tpr")
        print("%.3f" % tpr)
