from sklearn.cluster import KMeans
from DataProcess import DataProcess
import numpy as np
from sklearn import metrics

data = DataProcess()
x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_processed_data_multiclass()


x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape((x.shape[0], -1))
x = np.divide(x, 255.)
# 2 clusters (binary)
n_clusters = 5
# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(x)
# Evaluate the K-Means clustering accuracy.
score = metrics.accuracy_score(y, y_pred_kmeans)
print(score)