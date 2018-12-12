# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv', header=0)
# dataset.replace('?', -99999, inplace=True) #-9999 biar outlier, gak masuk ke grafik 
dataset.drop("id",1)
mapping={'M':4, 'B':2}
print(dataset.shape)
dataset['diagnosis'] = dataset['diagnosis'].map(mapping)
X = dataset.iloc[:, 1:31].values # parameter yang mau di train
y = dataset.iloc[:, 1].values # target

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
# Predicting kNN Test set results
y_pred = classifier.predict(X_test)
#print results of knn 
from sklearn.metrics import classification_report
print('---kNN--')
print(classification_report(y_test, y_pred))
print('--------')


#predicitng svm test set results
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)
#print results of svm 
from sklearn.metrics import classification_report
print('---svm--')
print(classification_report(y_test, y_predict))
print('--------')

#predicting naive bayes test set results
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_predict = gaussian.predict(X_test)
#print results of naive bayes
from sklearn.metrics import classification_report
print('---NaiveBayes--')
print(classification_report(y_test, y_predict))
print('--------')

