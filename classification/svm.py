# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('data.csv', header=0)
dataset.drop("id",1)
mapping={'M':4, 'B':2}
print(dataset.shape)
dataset['diagnosis'] = dataset['diagnosis'].map(mapping)
X = dataset.iloc[:, 1:31].values # parameter yang mau di train
y = dataset.iloc[:, 1].values # target

print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(dataset.shape))
dataset.info()

print(dataset.head(3))


#visualizing data 
features_mean = list(dataset.columns[0:20])
print(features_mean)
# plt.figure(figsize=(32,32))
sns.heatmap(dataset[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()




print(dataset.columns)

sns.pairplot(dataset, hue='diagnosis', vars = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"])
plt.show()

#Print benign and malign cancer 

sns.countplot(dataset['diagnosis'], label = "Hitung")
plt.show()
X = dataset.drop(['diagnosis'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
print(X.head())
y = dataset['diagnosis']
print('\n')
print(y.head())


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# check the test and train data 
print ('The size of our training "X" (input features) is', X_train.shape)
print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)

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


from sklearn.svm import SVC
svc_model = SVC()

svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)
score = svc_model.score(X_test, y_test)
print("Test Accuracy: ", score)
score = svc_model.score(X_train, y_train)
print("Train Accuracy: ", score)


# visuaslisai preidksi menggunakan svm dengan confusion matrix 
from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[2,4]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'], columns=['predicted_cancer','predicted_healthy'])
print(confusion)
sns.heatmap(confusion, annot=True)
plt.show()

print(classification_report(y_test, y_predict))

