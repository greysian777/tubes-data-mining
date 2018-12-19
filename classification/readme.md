# Classification 
menentukan kanker benign(pasif) / malign(aktif) terhadap umur dan gaji 
menggunakan f1 score untuk menentukan "nilai" klasifikasi
- k nearest neighbor (knn.py)   
Test Accuracy:  0.9736842105263158
Train Accuracy:  0.9736263736263736
  ![](https://i.imgur.com/iI4iCHj.png)   
  ![](https://i.imgur.com/w3REUNf.png)
- SVM (new_svm.py)    
Test Accuracy:  0.9912280701754386
Train Accuracy:  0.9692307692307692
  ![](https://i.imgur.com/mFV3YfG.png)
  ![](https://i.imgur.com/mJdb88S.png)
- Naive Bayes (naivebayes.py)   
Test Accuracy:  0.9385964912280702   
Train Accuracy:  0.9318681318681319   
  ![](https://i.imgur.com/XslH2la.png)
  ![](https://i.imgur.com/DjlCjXs.png)

--------

## Penilaian classification_report(y_test, y_pred) (cross validation)
dimana `y_test` adalah 20% dari dataset, dipilih dengan parameter `random_state=1`, dan `y_pred` adalah `classifier.predict(X_test)`
![](https://i.imgur.com/Zz25GJF.png)

### kNN
|     Type     	| Precision 	| recall 	| f1-score 	| support 	|
|:------------:	|-----------	|--------	|----------	|---------	|
| 2            	| 1.00      	| 0.96   	| 0.98     	| 70      	|
| 4            	| 0.94      	| 1.00   	| 0.97     	| 44      	|
| micro avg    	| 0.97      	| 0.97   	| 0.97     	| 114     	|
| macro avg    	| 0.97      	| 0.98   	| 0.97     	| 114     	|
| weighted avg 	| 0.98      	| 0.97   	| 0.97     	| 114     	|

--------
### SVM
|     Type     	| Precision 	| recall 	| f1-score 	| support 	|
|:------------:	|-----------	|--------	|----------	|---------	|
| 2            	| 1.00      	| 0.93   	| 0.96     	| 70      	|
| 4            	| 0.90      	| 1.00   	| 0.95     	| 44      	|
| micro avg    	| 0.96      	| 0.96   	| 0.97     	| 114     	|
| macro avg    	| 0.97      	| 0.96   	| 0.95     	| 114     	|
| weighted avg 	| 0.96      	| 0.96   	| 0.96     	| 114     	|

--------
### NaiveBayes
|     Type     	| Precision 	| recall 	| f1-score 	| support 	|
|:------------:	|-----------	|--------	|----------	|---------	|
| 2            	| 0.98      	| 0.91  	| 0.95     	| 70      	|
| 4            	| 0.88      	| 0.98   	| 0.92     	| 44      	|
| micro avg    	| 0.94      	| 0.94   	| 0.94     	| 114     	|
| macro avg    	| 0.93      	| 0.95   	| 0.94     	| 114     	|
| weighted avg 	| 0.94      	| 0.94   	| 0.94     	| 114     	|

--------

## Dataset 
- [UCI Breast Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))

## Column Head
"id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"

## Kernel menggunakan 'RBF' 
Radial basis filter
