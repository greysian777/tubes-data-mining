import numpy as np 
from scipy import stats

a = [80,
81.66666667,
78.33333333,
76.66666667,
76.66666667,
76.66666667,
76.66666667]

zscore = stats.zscore(a)
print(zscore)


import numpy as np
from sklearn.preprocessing import minmax_scale

# your function
def normalize_list(list_normal):
    max_value = max(list_normal)
    min_value = min(list_normal)
    for i in range(len(list_normal)):
        list_normal[i] = (list_normal[i] - min_value) / (max_value - min_value)
    return list_normal

#Scikit learn version 
def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy, feature_range=(70, 90))
    return normalized_list

test_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
test_array_numpy = np.array(zscore)

print(normalize_list(zscore))
print(normalize_list_numpy(test_array_numpy))