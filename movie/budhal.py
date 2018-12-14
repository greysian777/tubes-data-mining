import pandas as pd 
import numpy as np 


movies_df = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
ratings_df = pd.read_csv('./data/ratings_small.csv', low_memory=False)

print(movies_df.shape)


movies_df.drop(movies_df.index[19730],inplace=True)
movies_df.drop(movies_df.index[29502],inplace=True)
movies_df.drop(movies_df.index[35585],inplace=True)

movies_df.id = movies_df.id.astype(np.int64)

print(ratings_df.head())

 #gabungin rating biar ada judulnya
ratings_df = pd.merge(ratings_df,movies_df[['title','id']],left_on='movieId',right_on='id')
ratings_df.drop(['timestamp','id'], axis = 1, inplace = True)

ratings_count = ratings_df.groupby(by="title")['rating'].count().reset_index().rename(columns={'rating':'totalRatings'})[['title','totalRatings']]

ratings_total = pd.merge(ratings_df,ratings_count,on='title', how='left')

print(ratings_count.shape[0])
print(ratings_count.sample(5))
print(ratings_total.shape)
print(ratings_total.head())
print(ratings_count['totalRatings'].describe())

votes_count_threshold = 20
ratings_top = ratings_total.query('totalRatings > @votes_count_threshold')
#mengurangi film yang vote nya kurang dari 20%

#cari duplikat 
if not ratings_top[ratings_top.duplicated(['userId','title'])].empty:
    ratings_top = ratings_top.drop_duplicates(['userId','title'])

print(ratings_top.shape)

print(list(ratings_top)) #judul column yang digunakan 

#reshape pake pivot 
''''
Reshape data (produce a “pivot” table) based on column values. Uses unique values from specified index / columns to form axes of the resulting DataFrame. This function does not support data aggregation, multiple values will result in a MultiIndex in the columns. See the User Guide for more on reshaping.
'''
df_for_knn = ratings_top.pivot(index='title', columns='userId', values='rating').fillna(0)

from scipy.sparse import csr_matrix
df_for_knn_sparse = csr_matrix(df_for_knn)

#recommendation using knn 
from sklearn.neighbors import NearestNeighbors

classifier = NearestNeighbors(metric='cosine', algorithm='brute')

print(classifier.fit(df_for_knn_sparse))



query_index = np.random.choice(df_for_knn.shape[0])
distances, indices = classifier.kneighbors(df_for_knn.loc['Batman Returns'].values.reshape(1,-1),n_neighbors=6)
distances, indices = classifier.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)

for i in range(0,len(distances.flatten())):
    if i==0:
        print("Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))

def encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1


from mlxtend.frequent_patterns import apriori, association_rules

df_for_ar = df_for_knn.T.applymap(encode_units)


frequent_itemsets = apriori(df_for_ar, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
query_index = df_for_knn.index.get_loc('Batman Returns')
distances, indices = classifier.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0,len(distances.flatten())):
    if i==0:
        print("KNN Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))

