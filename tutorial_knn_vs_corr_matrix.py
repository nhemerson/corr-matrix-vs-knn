import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import uuid
import random
import string
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# define random generator
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

# create random integers as metrics per customer
df = pd.DataFrame(np.random.randint(0,50,size=(100, 5)), columns=['Unique_Clicks','Unique_Views','Unique_Products','Total_Comments', 'Total_Reviews'])
df['Visits'] = [np.random.randint(0,200) for _ in range(len(df.index))]
df['Total_Spend'] = [np.random.randint(0,10000) for _ in range(len(df.index))]
df['Top_Department'] = [random.choice('ABCDEFG') for _ in range(len(df.index))]

# add unique identifiers for each customer row
df['uuid'] = [randomString() for _ in range(len(df.index))]

# reorder columns
df = df[['uuid','Visits','Total_Spend','Unique_Clicks','Unique_Views','Unique_Products','Total_Comments', 'Total_Reviews','Top_Department']]
df

# preprocess values for correlation Matrix and KNN

# one hot encode the top department
df_processed = pd.get_dummies(df)

# select numeric columns only
df_num = df_processed.select_dtypes(include=[np.number])

# move to numpy array for scaling
df_num_array = df_num.to_numpy()

# save the scaling function
scalar = StandardScaler()

# scale the numpy array
df_num_array = scalar.fit_transform(df_num_array)

# create datafrome from called numpy array
df_num_scaled = pd.DataFrame(df_num_array)

# bring in uuid's and add to scaled numeric data for correlation matrix
df_id = df.iloc[:,0]
df = pd.concat([df_id.reset_index(drop=True),df_num_scaled.reset_index(drop=True)], axis =1)
df_melt = pd.melt(df, id_vars=['uuid'])
df_melt.rename(columns = {'uuid':'foo'}, inplace = True)
df_corr_pivot = df_melt.pivot(index = 'variable', columns = 'foo', values = 'value' )

# bring in uuid's and add to scaled numeric data for KNN model
df_num_scaled.columns = df_num.columns
df_id = df.iloc[:,0]
df = pd.concat([df_id.reset_index(drop=True),df_num_scaled.reset_index(drop=True)], axis =1)
df_melt = pd.melt(df, id_vars=['uuid'])
df_melt.rename(columns = {'uuid':'foo'}, inplace = True)
df_knn_pivot = df_melt.pivot(index = 'foo', columns = 'variable', values = 'value' ).fillna(0)

###################################################
#### a correlation table is not very reasonable
###################################################

# get pearson correlation coefficients
df_corr = df_corr_pivot.corr(method = 'pearson')

# reset the index to move correlation matrix to a table
df_corr = df_corr.reset_index()

# rename columns for correct table 
df_corr.rename(columns = {'foo' : 'uuid'}, inplace = True)

# melt matrix into table  
df_corr = pd.melt(df_corr, id_vars=['uuid'])

# rename melted columns
df_corr.columns = ['uuid', 'uuid2', 'Correlation']

# turn table into dataframe
df_corr = pd.DataFrame(df_corr)

df_corr.loc[df_corr.uuid == df_corr.uuid2, 'id_match'] = 'Match' 
df_corr = df_corr[df_corr.id_match != 'Match'].drop('id_match', axis = 1)
df_corr = pd.DataFrame(df_corr)

# select top id for cohort
corr_id = df_corr['uuid2'].iloc[0]

df_corr = df_corr[df_corr.uuid == corr_id].sort_values('Correlation', ascending = False).head()
df_corr

###################################################
#### k nearest neighbor makes way more sense
###################################################

# knn needs a csr matrix to calculate
df_matrix = csr_matrix(df_knn_pivot.values)

# fit the model via the sklearn API
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
model_knn.fit(df_matrix)

# select top index name to get a cohort
knn_id = df_knn_pivot.index[0]

query_index = df_knn_pivot.loc[knn_id].values.reshape(1,-1)
distances, indices = model_knn.kneighbors(query_index, n_neighbors=11)
arrayKNN = [indices,distances]

#get array of cohort to pass on
cohort_list = df_knn_pivot.index[indices.flatten()[1:]].values.tolist()
cohort_array = [knn_id,cohort_list]
cohort_array
