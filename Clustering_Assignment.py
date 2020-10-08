# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:25:08 2020

@author: nitin
"""
#### CLUSTERING ON CRIME DATASET ####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

#loadcsv file of crime
crime = pd.read_csv("C:\\Users\\nitin\\Desktop\\Assignments\\Clustering\\crime_data.csv")

#top 10 records of data set crime
crime.head(10)
crime.describe()

crime.rename(columns={'Unnamed: 0':'States'},inplace= True)
crime.head()

#number which is near to 1 is good 
corr = crime.corr()
corr

plt.figure(figsize=(15,9)) # 15 is width of box and 9 is height of box
ax = sns.heatmap(corr,  annot=True)
plt.show()

from sklearn.cluster import KMeans

plt.figure(figsize=(18,9))
sns.scatterplot('Murder','Assault',data=crime,s=100)# s = size of dot
plt.show()

def ElbowMethod(data):
    wcss = list()
    for x in range(1,11):
        kmean = KMeans(x)
        y = kmean.fit(data)
        wcss.append(y.inertia_)
        
    return wcss

X = crime.iloc[:,[1,2]].values
elbow = ElbowMethod(X)
elbow

plt.figure(figsize=(18,9))
sns.lineplot(range(1,11),elbow)
plt.show()

# lets take 3 cluster for Murder, Assault and Rape
kmeans = KMeans(3)
yPred = kmeans.fit_predict(X)
yPred

# plotting scatter plot for the Murder and Assault
plt.figure(figsize=(18,9))
sns.scatterplot(X[yPred == 0,0],X[yPred==0,1],s=70)
sns.scatterplot(X[yPred == 1,0],X[yPred==1,1],s=70)
sns.scatterplot(X[yPred == 2,0],X[yPred==2,1],s=70)

sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=150,marker='<')

plt.xlabel('Murder',fontsize=20)
plt.ylabel('Assault',fontsize=20)
plt.show()


#### Hierarchial clustering for Airlines Dataset ####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

airlines = pd.read_excel("C:\\Users\\nitin\\Desktop\\Assignments\\Clustering\\EastWestAirlines.xlsx",sheet_name='data')
airlines.head(10)
airlines.info()
airlines.describe()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
#there is no need of column id so we will start from 1 inplace of 0 which is id
df_norm = norm_func(airlines.iloc[:,1:])
df_norm
type(df_norm)

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 



#p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)
# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
airlines['clust']=cluster_labels # creating a  new column and assigning it to new column 
airlines = airlines.iloc[:,[0,1,2,3,4,5,6,7,8]]
airlines.head()

# getting aggregate mean of each cluster
airlines.iloc[:,2:].groupby(airlines.clust).mean()


####KMeans Clustering on Airlines Dataset####
import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

# Kmeans on airlines Data set 
airlines1=airlines
airlines1.head()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airlines1.iloc[:,1:])
df_norm.head(10)  # Top 10 rows

###### scree plot or elbow curve ############
k = list(range(2,23))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines1['clust']=md # creating a  new column and assigning it to new column 

airlines1.iloc[:,1:11].groupby(airlines1.clust).mean()











































