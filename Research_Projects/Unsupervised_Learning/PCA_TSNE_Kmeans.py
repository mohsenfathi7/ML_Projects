import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt 
from matplotlib import pyplot
import random
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import seaborn as sns
from sklearn import decomposition
import seaborn as sn
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from collections import Counter
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import csv
from sklearn.feature_selection import SequentialFeatureSelector


with open('features.csv', newline='') as csvfile:
    x = list(csv.reader(csvfile))
X=np.array(x)
with open('HPWL.csv', newline='') as csvfile:
    h = list(csv.reader(csvfile))
hpwl=[]
for i in range(3618):
    hpwl=hpwl + h[i]   
    HPWL=np.array(hpwl)
HPWL=HPWL.astype(np.float32)
scaler = preprocessing.StandardScaler().fit(X)
# print(normalized_arr)
xn = scaler.transform(X)






#### PCA

pca=decomposition.PCA()
pca.n_components=2
pca_Data=pca.fit_transform(xn)


#creating Label
l=[]
for i in range(1,13):
    for j in range (0,31):
        l.append(i)
print(l)  
L=np.array(l)



pca_data=np.vstack((pca_Data.T,L)).T
print(pca_data.shape)


pca_df=pd.DataFrame(data=pca_data, columns=("1st","2nd","label")


plt.figure(figsize=(60,15))
rcParams['figure.figsize'] = 30,8
sn.FacetGrid(pca_df, hue="label", size=8).map(plt.scatter, "1st", "2nd").add_legend()
plt.savefig("pca.png")



### TSNA
pca_data=np.vstack((tsne_features.T,L)).T
print(pca_data.shape)
pca_df=pd.DataFrame(data=pca_data, columns=("1st","2nd","label"))



plt.figure(figsize=(3, 3))
sn.FacetGrid(pca_df, hue="label", size=8).map(plt.scatter, "1st", "2nd").add_legend()
plt.show()

### Kmeans

kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans4 = KMeans(n_clusters=4, random_state=0).fit(X)
kmeans5 = KMeans(n_clusters=5, random_state=0).fit(X)
kmeans6 = KMeans(n_clusters=6, random_state=0).fit(X)
kmeans7 = KMeans(n_clusters=7, random_state=0).fit(X)
kmeans8 = KMeans(n_clusters=8, random_state=0).fit(X)
kmeans9 = KMeans(n_clusters=9, random_state=0).fit(X)
kmeans10 = KMeans(n_clusters=10, random_state=0).fit(X)
kmeans11 = KMeans(n_clusters=11, random_state=0).fit(X)
kmeans12 = KMeans(n_clusters=12, random_state=0).fit(X)
kmeans13 = KMeans(n_clusters=13, random_state=0).fit(X)
kmeans14 = KMeans(n_clusters=14, random_state=0).fit(X)
kmeans15 = KMeans(n_clusters=15, random_state=0).fit(X)




loss=[kmeans2 .inertia_,kmeans3 .inertia_,kmeans4 .inertia_,kmeans5 .inertia_,kmeans6 .inertia_,kmeans7 .inertia_,kmeans8 .inertia_,kmeans9 .inertia_,kmeans10.inertia_,kmeans11.inertia_,kmeans12.inertia_,kmeans13.inertia_,kmeans14.inertia_,kmeans15.inertia_]



plt.plot(t, loss, color='black');
plt.xlabel('number of cluster')
plt.ylabel('loss')




label=[kmeans2 .labels_,kmeans3 .labels_,kmeans4 .labels_,kmeans5 .labels_,kmeans6 .labels_,kmeans7 .labels_,kmeans8 .labels_,kmeans9 .labels_,kmeans10.labels_,kmeans11.labels_,kmeans12.labels_,kmeans13.labels_,kmeans14.labels_,kmeans15.labels_]



sil=[]
for i in range(14):
    sil.append(silhouette_score(X, label[i], metric = 'euclidean'))
	
	

	
plt.plot(t, sil, color='black');
plt.xlabel('number of cluster')
plt.ylabel('silhouette score')


for j in range (0,7):
    for i in range (0,372):
        t=0
        if(label[5][i]==j):
            t=(i/31)
            t=int(t)
#             print(i)
            groups[j,t]=groups[j,t]+1
			
			
			
			
			
			
c=['FPGA1','FPGA2','FPGA3','FPGA4','FPGA5','FPGA6','FPGA7','FPGA8','FPGA9','FPGA10','FPGA11','FPGA12']
r=['CLUSTER1','CLUSTER2','CLUSTER3','CLUSTER4','CLUSTER5','CLUSTER6','CLUSTER7']



df = pd.DataFrame(groups,columns=c)
m=sn.heatmap(df, annot=True, fmt='g')
m.savefig('svm_conf.png', dpi=400)

#####Statistical analysis
df = pd.DataFrame(X,columns=name)
plt.figure(figsize=(30,30))
# covMatrix = pd.DataFrame.cov(df)


stats=pd.DataFrame()
stats["mean"]=df.mean()
stats["Std.Dev"]=df.std()
stats["Var"]=df.var()
stats["CV"]=df.std()/df.mean()
print(stats)



