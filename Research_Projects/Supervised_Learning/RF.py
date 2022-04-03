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
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
import numpy
import csv


############
############
############
############
############ Features, Dataset, Train and Test set preparation are Not Shared Publicly, for more information please contact mfathiri @uoguelph.ca
############ Resulats are published at https:##ieeexplore.ieee.org#document#9664954
############ Thesis: https:##atrium.lib.uoguelph.ca#xmlui#handle#10214#26602
############
############
############
############

with open('features.csv', newline='') as csvfile:
    x = list(csv.reader(csvfile))
X=np.array(x)
with open('HPWL.csv', newline='') as csvfile:
    h = list(csv.reader(csvfile))
hpwl=[]
for i in range(3618):
    hpwl=hpwl + h[i]   
    HPWL=np.array(hpwl)
F=HPWL.astype(np.float32)



#standard features
scaler = preprocessing.StandardScaler().fit(X)
# print(normalized_arr)
Xn = scaler.transform(X)


X_train=X[trainindex,:]
y_train=F[trainindex]
X_test=X[testindex,:]
y_test=F[testindex]

###########Find training and tessting loss by incresing number of samples for initial model
########### Detects Under and over fitting


rf =  RandomForestRegressor()
rnd=random.sample(range(3024), 3024)

error=[]
error1=[]
xtrain=[]
ytrain=[]


for i in range(0,302):
    xtrain=xtrain+(X_train[rnd[i:(i+1)*10],:]).tolist()
    ytrain=ytrain+(y_train[rnd[i:(i+1)*10]]).tolist()
    xtrain1=np.array(xtrain)
    ytrain1=np.array(ytrain)
    model=rf.fit(xtrain1, ytrain1)
    predict=model.predict(X_test)
    predict1=model.predict(xtrain1)

    error.append((mean_squared_error(y_test,predict)))
    error1.append((mean_squared_error(ytrain,predict1)))
print(error)
print(error1)


#########Initial Model, train and test results

rf = RandomForestRegressor()

start=datetime.now()
model=rf.fit(X_train,y_train)
print(datetime.now()-start)
predict=model.predict(X_train)
start=datetime.now()
predict1=model.predict(X_test)
print(datetime.now()-start)

##########
##########Accuracy, MAE, MAPE, RMSE, MSE
##########


start=datetime.now()
model=rf.fit(X_train,y_train)
print(datetime.now()-start)
predict=model.predict(X_train)
start=datetime.now()
predict1=model.predict(X_test)
print(datetime.now()-start)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
print(mean_absolute_error(predict, y_train))
print(mean_absolute_error(predict1, y_test))
print(mean_squared_error(predict, y_train))
print(mean_squared_error(predict1, y_test))
print(sqrt(mean_squared_error(predict, y_train)))
print(sqrt(mean_squared_error(predict1, y_test)))
tmp=0
for i in range(0,len(predict)):
    tmp=tmp+(abs(predict[i]-y_train[i])#y_train[i])
print((100*tmp)#len(predict))

tmp=0
for i in range(0,len(predict1)):
    tmp=tmp+ (abs(predict1[i]-y_test[i])#y_test[i])
print((100*tmp)#len(predict1))




#########
#########Feature Selection
#########

for i in range(0,22):
    sfs = SequentialFeatureSelector(rf, n_features_to_select=22-i)
    sfs.fit(X, F)
    m=sfs.transform(X[trainindex,:])
    model=rf.fit(m, F[trainindex])
    m2=sfs.transform(X[allindex,:])
    predict=model.predict(m2)
    print("features {}: {}".format( i, mean_absolute_error(F[allindex],predict)))

	
sfs = SequentialFeatureSelector(rf, n_features_to_select=10)
sfs.fit(X, F)
sfs.get_support()


########
######## Hyper-parameters
########
######## Random search for narrowing down the serach range
param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [100, 200, 400, 600, 800, 1000]}
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train,y_train)
print(rf_random.best_params_)









####### Grid search for fine tuning the parameters

param_grid = {'bootstrap': [True],
 'max_depth': [ 10,14,17, 20, , None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [2, 4,6],
 'min_samples_split': [2, 5, 7,10],
 'n_estimators': [30,40,50]}


rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)


########
######## plots
########

plt.figure(figsize=(20,10))
plt.scatter(num, predict1,color='red',s=10,Label='Predicted FMAX')
plt.plot(num, y_test,color='blue', Label='Actual FMAX')
plt.xticks(rotation=90)
plt.xlabel('S',fontsize=20)
plt.ylabel('Frequency (MHz)',fontsize=20)
plt.xlabel('Sample #',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
for i in range(12):
    plt.axvline(x=kkk[i],color='black')
plt.legend()    
plt.savefig('model4pred1.jpg')



one, two = zip(*sorted(zip(y_test, predict1)))
one=np.array(one)
two=np.array(two)

plt.figure(figsize=(20,10))
plt.scatter(num, two,color='red',s=10,Label='Predicted Fmax')
plt.plot(num, one,color='blue', Label='Actual Fmax')
plt.xticks(rotation=90)
plt.xlabel('S',fontsize=30)
plt.ylabel('Frequency (MHz)',fontsize=30)
plt.xlabel('Sample #',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(fontsize=30)    
plt.savefig('model4pred2.jpg')