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






# we create our test set with the following criteria: using same number of samples from phase 2 and 3 for all benchmarks, so we don't use python random


phase3index=[5, 20, 39, 52, 68, 88, 103, 119, 136, 156, 170, 186, 196, 206, 216, 227, 236, 248, 256, 266, 279, 286, 298, 308, 316, 326, 343, 366, 377, 394, 413, 433, 450, 465, 484, 497, 507, 517, 528, 537, 549, 556, 568, 576, 589, 597, 609, 616, 627, 636, 648, 656, 668, 686, 703, 722, 736, 752, 768, 790, 807, 819, 830, 841, 850, 859, 869, 880, 892, 900, 910, 920, 929, 939, 952, 961, 976, 995, 1012, 1030, 1046, 1061, 1080, 1097, 1112, 1130, 1146, 1164, 1179, 1190, 1200, 1209, 1219, 1230, 1241, 1249, 1258, 1268, 1279, 1292, 1298, 1313, 1326, 1340, 1349, 1364, 1376, 1385, 1395, 1408, 1419, 1436, 1449, 1458, 1469, 1478, 1489, 1499, 1508, 1520, 1526, 1537, 1547, 1557, 1569, 1576, 1591, 1609, 1620, 1642, 1654, 1675, 1688, 1708, 1726, 1744, 1756, 1774, 1786, 1794, 1804, 1817, 1826, 1837, 1845, 1854, 1864, 1874, 1885, 1894, 1905, 1914, 1924, 1935, 1955, 1968, 1984, 2002, 2020, 2032, 2050, 2064, 2081, 2087, 2098, 2107, 2117, 2130, 2140, 2147, 2157, 2169, 2181, 2188, 2197, 2208, 2220, 2227, 2239, 2256, 2275, 2289, 2307, 2323, 2339, 2357, 2375, 2386, 2393, 2404, 2414, 2424, 2434, 2444, 2454, 2463, 2474, 2486, 2495, 2503, 2513, 2526, 2533, 2549, 2560, 2575, 2585, 2594, 2604, 2616, 2625, 2634, 2644, 2654, 2665, 2675, 2687, 2697, 2705, 2716, 2725, 2736, 2744, 2755, 2766, 2788, 2802, 2815, 2837, 2849, 2866, 2886, 2900, 2915, 2924, 2933, 2944, 2954, 2964, 2974, 2984, 2994, 3005, 3014, 3023, 3034, 3046, 3055, 3063, 3073, 3088, 3099, 3108, 3126, 3140, 3160, 3171, 3190, 3207, 3218, 3229, 3244, 3254, 3265, 3275, 3284, 3294, 3305, 3315, 3324, 3334, 3347, 3354, 3366, 3377, 3385, 3394, 3404, 3421, 3432, 3449, 3459, 3468, 3478, 3489, 3499, 3509, 3518, 3528, 3538, 3548, 3558, 3568, 3579, 3588, 3598, 3610]
phase2index=[0, 17, 34, 51, 69, 85, 102, 121, 137, 154, 171, 188, 197, 208, 217, 226, 237, 246, 259, 267, 276, 288, 296, 306, 317, 328, 344, 360, 378, 396, 411, 428, 445, 463, 479, 496, 506, 516, 526, 536, 546, 558, 566, 578, 586, 596, 606, 619, 626, 639, 646, 658, 666, 683, 700, 717, 734, 751, 769, 785, 802, 820, 829, 839, 851, 861, 870, 879, 889, 899, 909, 919, 931, 940, 949, 959, 977, 993, 1010, 1027, 1044, 1062, 1078, 1095, 1113, 1129, 1144, 1161, 1178, 1189, 1198, 1208, 1218, 1228, 1238, 1248, 1261, 1270, 1278, 1289, 1299, 1312, 1328, 1338, 1348, 1362, 1378, 1382, 1396, 1409, 1418, 1432, 1445, 1459, 1468, 1480, 1488, 1498, 1511, 1517, 1527, 1536, 1546, 1556, 1566, 1577, 1586, 1603, 1621, 1637, 1655, 1672, 1689, 1705, 1723, 1739, 1757, 1773, 1784, 1795, 1805, 1814, 1824, 1834, 1844, 1855, 1865, 1876, 1884, 1895, 1904, 1915, 1925, 1934, 1951, 1969, 1985, 2001, 2018, 2031, 2045, 2061, 2077, 2088, 2097, 2108, 2119, 2127, 2137, 2148, 2158, 2167, 2178, 2187, 2199, 2207, 2217, 2228, 2237, 2254, 2271, 2288, 2305, 2322, 2340, 2356, 2373, 2383, 2394, 2403, 2415, 2423, 2433, 2443, 2453, 2464, 2473, 2483, 2493, 2504, 2514, 2523, 2535, 2544, 2559, 2574, 2584, 2595, 2605, 2614, 2624, 2635, 2645, 2655, 2664, 2674, 2684, 2695, 2704, 2715, 2726, 2735, 2745, 2754, 2764, 2781, 2799, 2816, 2833, 2850, 2867, 2883, 2901, 2913, 2923, 2936, 2943, 2953, 2963, 2973, 2983, 2993, 3003, 3013, 3024, 3033, 3043, 3053, 3064, 3075, 3083, 3098, 3109, 3123, 3139, 3156, 3172, 3187, 3203, 3217, 3230, 3245, 3255, 3264, 3274, 3285, 3296, 3304, 3314, 3325, 3335, 3344, 3355, 3364, 3374, 3384, 3395, 3405, 3423, 3431, 3448, 3458, 3470, 3479, 3488, 3498, 3508, 3519, 3529, 3539, 3549, 3559, 3569, 3578, 3589, 3599, 3608]
testindex=[]





#train and test set

for i in range(0,297):
    testindex.append(phase2index[i])
    testindex.append(phase3index[i])
    trainindex=[]
for i in range(3618):
    if i not in  testindex:
        trainindex.append(i)
		
		
		
		
		
		
		
#loading features and labels

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




#standard features
scaler = preprocessing.StandardScaler().fit(X)
# print(normalized_arr)
Xn = scaler.transform(X)




#train and test set
X_train=Xn[trainindex,:]
y_train=HPWL[trainindex]
X_test=Xn[testindex,:]
y_test=HPWL[testindex]
X_train=X_train.astype(np.float32)
y_train=y_train.astype(np.float32)
X_test=X_test.astype(np.float32)
y_test=y_test.astype(np.float32)




#MLP MODEL
start=datetime.now()
model = MLPRegressor(solver='lbfgs',random_state=1, max_iter=500,hidden_layer_sizes=[50,50]).fit(X_train, y_train)
print('train time:',datetime.now()-start)
start=datetime.now()
predict1=model.predict(X_test)
print('inference time:',datetime.now()-start)
predict=model.predict(X_train)






#Performance Metrics

print('train accuracy:',model.score(X_train, y_train))
print('test accuracy:',model.score(X_test, y_test))
print('train MAE:',mean_absolute_error(predict, y_train))
print('test MAE:',mean_absolute_error(predict1, y_test))
print('train MSE:',mean_squared_error(predict, y_train))
print('test MSE:',mean_squared_error(predict1, y_test))
print('train RMSE:',sqrt(mean_squared_error(predict, y_train)))
print('test RMSE:',sqrt(mean_squared_error(predict1, y_test)))
tmp=0
for i in range(0,len(predict)):
    tmp=tmp+(abs(predict[i]-y_train[i])/y_train[i])
print('train MAPE:',(100*tmp)/len(predict))

tmp=0
for i in range(0,len(predict1)):
    tmp=tmp+ (abs(predict1[i]-y_test[i])/y_test[i])
print('test MPAE:',(100*tmp)/len(predict1))




num=[]
for i in range(0,len(predict1)):
    num.append(i)
one, two = zip(*sorted(zip(y_test, predict1)))
one=np.array(one)
two=np.array(two)

plt.figure(figsize=(20,10))
plt.scatter(num, two,color='red',s=10,Label='Predicted HPWL')
plt.plot(num, one,color='blue', Label='Actual HPWL')
plt.xticks(rotation=90)
plt.xlabel('S',fontsize=20)
plt.ylabel('HPWL',fontsize=20)
plt.xlabel('Sample #',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend()    
plt.savefig('asc1.jpg')











#LOSS VS Teain size

rf =  MLPRegressor(solver='lbfgs',random_state=1, max_iter=500,hidden_layer_sizes=[50,50]).fit(X_train, y_train)
rnd=random.sample(range(3024), 3024)


for i in range(0,29):
    sfs = SequentialFeatureSelector(model, n_features_to_select=29-i)
    sfs.fit(Xn, HPWL)
    m=sfs.transform(Xn[trainindex,:])
    model=rf.fit(m, HPWL[trainindex])
    m2=sfs.transform(Xn[testindex,:])
    predict=model.predict(m2)
    print("features {}: {}".format( i, mean_absolute_error(HPWL[testindex],predict)))


error=[]
error1=[]
xtrain=[]
ytrain=[]


for i in range(0,302):
    print(i)
    xtrain=xtrain+(X_train[rnd[i:(i+1)*10],:]).tolist()
    ytrain=ytrain+(y_train[rnd[i:(i+1)*10]]).tolist()
    xtrain1=np.array(xtrain)
    ytrain1=np.array(ytrain)
    model=rf.fit(xtrain1, ytrain1)
    predict=model.predict(X_test)
    predict1=model.predict(xtrain1)

    error.append((mean_squared_error(y_test,predict)))
    error1.append((mean_squared_error(ytrain,predict1)))

	
	
	
	
	
	
number=[]
for i in range(302):
    number.append(i*10)
plt.figure(figsize=(20,10))
plt.ylim(0,0.001)
plt.xlabel('Training Samples',fontsize=20)
plt.ylabel('Loss (MSE)',fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.plot(number,error, label='Test');
plt.plot(number,error1, label='Train');
plt.legend()
plt.savefig('mlploss.png')