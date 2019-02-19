---
title: Insurance Model
description: Identify the steps involved in an insurance prediction model. 
keywords: insurance, financial, bank, risk, risk management
---

This insurance model seeks to explain all the steps involved in the prediction process. 


***
## Data statistics
* Shape
* Peek
* Description
* Skew

## Transformation
* Correction of skew

## Data Interaction
* Correlation
* Scatter plot

## Data Visualization
* Box and density plots
* Grouping of one hot encoded attributes

## Data Preparation
* One hot encoding of categorical data
* Test-train split

## Evaluation, prediction, and analysis
* Linear Regression (Linear algo)
* Ridge Regression (Linear algo)
* LASSO Linear Regression (Linear algo)
* Elastic Net Regression (Linear algo)
* KNN (non-linear algo)
* CART (non-linear algo)
* SVM (Non-linear algo)
* Bagged Decision Trees (Bagging)
* Random Forest (Bagging)
* Extra Trees (Bagging)
* AdaBoost (Boosting)
* Stochastic Gradient Boosting (Boosting)
* MLP (Deep Learning)
* XGBoost

## Make Predictions
***
Learning: 
We need to predict the 'loss' based on the other attributes. Hence, this is a regression problem.


```python
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas #provides data structures to quickly analyze data
#Since this code runs on Kaggle server, data can be accessed directly in the 'input' folder
#Read the train dataset
dataset = pandas.read_csv("train.csv") 

#Read test dataset
dataset_test = pandas.read_csv("test.csv")
#Save the id's for submission file
ID = dataset_test['id']
#Drop unnecessary columns
dataset_test.drop('id',axis=1,inplace=True)

#Print all rows and columns. Dont hide any
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

#Display the first five rows to get a feel of the data
print(dataset.head(5))

#Learning : cat1 to cat116 contain alphabets
```

## Data statistics
* Shape


```python
# Size of the dataframe

print(dataset.shape)

# We can see that there are 188318 instances having 132 attributes

#Drop the first column 'id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]

#Learning : Data is loaded successfully as dimensions match the data description
```

## Data statistics
* Description


```python
# Statistical description

print(dataset.describe())

# Learning :
# No attribute in continuous columns is missing as count is 188318 for all, all rows can be used
# No negative values are present. Tests such as chi2 can be used
# Statistics not displayed for categorical data
```

## Data statistics
* Skew


```python
# Skewness of the distribution

print(dataset.skew())

# Values close to 0 show less ske
# loss shows the highest skew. Let us visualize it
```

## Data Visualization
* Box and density plots


```python
# We will visualize all the continuous attributes using Violin Plot - a combination of box and density plots

import numpy

#import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

#range of features considered
split = 116 

#number of features considered
size = 15

#create a dataframe with only continuous features
data=dataset.iloc[:,split:] 

#get the names of all the columns
cols=data.columns 

#Plot violin for all attributes in a 7x2 grid
n_cols = 2
n_rows = 7

for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=dataset, ax=ax[j])


#cont1 has many values close to 0.5
#cont2 has a pattern where there a several spikes at specific points
#cont5 has many values near 0.3
#cont14 has a distinct pattern. 0.22 and 0.82 have a lot of concentration
#loss distribution must be converted to normal
```

## Data Transformation
* Skew correction


```python
#log1p function applies log(1+x) to all elements of the column
dataset["loss"] = numpy.log1p(dataset["loss"])
#visualize the transformed column
sns.violinplot(data=dataset,y="loss")  
plt.show()

#Plot shows that skew is corrected to a large extent
```

## Data Interaction
* Correlation


```python
# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore categorical data

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

# Strong correlation is observed between the following pairs
# This represents an opportunity to reduce the feature set through transformations such as PCA
```

## Data Interaction
* Scatter plot


```python
# Scatter plot of only the highly correlated pairs
for v,i,j in s_corr_list:
    sns.pairplot(dataset, size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()

#cont11 and cont12 give an almost linear pattern...one must be removed
#cont1 and cont9 are highly correlated ...either of them could be safely removed 
#cont6 and cont10 show very good correlation too
```

## Data Visualization
* Categorical attributes


```python
# Count of each label in each category

#names of all the columns
cols = dataset.columns

#Plot count plot for all attributes in a 29x4 grid
n_cols = 4
n_rows = 29
for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))
    for j in range(n_cols):
        sns.countplot(x=cols[i*n_cols+j], data=dataset, ax=ax[j])

#cat1 to cat72 have only two labels A and B. In most of the cases, B has very few entries
#cat73 to cat 108 have more than two labels
#cat109 to cat116 have many labels
```

##Data Preparation
* One Hot Encoding of categorical data


```python
import pandas

#cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
#One-hot encoding converts an attribute to a binary vector

#Variable to hold the list of variables for an attribute in the train and test data
labels = []

for i in range(0,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))    

del dataset_test

#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)

#Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
del cats
del feature
del dataset
del encoded_cats
print(dataset_encoded.shape)
```

##Data Preparation
* Split into train and validation


```python
#get the number of rows and columns
r, c = dataset_encoded.shape

#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

#Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded

#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
del X
del Y

#All features
X_all = []

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae = []

#Scoring parameter
from sklearn.metrics import mean_absolute_error

#Add this version of X to the list 
n = "All"
#X_all.append([n, X_train,X_val,i_cols])
X_all.append([n, i_cols])
```

## Evaluation, prediction, and analysis
* Linear Regression (Linear algo)


```python
#Evaluation of various combinations of LinearRegression

#Import the library
from sklearn.linear_model import LinearRegression

#uncomment the below lines if you want to run the algo
##Set the base model
#model = LinearRegression(n_jobs=-1)
#algo = "LR"
#
##Accuracy of the model using all features
#for name,i_cols_list in X_all:
#    model.fit(X_train[:,i_cols_list],Y_train)
#    result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
#    mae.append(result)
#    print(name + " %s" % result)
#comb.append(algo)

#Result obtained after running the algo. Comment the below two lines if you want to run the algo
mae.append(1278)
comb.append("LR" )    

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#MAE achieved is 1278
```

## Evaluation, prediction, and analysis
* Ridge Regression (Linear algo)


```python
#Evaluation of various combinations of Ridge LinearRegression

#Import the library
from sklearn.linear_model import Ridge

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([])

for alpha in a_list:
    #Set the base model
    model = Ridge(alpha=alpha,random_state=seed)
    
    algo = "Ridge"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % alpha )

#Result obtained by running the algo for alpha=1.0    
if (len(a_list)==0):
    mae.append(1267.5)
    comb.append("Ridge" + " %s" % 1.0 )    
    
##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 1267 with alpha=1
```

## Evaluation, prediction, and analysis
* LASSO Linear Regression (Linear algo)


```python
#Evaluation of various combinations of Lasso LinearRegression

#Import the library
from sklearn.linear_model import Lasso

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([])

for alpha in a_list:
    #Set the base model
    model = Lasso(alpha=alpha,random_state=seed)
    
    algo = "Lasso"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % alpha )

#Result obtained by running the algo for alpha=0.001    
if (len(a_list)==0):
    mae.append(1262.5)
    comb.append("Lasso" + " %s" % 0.001 )
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#High computation time
#Best estimated performance is 1262.5 for alpha = 0.001
```

## Evaluation, prediction, and analysis
* Elastic Net Regression (Linear algo)


```python
#Evaluation of various combinations of ElasticNet LinearRegression

#Import the library
from sklearn.linear_model import ElasticNet

#Add the alpha value to the below list if you want to run the algo
a_list = numpy.array([])

for alpha in a_list:
    #Set the base model
    model = ElasticNet(alpha=alpha,random_state=seed)
    
    algo = "Elastic"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % alpha )

if (len(a_list)==0):
    mae.append(1260)
    comb.append("Elastic" + " %s" % 0.001 )
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#High computation time
#Best estimated performance is 1260 for alpha = 0.001
```

## Evaluation, prediction, and analysis
* KNN (non-linear algo)


```python
#Evaluation of various combinations of KNN

#Import the library
from sklearn.neighbors import KNeighborsRegressor

#Add the N value to the below list if you want to run the algo
n_list = numpy.array([])

for n_neighbors in n_list:
    #Set the base model
    model = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)
    
    algo = "KNN"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_neighbors )

if (len(n_list)==0):
    mae.append(1745)
    comb.append("KNN" + " %s" % 1 )
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Very high computation time
#Best estimated performance is 1745 for n=1
```

## Evaluation, prediction, and analysis
* CART (non-linear algo)


```python
#Evaluation of various combinations of CART

#Import the library
from sklearn.tree import DecisionTreeRegressor

#Add the max_depth value to the below list if you want to run the algo
d_list = numpy.array([])

for max_depth in d_list:
    #Set the base model
    model = DecisionTreeRegressor(max_depth=max_depth,random_state=seed)
    
    algo = "CART"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % max_depth )

if (len(a_list)==0):
    mae.append(1741)
    comb.append("CART" + " %s" % 5 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#High computation time
#Best estimated performance is 1741 for depth=5
```

## Evaluation, prediction, and analysis
* SVM (Non-linear algo)


```python
#Evaluation of various combinations of SVM

#Import the library
from sklearn.svm import SVR

#Add the C value to the below list if you want to run the algo
c_list = numpy.array([])

for C in c_list:
    #Set the base model
    model = SVR(C=C)
    
    algo = "SVM"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % C )

##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#very very high computation time
```

## Evaluation, prediction, and analysis
* Bagged Decision Trees (Bagging)


```python
#Evaluation of various combinations of Bagged Decision Trees

#Import the library
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([])

for n_estimators in n_list:
    #Set the base model
    model = BaggingRegressor(n_jobs=-1,n_estimators=n_estimators)
    
    algo = "Bag"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#very high computation time
```

## Evaluation, prediction, and analysis
* Random Forest (Bagging)


```python
#Evaluation of various combinations of RandomForest

#Import the library
from sklearn.ensemble import RandomForestRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([])

for n_estimators in n_list:
    #Set the base model
    model = RandomForestRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
    
    algo = "RF"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1213)
    comb.append("RF" + " %s" % 50 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 1213 when the number of estimators is 50
```

## Evaluation, prediction, and analysis
* Extra Trees (Bagging)


```python
#Evaluation of various combinations of ExtraTrees

#Import the library
from sklearn.ensemble import ExtraTreesRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([])

for n_estimators in n_list:
    #Set the base model
    model = ExtraTreesRegressor(n_jobs=-1,n_estimators=n_estimators,random_state=seed)
    
    algo = "ET"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1254)
    comb.append("ET" + " %s" % 100 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 1254 for 100 estimators
```

## Evaluation, prediction, and analysis
* AdaBoost (Boosting)


```python
#Evaluation of various combinations of AdaBoost

#Import the library
from sklearn.ensemble import AdaBoostRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([])

for n_estimators in n_list:
    #Set the base model
    model = AdaBoostRegressor(n_estimators=n_estimators,random_state=seed)
    
    algo = "Ada"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1678)
    comb.append("Ada" + " %s" % 100 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 1678 with n=100
```

## Evaluation, prediction, and analysis
* Stochastic Gradient Boosting (Boosting)


```python
#Evaluation of various combinations of SGB

#Import the library
from sklearn.ensemble import GradientBoostingRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([])

for n_estimators in n_list:
    #Set the base model
    model = GradientBoostingRegressor(n_estimators=n_estimators,random_state=seed)
    
    algo = "SGB"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1278)
    comb.append("SGB" + " %s" % 50 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is ?
```

## Evaluation, prediction, and analysis
* XGBoost


```python
#Evaluation of various combinations of XGB

#Import the library
from xgboost import XGBRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([])

for n_estimators in n_list:
    #Set the base model
    model = XGBRegressor(n_estimators=n_estimators,seed=seed)
    
    algo = "XGB"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1169)
    comb.append("XGB" + " %s" % 1000 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 1169 with n=1000
```

## Evaluation, prediction, and analysis
* MLP (Deep Learning)


```python
#Evaluation of various combinations of multi-layer perceptrons

#Import libraries for deep learning
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

# define baseline model
def baseline(v):
     # create model
     model = Sequential()
     model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
     model.add(Dense(1, init='normal'))
     # Compile model
     model.compile(loss='mean_absolute_error', optimizer='adam')
     return model

# define smaller model
def smaller(v):
     # create model
     model = Sequential()
     model.add(Dense(v*(c-1)/2, input_dim=v*(c-1), init='normal', activation='relu'))
     model.add(Dense(1, init='normal', activation='relu'))
     # Compile model
     model.compile(loss='mean_absolute_error', optimizer='adam')
     return model

# define deeper model
def deeper(v):
 # create model
 model = Sequential()
 model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
 model.add(Dense(v*(c-1)/2, init='normal', activation='relu'))
 model.add(Dense(1, init='normal', activation='relu'))
 # Compile model
 model.compile(loss='mean_absolute_error', optimizer='adam')
 return model

# Optimize using dropout and decay
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm

def dropout(v):
    #create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu',W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(v*(c-1)/2, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='mean_absolute_error', optimizer=sgd)
    return model

# define decay model
def decay(v):
    # create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)
    model.compile(loss='mean_absolute_error', optimizer=sgd)
    return model

est_list = []
#uncomment the below if you want to run the algo
#est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]

for name, est in est_list:
 
    algo = name

    #Accuracy of the model using all features
    for m,i_cols_list in X_all:
        model = KerasRegressor(build_fn=est, v=1, nb_epoch=10, verbose=0)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo )

if (len(est_list)==0):
    mae.append(1168)
    comb.append("MLP" + " baseline" )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

#Plot the MAE of all combinations
fig, ax = plt.subplots()
plt.plot(mae)
#Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
#Plot the accuracy for all combinations
plt.show()    

#Best estimated performance is MLP=1168
```

## Make Predictions


```python
# Make predictions using XGB as it gave the best estimated performance        

X = numpy.concatenate((X_train,X_val),axis=0)
del X_train
del X_val
Y = numpy.concatenate((Y_train,Y_val),axis=0)
del Y_train
del Y_val

n_estimators = 1000

#Best model definition
best_model = XGBRegressor(n_estimators=n_estimators,seed=seed)
best_model.fit(X,Y)
del X
del Y
#Read test dataset
dataset_test = pandas.read_csv("../input/test.csv")
#Drop unnecessary columns
ID = dataset_test['id']
dataset_test.drop('id',axis=1,inplace=True)

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset_test.iloc[:,i])
    feature = feature.reshape(dataset_test.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

del cats

#Concatenate encoded attributes with continuous attributes
X_test = numpy.concatenate((encoded_cats,dataset_test.iloc[:,split:].values),axis=1)

del encoded_cats
del dataset_test

#Make predictions using the best model
predictions = numpy.expm1(best_model.predict(X_test))
del X_test
# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))
```

## What follows is some XGB Code


```python
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools

shift = 200
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)
def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain

if __name__ == "__main__":

    print('\nStarted')
    directory = '../input/'
    train = pd.read_csv(directory + 'train.csv')
    test = pd.read_csv(directory + 'test.csv')

    numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    categorical_feats = [x for x in train.columns[1:-1] if 'cat' in x]
    train_test, ntrain = mungeskewed(train, test, numeric_feats)
    
    # taken from Vladimir's script (https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114)
    for column in list(train.select_dtypes(include=['object']).columns):
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train

            remove = remove_train.union(remove_test)


            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x


            train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)

    # taken from Ali's script (https://www.kaggle.com/aliajouz/allstate-claims-severity/singel-model-lb-1117)
    train_test["cont1"] = np.sqrt(preprocessing.minmax_scale(train_test["cont1"]))
    train_test["cont4"] = np.sqrt(preprocessing.minmax_scale(train_test["cont4"]))
    train_test["cont5"] = np.sqrt(preprocessing.minmax_scale(train_test["cont5"]))
    train_test["cont8"] = np.sqrt(preprocessing.minmax_scale(train_test["cont8"]))
    train_test["cont10"] = np.sqrt(preprocessing.minmax_scale(train_test["cont10"]))
    train_test["cont11"] = np.sqrt(preprocessing.minmax_scale(train_test["cont11"]))
    train_test["cont12"] = np.sqrt(preprocessing.minmax_scale(train_test["cont12"]))

    train_test["cont6"] = np.log(preprocessing.minmax_scale(train_test["cont6"]) + 0000.1)
    train_test["cont7"] = np.log(preprocessing.minmax_scale(train_test["cont7"]) + 0000.1)
    train_test["cont9"] = np.log(preprocessing.minmax_scale(train_test["cont9"]) + 0000.1)
    train_test["cont13"] = np.log(preprocessing.minmax_scale(train_test["cont13"]) + 0000.1)
    train_test["cont14"] = (np.maximum(train_test["cont14"] - 0.179722, 0) / 0.665122) ** 0.25

    print('')
    for comb in itertools.combinations(COMB_FEATURE, 2):
        feat = comb[0] + "_" + comb[1]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
        train_test[feat] = train_test[feat].apply(encode)
        print('Combining Columns:', feat)

    print('')
    for col in categorical_feats:
        print('Analyzing Column:', col)
        train_test[col] = train_test[col].apply(encode)

    print(train_test[categorical_feats])

    ss = StandardScaler()
    train_test[numeric_feats] = \
        ss.fit_transform(train_test[numeric_feats].values)

    train = train_test.iloc[:ntrain, :].copy()
    test = train_test.iloc[ntrain:, :].copy()

    print('\nMedian Loss:', train.loss.median())
    print('Mean Loss:', train.loss.mean())

    ids = pd.read_csv('input/test.csv')['id']
    train_y = np.log(train['loss'] + shift)
    train_x = train.drop(['loss','id'], axis=1)
    test_x = test.drop(['loss','id'], axis=1)

    n_folds = 10
    cv_sum = 0
    early_stopping = 100
    fpred = []
    xgb_rounds = []

    d_train_full = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x)

    kf = KFold(train.shape[0], n_folds=n_folds)
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d' % (i+1))
        X_train, X_val = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_val = train_y.iloc[train_index], train_y.iloc[test_index]

        rand_state = 2016

        params = {
            'seed': 0,
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.03,
            'objective': 'reg:linear',
            'max_depth': 12,
            'min_child_weight': 100,
            'booster': 'gbtree'}

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        clf = xgb.train(params,
                        d_train,
                        100000,
                        watchlist,
                        early_stopping_rounds=50,
                        obj=fair_obj,
                        feval=xg_eval_mae)

        xgb_rounds.append(clf.best_iteration)
        scores_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
        cv_score = mean_absolute_error(np.exp(y_val), np.exp(scores_val))
        print('eval-MAE: %.6f' % cv_score)
        y_pred = np.exp(clf.predict(d_test, ntree_limit=clf.best_ntree_limit)) - shift

        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred
        cv_sum = cv_sum + cv_score

    mpred = pred / n_folds
    score = cv_sum / n_folds
    print('Average eval-MAE: %.6f' % score)
    n_rounds = int(np.mean(xgb_rounds))

    print("Writing results")
    result = pd.DataFrame(mpred, columns=['loss'])
    result["id"] = ids
    result = result.set_index("id")
    print("%d-fold average prediction:" % n_folds)

    now = datetime.now()
    score = str(round((cv_sum / n_folds), 6))
    sub_file = 'output/submission_5fold-average-xgb_fairobj_' + str(score) + '_' + str(
        now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("Writing submission: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='id')
```

## What follows is some keras code


```python
''' 
Author: Danijel Kivaranovic 
Title: Neural network (Keras) with sparse data
'''

## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

index = list(train.index)
print index[0:10]
np.random.shuffle(index)
print index[0:10]
train = train.iloc[index]
'train = train.iloc[np.random.permutation(len(train))]'

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y = np.log(train['loss'].values+200)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
def nn_model():
    model = Sequential()
    
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
        
    model.add(Dense(200, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.2))
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta')
    return(model)

## cv-folds
nfolds = 10
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 10
nepochs = 55
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True),
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 0)
        pred += np.exp(model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0])-200
        pred_test += np.exp(model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0])-200
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(yte)-200, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_absolute_error(np.exp(y)-200, pred_oob))

## train predictions
df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
df.to_csv('preds_oob.csv', index = False)

## test predictions
pred_test /= (nfolds*nbags)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('submission_keras_shift_perm.csv', index = False)
```

## What follows is a Markov Chain Model 


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
np.random.seed(123)
from subprocess import check_output
import xgboost as xgb
import gc

# Any results you write to the current directory are saved as output.

#### ANYTHING ABOVE IS KAGGLE JUNK #####

#''' This code gets a xgboost ensemble of 5 models, and tries to find the optimum weights through MCMC magic''''



num_folds = 2 #should be larger, but kaggle scirpts has run time 

def MAE(y,dtrain):
    answer = dtrain.get_label()
    answer = np.array(answer)
    prediction = np.array(y)
    error = np.exp(prediction) -np.exp(answer)
    error = np.mean((error**2)**.5)
    return 'mcc error',error
    
def MAE2(y,dtrain):
    answer = dtrain.loss2
    answer = np.array(answer)
    prediction = np.array(y)
    error = prediction - answer
    error = np.mean((error**2)**.5)
    return 'mcc error',error


## smaller dataset for faster training ###
train=pd.read_csv('train.csv',nrows=10000)
test=pd.read_csv('test.csv',nrows=10000)
train['loss']=np.log(train['loss']+200)
train['loss2']=np.exp(train['loss'])-200

## encode cat variables as discrete integers 
for i in list(train.keys()):
	if 'cat' in i:
		dictt = {}
		var = sorted(list(train[i].unique()))
		for ii in range(0,len(var)):
			dictt[var[ii]]=ii
		train[i] = train[i].map(dictt)
		test[i] = test[i].map(dictt)
        
parameters =[]
for i in (6,12):
    for j in (60,):
            for l in (1,2):
                depth = i
                min_child_weight = j
                gamma=l
                parameters += [[depth,min_child_weight,gamma],]
predictors = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat109', 'cat110', 'cat111', 'cat112', 'cat113', 'cat114', 'cat115', 'cat116', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']
target='loss'
result={}

## train 4 models with different paremeters ###
for i,j,l in parameters:
    xgtest=xgb.DMatrix(test[predictors].values,missing=np.NAN,feature_names=predictors)
    depth,min_child_weight,gamma=i,j,l
    result[(depth,min_child_weight,gamma)]=[]
    ### name of prediction ###
    name = 'feature_L2_%s_%s_%s_%s' %(str(depth), str(min_child_weight), str(gamma),str(num_folds))
    train  [name]=0
    test[name]=0
    for fold in range(0,num_folds):
        print ('\ntraining  parameters', i,j,l,',fold',fold)
        gc.collect() #to clear ram of garbage
        train_i = [x for x in train.index if x%num_folds != fold]
        cv_i = [x for x in train.index if x%num_folds == fold]
        dtrain= train.iloc[train_i]
        dcv = train.iloc[cv_i]
        xgcv    = xgb.DMatrix(dcv[predictors].values, label=dcv[target].values,missing=np.NAN,feature_names=predictors)
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values,missing=np.NAN,feature_names=predictors)

        #watchlist  = [ (xgtrain,'train'),(xgcv,'eval')] #i got valueerror in this
        params = {}
        params["objective"] =  "reg:linear"
        params["eta"] = 0.1
        params["min_child_weight"] = min_child_weight
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        params["scale_pos_weight"] = 1.0
        params["silent"] = 1
        params["max_depth"] = depth
        params['seed']=1
        params['lambda']=1
        params[ 'gamma']= gamma
        plst = list(params.items())
        early_stopping_rounds=5
        result_d=xgb.train(plst,xgtrain,50,maximize=0,feval = MAE)
        #print (result_d.predict(xgcv))
        print ('train_result',MAE(result_d.predict(xgcv),xgcv))
        ### write predictions onto train and test set ###
        train.set_value(cv_i,name,np.exp(result_d.predict(xgcv))-200)
        test.set_value(test.index,name,test[name]+(np.exp(result_d.predict(xgtest)-200)/num_folds))
        gc.collect()


#### NOW THE MCMC PART to find individal weights for ensemble####

features = [x for x in  train.keys() if 'feature' in x]
print ('features are these:', features)
num=len(features)
#intialize weights
weight = np.array([1.0/num,]*num)

# This is to define variables to be used later
train['pred_new']=0
train['pred_old']=0
counter = 0
n=1000 ###MCMC steps
result={}

for i in range(0,len(features)):
    train['pred_new'] += train[features[i]]*weight[i]
    print ('feature:',features[i],',MAE=',MAE2(train[features[i]],train))
print ('combined all features',',MAE=', MAE2(train.pred_new,train))
train['pred_old']=train['pred_new']
#### MCMC  #### 
### MCMC algo for dummies 
### 1. Get initialize ensemble weights
### 2. Generate new weights 
### 3. if MAE is lower, accept new weights immediately , or else accept new weights with probability of np.exp(-diff/.3)
### 4. repeat 2-3
for i in range(0,n):
     new_weights = weight+ np.array([0.005,]*num)*np.random.normal(loc=0.0, scale=1.0, size=num)
     new_weights[new_weights < 0.01]=0.01
     train['pred_new']=0
     for ii in range(0,len(features)):
         train['pred_new'] += train[features[ii]]*new_weights[ii]
     diff = MAE2(train.pred_new,train)[1] - MAE2(train.pred_old,train)[1]
     prob = min(1,np.exp(-diff/.3))
     random_prob = np.random.rand()
     if random_prob < prob:
         weight= new_weights
         train['pred_old']=train['pred_new']
         result[i] = (MAE2(train.pred_new,train)[1] ,MAE2(train.pred_old,train)[1],prob,random_prob ,weight)
         #print (MAE2(train.pred_new,train)[1] ,MAE2(train.pred_old,train)[1],prob,random_prob),
         counter +=1
print (counter *1.0 / n, 'Acceptance Ratio') #keep this [0.4,0.6] for best results
print ('best result MAE', sorted([result[i] for i in result])[0:1][0])

weight=sorted([result[i] for i in result])[0:1][-1]
train['pred_new']=0
for i in range(0,len(features)):
    train['pred_new'] += train[features[i]]*weight[i]
print ('combined all features plus MCMC weights:',',MAE=', MAE2(train.pred_new,train))

print ('weights:', weight[-1])
### notice the weights do not necessarily sum to 1 ###
```

## Ecoding nd Feature Comb - xgboost


```python

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
np.random.seed(123)
from subprocess import check_output
import xgboost as xgb
import gc

# Any results you write to the current directory are saved as output.

#### ANYTHING ABOVE IS KAGGLE JUNK #####

#''' This code gets a xgboost ensemble of 5 models, and tries to find the optimum weights through MCMC magic''''



num_folds = 2 #should be larger, but kaggle scirpts has run time 

def MAE(y,dtrain):
    answer = dtrain.get_label()
    answer = np.array(answer)
    prediction = np.array(y)
    error = np.exp(prediction) -np.exp(answer)
    error = np.mean((error**2)**.5)
    return 'mcc error',error
    
def MAE2(y,dtrain):
    answer = dtrain.loss2
    answer = np.array(answer)
    prediction = np.array(y)
    error = prediction - answer
    error = np.mean((error**2)**.5)
    return 'mcc error',error


## smaller dataset for faster training ###
train=pd.read_csv('train.csv',nrows=10000)
test=pd.read_csv('test.csv',nrows=10000)
train['loss']=np.log(train['loss']+200)
train['loss2']=np.exp(train['loss'])-200

## encode cat variables as discrete integers 
for i in list(train.keys()):
	if 'cat' in i:
		dictt = {}
		var = sorted(list(train[i].unique()))
		for ii in range(0,len(var)):
			dictt[var[ii]]=ii
		train[i] = train[i].map(dictt)
		test[i] = test[i].map(dictt)
        
parameters =[]
for i in (6,12):
    for j in (60,):
            for l in (1,2):
                depth = i
                min_child_weight = j
                gamma=l
                parameters += [[depth,min_child_weight,gamma],]
predictors = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat109', 'cat110', 'cat111', 'cat112', 'cat113', 'cat114', 'cat115', 'cat116', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']
target='loss'
result={}

## train 4 models with different paremeters ###
for i,j,l in parameters:
    xgtest=xgb.DMatrix(test[predictors].values,missing=np.NAN,feature_names=predictors)
    depth,min_child_weight,gamma=i,j,l
    result[(depth,min_child_weight,gamma)]=[]
    ### name of prediction ###
    name = 'feature_L2_%s_%s_%s_%s' %(str(depth), str(min_child_weight), str(gamma),str(num_folds))
    train  [name]=0
    test[name]=0
    for fold in range(0,num_folds):
        print ('\ntraining  parameters', i,j,l,',fold',fold)
        gc.collect() #to clear ram of garbage
        train_i = [x for x in train.index if x%num_folds != fold]
        cv_i = [x for x in train.index if x%num_folds == fold]
        dtrain= train.iloc[train_i]
        dcv = train.iloc[cv_i]
        xgcv    = xgb.DMatrix(dcv[predictors].values, label=dcv[target].values,missing=np.NAN,feature_names=predictors)
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values,missing=np.NAN,feature_names=predictors)

        #watchlist  = [ (xgtrain,'train'),(xgcv,'eval')] #i got valueerror in this
        params = {}
        params["objective"] =  "reg:linear"
        params["eta"] = 0.1
        params["min_child_weight"] = min_child_weight
        params["subsample"] = 0.5
        params["colsample_bytree"] = 0.5
        params["scale_pos_weight"] = 1.0
        params["silent"] = 1
        params["max_depth"] = depth
        params['seed']=1
        params['lambda']=1
        params[ 'gamma']= gamma
        plst = list(params.items())
        early_stopping_rounds=5
        result_d=xgb.train(plst,xgtrain,50,maximize=0,feval = MAE)
        #print (result_d.predict(xgcv))
        print ('train_result',MAE(result_d.predict(xgcv),xgcv))
        ### write predictions onto train and test set ###
        train.set_value(cv_i,name,np.exp(result_d.predict(xgcv))-200)
        test.set_value(test.index,name,test[name]+(np.exp(result_d.predict(xgtest)-200)/num_folds))
        gc.collect()


#### NOW THE MCMC PART to find individal weights for ensemble####

features = [x for x in  train.keys() if 'feature' in x]
print ('features are these:', features)
num=len(features)
#intialize weights
weight = np.array([1.0/num,]*num)

# This is to define variables to be used later
train['pred_new']=0
train['pred_old']=0
counter = 0
n=1000 ###MCMC steps
result={}

for i in range(0,len(features)):
    train['pred_new'] += train[features[i]]*weight[i]
    print ('feature:',features[i],',MAE=',MAE2(train[features[i]],train))
print ('combined all features',',MAE=', MAE2(train.pred_new,train))
train['pred_old']=train['pred_new']
#### MCMC  #### 
### MCMC algo for dummies 
### 1. Get initialize ensemble weights
### 2. Generate new weights 
### 3. if MAE is lower, accept new weights immediately , or else accept new weights with probability of np.exp(-diff/.3)
### 4. repeat 2-3
for i in range(0,n):
     new_weights = weight+ np.array([0.005,]*num)*np.random.normal(loc=0.0, scale=1.0, size=num)
     new_weights[new_weights < 0.01]=0.01
     train['pred_new']=0
     for ii in range(0,len(features)):
         train['pred_new'] += train[features[ii]]*new_weights[ii]
     diff = MAE2(train.pred_new,train)[1] - MAE2(train.pred_old,train)[1]
     prob = min(1,np.exp(-diff/.3))
     random_prob = np.random.rand()
     if random_prob < prob:
         weight= new_weights
         train['pred_old']=train['pred_new']
         result[i] = (MAE2(train.pred_new,train)[1] ,MAE2(train.pred_old,train)[1],prob,random_prob ,weight)
         #print (MAE2(train.pred_new,train)[1] ,MAE2(train.pred_old,train)[1],prob,random_prob),
         counter +=1
print (counter *1.0 / n, 'Acceptance Ratio') #keep this [0.4,0.6] for best results
print ('best result MAE', sorted([result[i] for i in result])[0:1][0])

weight=sorted([result[i] for i in result])[0:1][-1]
train['pred_new']=0
for i in range(0,len(features)):
    train['pred_new'] += train[features[i]]*weight[i]
print ('combined all features plus MCMC weights:',',MAE=', MAE2(train.pred_new,train))

print ('weights:', weight[-1])
### notice the weights do not necessarily sum to 1 ###
```
