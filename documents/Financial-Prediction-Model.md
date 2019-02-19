---
title: Asset Price Prediction 
description: Predicting asset price in time series 
keywords: stock prediction, stock, asset price, predicting, time series, forecast 
---

##### Financial Prediction Model (FPM)

The FPM has applications in predicting volatile time-series information. The features can be of any kind as long as it is numerically represented. In many time-series models it is essentially to create hand engineered features, in this scenario lagging certain features lead to much better results than not doing so.

#### Finance

In finance a model similar to what is created here, can be used to predict the future movement of asset prices. As expected an extra tree regressor (or random forest if less constraint) including linear regressions on important variables led to low cross validated errors.

* Asset pricing prediction and modelling.
* Economic forecasts and decision making.
* Modelling a time series of a firm or individual's operating risk.

#### Business

Similar to the finance applications, there are thousands of opportunities to improve your internal business management.

* Tracking and predicting on the time series of operational data, such as dataroom temperature, wind-turbine speed etc to increase performance and efficiency.
* Tracking and predicting asset values and similarly tracking inventory levels.

---

```python
# Top 10% Financial Modeling Challenge
# https://www.kaggle.com/sonder/two-sigma-financial-modeling/d-play/run/946793

import kagglegym
import numpy as np
import pandas as pd
import random
from sklearn import ensemble, linear_model, metrics

env = kagglegym.make()
o = env.reset()
train = o.train
print(train.shape)
d_mean= train.median(axis=0)
train["nbnulls"]=train.isnull().sum(axis=1)
col=[x for x in train.columns if x not in ['id', 'timestamp', 'y']]

rnd=17

#keeping na information on some columns (best selected by the tree algorithms)
add_nas_ft=True
nas_cols=['technical_9', 'technical_0', 'technical_32', 'technical_16', 'technical_38',
'technical_44', 'technical_20', 'technical_30', 'technical_13']
#columns kept for evolution from one month to another (best selected by the tree algorithms)
add_diff_ft=True
diff_cols=['technical_22','technical_20', 'technical_30', 'technical_13', 'technical_34']

#homemade class used to infer randomly on the way the model learns
class createLinearFeatures:

    def __init__(self, n_neighbours=1, max_elts=None, verbose=True, random_state=None):
        self.rnd=random_state
        self.n=n_neighbours
        self.max_elts=max_elts
        self.verbose=verbose
        self.neighbours=[]
        self.clfs=[]

    def fit(self,train,y):
        if self.rnd!=None:
            random.seed(self.rnd)
        if self.max_elts==None:
            self.max_elts=len(train.columns)
        list_vars=list(train.columns)
        random.shuffle(list_vars)

        lastscores=np.zeros(self.n)+1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars=list_vars[self.n:]

        for elt in list_vars:
            indice=0
            scores=[]
            for elt2 in self.neighbours:
                if len(elt2)<self.max_elts:
                    clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
                    clf.fit(train[elt2+[elt]], y)
                    scores.append(metrics.mean_squared_error(y,clf.predict(train[elt2 + [elt]])))
                    indice=indice+1
                else:
                    scores.append(lastscores[indice])
                    indice=indice+1
            gains=lastscores-scores
            if gains.max()>0:
                temp=gains.argmax()
                lastscores[temp]=scores[temp]
                self.neighbours[temp].append(elt)

        indice=0
        for elt in self.neighbours:
            clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice=indice+1

    def transform(self, train):
        indice=0
        for elt in self.neighbours:
            #this line generates a warning. Could be avoided by working and returning
            #with a copy of train.
            #kept this way for memory management
            train['neighbour'+str(indice)]=self.clfs[indice].predict(train[elt])
            indice=indice+1
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)

#a home-made class attempt to remove outliers by successive quantization on residuals
class recurrent_linear_approx():
    def __init__(self, quant=.999, limit_size_train=.9):
        self.quant=quant
        self.limit_size_train=limit_size_train
        self.bestmodel=[]

    def fit(self, train, y):
        internal_model=linear_model.Ridge(fit_intercept=False)
        bestscore=1e15
        better=True
        indextrain=train.dropna().index
        limitlen=len(train)*self.limit_size_train
        while better:
            internal_model.fit(train.ix[indextrain], y.ix[indextrain])
            score=metrics.mean_squared_error(internal_model.predict(train.ix[indextrain]), y.ix[indextrain])
            if score < bestscore:
                bestscore=score
                self.bestmodel=internal_model
                residual=y.ix[indextrain]-internal_model.predict(train.ix[indextrain])
                indextrain=residual[abs(residual)<=abs(residual).quantile(self.quant)].index
                if len(indextrain)<limitlen:
                    better=False
            else:
                better=False
                self.bestmodel=internal_model

    def predict(self, test):
        return self.bestmodel.predict(test)

if add_nas_ft:
    for elt in nas_cols:
        train[elt + '_na'] = pd.isnull(train[elt]).apply(lambda x: 1 if x else 0)
        #no need to keep columns with no information
        if len(train[elt + '_na'].unique())==1:
            print("removed:", elt, '_na')
            del train[elt + '_na']
            nas_cols.remove(elt)

if add_diff_ft:
    train=train.sort_values(by=['id','timestamp'])
    for elt in diff_cols:
        #a quick way to obtain deltas from one month to another but it is false on the first
        #month of each id
        train[elt+"_d"]= train[elt].rolling(2).apply(lambda x:x[1]-x[0]).fillna(0)
    #removing month 0 to reduce the impact of erroneous deltas
    train=train[train.timestamp!=0]

print(train.shape)
cols=[x for x in train.columns if x not in ['id', 'timestamp', 'y']]

#generation of linear models
cols2fit=['technical_22','technical_20', 'technical_30_d', 'technical_20_d', 'technical_30',
'technical_13', 'technical_34']
models=[]
columns=[]
residuals=[]
for elt in cols2fit:
    print("fitting linear model on ", elt)
    model=recurrent_linear_approx(quant=.99, limit_size_train=.9)
    model.fit(train.loc[:,[elt]],train.loc[:, 'y'])
    models.append(model)
    columns.append([elt])
    residuals.append(abs(model.predict(train[[elt]].fillna(d_mean))-train.y))

train=train.fillna(d_mean)

#adding all trees generated by a tree regressor
print("adding new features")
featureexpander=createLinearFeatures(n_neighbours=30, max_elts=2, verbose=True, random_state=rnd)
index2use=train[abs(train.y)<0.07].index
featureexpander.fit(train.ix[index2use,cols],train.ix[index2use,'y'])
trainer=featureexpander.transform(train[cols])
treecols=trainer.columns

print("training trees")
model = ensemble.ExtraTreesRegressor(n_estimators=29, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
model.fit(trainer,train.y)
print(pd.DataFrame(model.feature_importances_,index=treecols).sort_values(by=[0]).tail(30))
for elt in model.estimators_:
    models.append(elt)
    columns.append(treecols)
    residuals.append(abs(elt.predict(trainer)-train.y))

#model selection : create a new target selecting models with lowest asolute residual for each line
#the objective at this step is to keep only the few best elements which should
#lead to a better generalization
num_to_keep=10
targetselector=np.array(residuals).T
targetselector=np.argmin(targetselector, axis=1)
print("selecting best models:")
print(pd.Series(targetselector).value_counts().head(num_to_keep))

tokeep=pd.Series(targetselector).value_counts().head(num_to_keep).index
tokeepmodels=[]
tokeepcolumns=[]
tokeepresiduals=[]
for elt in tokeep:
    tokeepmodels.append(models[elt])
    tokeepcolumns.append(columns[elt])
    tokeepresiduals.append(residuals[elt])

#creating a new target for a model in charge of predicting which model is best for the current line
targetselector=np.array(tokeepresiduals).T
targetselector=np.argmin(targetselector, axis=1)

print("training selection model")
modelselector = ensemble.ExtraTreesClassifier(n_estimators=30, max_depth=4, n_jobs=-1, random_state=rnd, verbose=0)
modelselector.fit(trainer, targetselector)
print(pd.DataFrame(modelselector.feature_importances_,index=treecols).sort_values(by=[0]).tail(30))

lastvalues=train[train.timestamp==905][['id']+diff_cols].copy()

print("end of trainind, now predicting")
indice=0
countplus=0
rewards=[]
while True:
    indice+=1
    test = o.features
    test["nbnulls"]=test.isnull().sum(axis=1)
    if add_nas_ft:
        for elt in nas_cols:
            test[elt + '_na'] = pd.isnull(test[elt]).apply(lambda x: 1 if x else 0)
    test=test.fillna(d_mean)

    pred = o.target
    if add_diff_ft:
        #creating deltas from lastvalues
        indexcommun=list(set(lastvalues.id) & set(test.id))
        lastvalues=pd.concat([test[test.id.isin(indexcommun)]['id'],
            pd.DataFrame(test[diff_cols][test.id.isin(indexcommun)].values-lastvalues[diff_cols][lastvalues.id.isin(indexcommun)].values,
            columns=diff_cols, index=test[test.id.isin(indexcommun)].index)],
            axis=1)
        #adding them to test data
        test=test.merge(right=lastvalues, how='left', on='id', suffixes=('','_d')).fillna(0)
        #storing new lastvalues
        lastvalues=test[['id']+diff_cols].copy()

    testid=test.id
    test=featureexpander.transform(test[cols])
    #prediction using modelselector and models list
    selected_prediction = modelselector.predict_proba(test.loc[: ,treecols])
    for ind,elt in enumerate(tokeepmodels):
        pred['y']+=selected_prediction[:,ind]*elt.predict(test[tokeepcolumns[ind]])

    indexbase=pred.index
    pred.index=testid
    oldpred=pred['y']
    pred.index=indexbase

    o, reward, done, info = env.step(pred)
    rewards.append(reward)
    if reward>0:
        countplus+=1

    if indice%100==0:
        print(indice, countplus, reward, np.mean(rewards))

    if done:
        print(info["public_score"])
        break
```

Following is two scripts that are in most parts different, which also scored high.

```python
# Rollng Regression Script: top 14%

import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.ensemble import ExtraTreesRegressor

# The "environment" is our interface for code competitions
env = kagglegym.make()
# We get our initial observation by calling "reset"
o = env.reset()
# Get the train dataframe

excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]

train = pd.read_hdf('../input/train.h5')
train = train[col]
d_mean= train.median(axis=0)

train = o.train[col]
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train_1 = train.fillna(d_mean)
train_1['znull'] = n
n = []

rfr = ExtraTreesRegressor(n_estimators=30, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train_1, o.train['y'])

train = o.train
cols_to_use = ['technical_20','technical_30','technical_13','y']
excl = ['id', 'y', 'timestamp']
allcol = [c for c in train.columns if ((c in excl)|(c in cols_to_use))]
allcol1 = [c for c in allcol if not (c == 'y')]
train=train[allcol]

low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

# mean_values = train.median(axis=0)
#train.fillna(mean_values, inplace=True)

pred = np.array(train[cols_to_use])
tis=np.array(train.loc[:, 'timestamp'],dtype=int)
ids=np.array(train.loc[:, 'id'],dtype=int)
del train

predtab=np.zeros((max(tis)+1,max(ids)+1,pred.shape[1]))
predtab[:,:,:]=np.nan
for c in range(0,max(ids)+1) :
  sel = np.array(ids==c)
  predtab[tis[sel],c,:]=pred[sel,:]
del pred,tis,ids

gconst = [1,-1]
for iter in range(0,2):
    dt=gconst[0]*predtab[:-1,:,0:3]+gconst[1]*predtab[1:,:,0:3]
    trg=predtab[:-1,:,-1]
    ok=np.array((np.sum(np.isnan(dt),axis=2)==0)&np.isfinite(trg)&(trg<high_y_cut)&(trg>low_y_cut))
    met1=lm.LinearRegression()
    dt = dt[np.repeat(ok.reshape((ok.shape[0],ok.shape[1],1)),dt.shape[2],axis=2)].reshape(np.sum(ok),dt.shape[2])
    met1.fit (dt,trg[ok])
    r2 = met1.score(dt,trg[ok])
    dconst = met1.coef_
    print('Dconst=',dconst,' R=',np.sqrt(r2))

    dt=np.dot(predtab[:,:,0:3],dconst).reshape((predtab.shape[0],predtab.shape[1],1))
    dt=np.concatenate((dt[:-1,:,:],dt[1:,:,:]),axis=2)
    ok=np.array((np.sum(np.isnan(dt),axis=2)==0)&np.isfinite(trg)&(trg<high_y_cut)&(trg>low_y_cut))
    met1=lm.LinearRegression()
    dt = dt[np.repeat(ok.reshape((ok.shape[0],ok.shape[1],1)),dt.shape[2],axis=2)].reshape(np.sum(ok),dt.shape[2])
    met1.fit (dt,trg[ok])
    r2 = met1.score(dt,trg[ok])
    gconst = met1.coef_
    print('Gconst=',gconst,' R=',np.sqrt(r2))
del dt, trg, ok

def expandmas2 (mas,n):
    if (mas.shape[1]<=n):
        mas1=np.zeros((mas.shape[0],int(n*1.2+1)))
        for i in range(0,mas.shape[0]):
            mas1[i,:]=mas[i,-1]
        mas1[:,:mas.shape[1]]=mas
        mas = mas1
    return mas
def expandmas (mas,n,m):
    if (mas.shape[0]<=n):
        mas1=np.zeros((int(n*1.2+1),mas.shape[1],mas.shape[2]))
        mas1[:,:,:]=np.nan
        mas1[:mas.shape[0],:mas.shape[1],:]=mas
        mas = mas1
    if (mas.shape[1]<=m):
        mas1=np.zeros((mas.shape[0],int(m*1.2+1),mas.shape[2]))
        mas1[:,:,:]=np.nan
        mas1[:mas.shape[0],:mas.shape[1],:]=mas
        mas = mas1
    return mas

realhist = predtab.shape[0]
coef = np.zeros((1,realhist))
def trainrolling (tset):
    for t in tset :
            s0=max(t-1,1)
            y=predtab[s0,:,-1]
            x=predtab[s0-1,:,-1]
            ok=np.array(np.isfinite(x)&np.isfinite(y)&(x>low_y_cut)&(x<high_y_cut)&(y<high_y_cut)&(y>low_y_cut))
#            ok=np.array(np.isfinite(x)&np.isfinite(y))
            if np.sum(ok)==0:
                    coef[0,t]=coef[0,t-1]
            else:
                    x1=x[ok]
                    y1=y[ok]
#                    alp1=0.65*(np.std(x1)+np.std(y1))*max(200,x1.shape[0])
                    alp1=np.std(np.concatenate((x1,y1)))*max(200,x1.shape[0])
                    x1=np.concatenate((x1,[alp1]))
                    y1=np.concatenate((y1,[alp1*coef[0,t-1]]))
                    coef[0,t]=np.sum(x1*y1)/np.sum(x1*x1)
            if t>=1:
                res = predtab[t-1,:,-1]*coef[0,t]
    return res,coef

reward=0
n = 0
rewards = []
t0 = 0
while True:
    test = o.features[allcol1].copy()
#    test['id'] = observation.target.id
#    test.fillna(mean_values, inplace=True)
    pred=np.array(test[cols_to_use[:-1]])
    maxts = int(max(test['timestamp']))
    maxid = int(max(test['id']))
    predtab=expandmas (predtab,maxts,maxid)
    coef =expandmas2 (coef,maxts)

    resout = np.zeros((pred.shape[0]))
    for t in range(int(min(test['timestamp'])),maxts+1) :
        sel=np.array(test['timestamp']==t)
        ids=np.array(test.loc[sel,'id'],dtype=int)

        predtab[t,ids,0:pred.shape[1]]=pred[sel,:]
        if (t<1):
            continue
        old = predtab[t-1,ids,-1]
#        new = np.dot(predtab[t,ids,0:3]-predtab[t-1,ids,0:3],dconst)
        new = np.dot(predtab[t-1:t+1,ids,0:3],dconst)
        new = np.dot(new.T,gconst)
        old[np.isnan(old)]=new[np.isnan(old)]
        predtab[t-1,ids,-1]=old
        t0=int(min(t0,t-1))

        res,coef = trainrolling (range(t0+1,t+1))
        res = res[ids]
        res [np.isnan(res)]=0.
        resout[sel]=res
        t0=t
    test = o.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n

    o.target.y = (resout.clip(low_y_cut, high_y_cut)*0.34) + (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.66)
    o.target.y = o.target.y
    #observation.target.fillna(0, inplace=True)
    target = o.target
    timestamp = o.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        print(np.mean(rewards))

    o, reward, done, info = env.step(target)
#    print(reward)
    if done:
        break
    rewards.append(reward)
    n = n + 1
print(info)
```

```python
# Outliers Script top 20%

import kagglegym
import numpy as np
import pandas as pd

# sklearn libraries
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 120)

environment = kagglegym.make() # This creates an environment in the API for me to play in
observation = environment.reset() # Resets to first observations "view of what you can see presently"

excl = [environment.ID_COL_NAME, environment.TARGET_COL_NAME, environment.TIME_COL_NAME,
environment.SAMPLE_COL_NAME]
col = [c for c in observation.train.columns if c not in excl]

from scipy import stats

df_old = observation.train
df = df_old[(np.abs(stats.zscore(df_old["y"])) < 3.6)]
#df = observation.train
df_full = pd.read_hdf('../input/train.h5')
d_mean= df[col].mean(axis=0)

min_y = df["y"].min()
max_y = df["y"].max()
print (min_y, max_y)

X_train =df[col]
n = X_train.isnull().sum(axis=1)

for c in col:
    r = pd.isnull(X_train.loc[:, c])
    X_train[c + '_nan_'] = r
    d_mean[c + '_nan_'] = 0

X_train = X_train.fillna(d_mean)
df = df.fillna(d_mean)
X_train['znull'] = n
n = []

cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
"""['technical_30', 'technical_20', 'fundamental_11', 'technical_27', 'technical_19', 'technical_35',
'technical_11', 'technical_2', 'technical_34', 'fundamental_53', 'fundamental_51',
'fundamental_58']"""

#observation = environment.reset()

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
fit_one = rfr.fit(X_train, df["y"].values)

lr = LinearRegression()
# See what happend if you change this to X_full
fit_two = lr.fit(np.array(df[cols_to_use]), df["y"].values)

X_train= []
ymean_dict = dict(observation.train.groupby(["id"])["y"].mean())

#observation = environment.reset()

while True:
    X_test = observation.features[col]
    #I reckoned what happened here is that the features column is a different set of data.
    n = X_test.isnull().sum(axis=1)
    for c in X_test.columns:
        X_test[c + '_nan_'] = pd.isnull(X_test[c])
    X_test = X_test.fillna(d_mean)
    X_test['znull'] = n

    temp = observation.features.fillna(d_mean)
    X_test_two = np.array(temp[cols_to_use])

    pred = observation.target

    pred['y'] = (fit_one.predict(X_test).clip(min_y, max_y) * 0.65)
    + (fit_two.predict(X_test_two).clip(min_y, max_y)* 0.35)
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]

    timestamp = temp["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = environment.step(pred)
    if done:
        break
info

#0.0115
```