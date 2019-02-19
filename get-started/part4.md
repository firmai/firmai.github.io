---
title: "Prediction, Time Series"
keywords: ts, rnn, xgboost, lgbm, time-series, ml, stocks, assets
description: Learn how to create clusters of Dockerized machines.
---
{% include_relative nav.html selected="4" %}

For business time series prediction we can differentiate between the ostensible randomness of a data series and the choice of appropriate models. In this section we have solely highlighted the prediction of continuous targets or resonse variables. All of these models' objective function can easily be configured to a classification problem i.e. will tomorrow's stock price increase or decrease from today's value instead of purely predicting the value. 

### Stochastic Time-Series 

It is often necessary to predict the future value of an ostensibly random variable or data series. Advanced machine learning techiques can be used to identify patterns in the data that might not at first be relevant by looking at the associated plots

<ul class="nav nav-tabs">
<li class="active"><a data-toggle="tab" href="#problem">Problem Sets</a></li>
<li><a data-toggle="tab" href="#data">Data Types</a></li>
<li><a data-toggle="tab" href="#code">Code Base</a></li>
<li><a data-toggle="tab" href="#examples">Examples</a></li>
</ul>

<div class="tab-content">
<div id="problem" class="tab-pane fade in active">
{% capture problem-content %}

##### Asset Class Prediction 

Asset Class Prediction:
e.g. commodity, stocks and bonds. 
* HFT
* Next Day
* Long Term

{% endcapture %}
{{ problem-content | markdownify }}
</div>
<div id="data" class="tab-pane fade" markdown="1">
{% capture data-content %}

| Data Types                                             | Description                                                     | Description                                                     |
|:-------------------------------------------------------|:----------------------------------------------------------------|:----------------------------------------------------------------|
| [Categorical](Categorical)               | Data that can be discretely classified.  | Country, Exchange, Currency, Dummy Variable, State, Industry.  |
| [Continuous ](Continuous )                 | Data that incrementally changes in values                          |Past Asset Price, Interest Rate, Competitors Price. |
| [Stepped](Stepped) | Similar to continuos but changes infrequently          | P/E, Quarterly Revenue,           |
| [Transformed Category  ](Transformed Category  )                 | A different datatype converted to categorical.                          |Traded inside standard deviation - yes, no. P/E above 10 - yes, no. |
| [Models](Models) | The prediction of additional models           | ARIMA, AR, MA.          |



{% endcapture %}
{{ data-content | markdownify }}
</div>
<div id="code" class="tab-pane fade" markdown="1">
{% capture code-content %}
<br>

<ul class="nav nav-tabs">
<div style="float:left; padding-right:0.5cm" class="active" ><a data-toggle="tab" href="#gbm">GBM</a></div>
<div style="float:left; padding-right:0.5cm"  ><a data-toggle="tab" href="#cnn">CNN</a></div>
<div  ><a data-toggle="tab" href="#rnn">RNN</a></div>
</ul>

<div class="tab-content">



<div id="gbm" class="tab-pane fade in active" markdown="1">
{% capture gbm-content %}


<div id="premodelcollapse" >

{% capture premodelcollapse-content %}

<div style="margin-top:0.5cm" >

<details ><summary>premodel</summary>

<div id="premodelinner" >
{% capture premodelinner-content %}

```python
#Load Data:
import pandas as pd
train = pd.read_csv("../input/train_1.csv")

#Explore For Insights:
import matplotlib.pyplot as plt
plt.plot(mean_group)
plt.show()

#Split Data in Three Sets:
from sklearn.model_selection import train_test_split



```

```
X_holdout = X.iloc[:int(len(X),:]
X_rest = X[X[~X_holdout]]

y_holdout = y.iloc[:int(len(y),:]
y_rest = y[y[~y_holdout]]

X_train, X_test, y_train, y_test = train_test_split(X_rest, y, test_size = 0.3, random_state = 0)

#Add Additional Features:
mean = X_train[col].mean()
```


{% endcapture %}
{{ premodelinner-content | markdownify }}

</div>


</details>


</div>

{% endcapture %}
{{ premodelcollapse-content | markdownify }}

</div>

<div id="modelcollapse" >

{% capture modelcollapse-content %}


<details open ><summary>model</summary>

<div id="modelinner" >
{% capture modelinner-content %}


<ul class="nav nav-tabs">
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#lgbm">LGBM</a></div>
<div  ><a data-toggle="tab" href="#xgb">XGBoost</a></div>
</ul>

<div class="tab-content">
<div id="lgbm" class="tab-pane fade in active" markdown="1">
{% capture lgbm-content %}

```python
import lightgbm as lgbm

learning_rate = 0.8
num_leaves =128
min_data_in_leaf = 1000
feature_fraction = 0.5
bagging_freq=1000
num_boost_round = 1000
params = {"objective": "regression",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
          "feature_fraction": feature_fraction, 
          "bagging_freq": bagging_freq,
          "verbosity": 0,
          "metric": "l2_root",
          "nthread": 4,
          "subsample": 0.9
          }

    dtrain = lgbm.Dataset(X_train, y_train)
    dvalid = lgbm.Dataset(X_validate, y_test, reference=dtrain)
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100,early_stopping_rounds=100)
    bst.predict(X_test, num_iteration=bst.best_iteration)
```


{% endcapture %}
{{ lgbm-content | markdownify }}
</div>

<div id="xgb" class="tab-pane fade" markdown="1">
{% capture xgb-content %}


```python
import xgboost as xgb

model = xgb.XGBRegressor(
                             learning_rate=0.037, max_depth=5, 
                             min_child_weight=20, n_estimators=180,
                             reg_lambda=0.8,booster = 'gbtree',
                             subsample=0.9, silent=1,
                             nthread = -1)

model.fit(train[feature_names], target)

pred = model.predict(test[feature_names])
```


{% endcapture %}
{{ xgb-content | markdownify }}


</div>

</div>

{% endcapture %}
{{ modelinner-content | markdownify }}

</div>

</details>



{% endcapture %}
{{ modelcollapse-content | markdownify }}


</div>

<div id="postmodelcollapse" >

{% capture postmodelcollapse-content %}

<div >

<details><summary>postmodel</summary>

<div id="inner" >
{% capture inner-content %}

```python
#Predict:

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)

#Assess Success of Prediction:

ROC AUC
TP/TN
F1
Confusion Matrix

#Tweak Parameters to Optimise Metrics:

#Select A new Model

#Repeat the process. 

#Final Showdown

Measure the performance of all models against the holdout set.
And pick the final model. 

```
{% endcapture %}
{{ inner-content | markdownify }}

</div>

</details>  

</div>

{% endcapture %}
{{ postmodelcollapse-content | markdownify }}


</div>
{% endcapture %}
{{ gbm-content | markdownify }}
</div>



<div id="cnn" class="tab-pane fade" markdown="1">
{% capture cnn-content %}


<div id="premodelcollapse" >

{% capture premodelcollapse-content %}

<div style="margin-top:0.5cm" >

<details ><summary>premodel</summary>

<div id="premodelinner" >
{% capture premodelinner-content %}

```python
#Load Data:
import pandas as pd
train = pd.read_csv("../input/train_1.csv")

#Explore For Insights:
import matplotlib.pyplot as plt
plt.plot(mean_group)
plt.show()

#Split Data in Three Sets:
from sklearn.model_selection import train_test_split



```

```
X_holdout = X.iloc[:int(len(X),:]
X_rest = X[X[~X_holdout]]

y_holdout = y.iloc[:int(len(y),:]
y_rest = y[y[~y_holdout]]

X_train, X_test, y_train, y_test = train_test_split(X_rest, y, test_size = 0.3, random_state = 0)

#Add Additional Features:
mean = X_train[col].mean()
```


{% endcapture %}
{{ premodelinner-content | markdownify }}

</div>


</details>


</div>

{% endcapture %}
{{ premodelcollapse-content | markdownify }}

</div>

<div id="modelcollapse" >

{% capture modelcollapse-content %}


<details open ><summary>model</summary>

<div id="modelinner" >
{% capture modelinner-content %}


<ul class="nav nav-tabs">
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#cnn2">2D CNN</a></div>
<div  ><a data-toggle="tab" href="#cnn1">1D CNN</a></div>
</ul>

<div class="tab-content">
<div id="cnn2" class="tab-pane fade in active" markdown="1">
{% capture cnn2-content %}

```python
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten

def create_model():
    conv = Sequential()
    conv.add(Conv2D(20, (1, 4), input_shape = PRED.shape[1:4], activation = 'relu'))
    conv.add(MaxPooling2D((1, 2)))
    conv.add(Flatten())
    conv.add(Dense(1, activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    conv.compile(loss = 'mse', optimizer = sgd, metrics = ['accuracy'])
    return conv
    
model = KerasRegressor(build_fn=create_model,  batch_size = 500, epochs = 20, verbose = 1,class_weight=class_weight)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


{% endcapture %}
{{ cnn2-content | markdownify }}
</div>

<div id="cnn1" class="tab-pane fade" markdown="1">
{% capture cnn1-content %}


```python
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten

def create_model():
    conv = Sequential()
    conv.add(Conv1D(20, 4, input_shape = PRED.shape[1:3], activation = 'relu'))
    conv.add(MaxPooling1D(2))
    conv.add(Dense(50, activation='relu'))
    conv.add(Flatten())
    conv.add(Dense(1, activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    conv.compile(loss = 'mse', optimizer = sgd, metrics = ['accuracy'])
    return conv
            
model = KerasRegressor(build_fn=create_model,  batch_size = 500, epochs = 20, verbose = 1,class_weight=class_weight)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


{% endcapture %}
{{ cnn1-content | markdownify }}


</div>

</div>

{% endcapture %}
{{ modelinner-content | markdownify }}

</div>

</details>



{% endcapture %}
{{ modelcollapse-content | markdownify }}


</div>

<div id="postmodelcollapse" >

{% capture postmodelcollapse-content %}

<div >

<details><summary>postmodel</summary>

<div id="inner" >
{% capture inner-content %}

```python
#Predict:

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)

#Assess Success of Prediction:

ROC AUC
TP/TN
F1
Confusion Matrix

#Tweak Parameters to Optimise Metrics:

#Select A new Model

#Repeat the process. 

#Final Showdown

Measure the performance of all models against the holdout set.
And pick the final model. 

```
{% endcapture %}
{{ inner-content | markdownify }}

</div>

</details>  

</div>

{% endcapture %}
{{ postmodelcollapse-content | markdownify }}


</div>
{% endcapture %}
{{ cnn-content | markdownify }}
</div>




<div id="rnn" class="tab-pane fade" markdown="1">
{% capture rnn-content %}


<div id="premodelcollapse" >

{% capture premodelcollapse-content %}

<div style="margin-top:0.5cm" >

<details ><summary>premodel</summary>

<div id="premodelinner" >
{% capture premodelinner-content %}


```python
#Load Data:
import pandas as pd
train = pd.read_csv("../input/train_1.csv")

#Explore For Insights:
import matplotlib.pyplot as plt
plt.plot(mean_group)
plt.show()

#Split Data in Three Sets:
from sklearn.model_selection import train_test_split

X_holdout = X.iloc[:int(len(X),:]
X_rest = X[X[~X_holdout]]

y_holdout = y.iloc[:int(len(y),:]
y_rest = y[y[~y_holdout]]

X_train, X_test, y_train, y_test = train_test_split(X_rest, y, test_size = 0.3, random_state = 0)

#Add Additional Features:
mean = X_train[col].mean()

```

{% endcapture %}
{{ premodelinner-content | markdownify }}

</div>


</details>


</div>

{% endcapture %}
{{ premodelcollapse-content | markdownify }}

</div>

<div id="modelcollapse" >

{% capture modelcollapse-content %}


<details open ><summary>model</summary>

<div id="modelinner" >
{% capture modelinner-content %}


<ul class="nav nav-tabs">
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#ltsm">LTSM</a></div>
</ul>

<div class="tab-content">
<div id="ltsm" class="tab-pane fade in active" markdown="1">
{% capture ltsm-content %}

```python
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten

def create_model():
    conv = Sequential()
    conv.add(Conv2D(20, (1, 4), input_shape = PRED.shape[1:4], activation = 'relu'))
    conv.add(MaxPooling2D((1, 2)))
    conv.add(Flatten())
    conv.add(Dense(1, activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    return conv
    
model = KerasClassifier(build_fn=create_model,  batch_size = 500, epochs = 20, verbose = 1,class_weight=class_weight)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


{% endcapture %}
{{ ltsm-content | markdownify }}
</div>

</div>

{% endcapture %}
{{ modelinner-content | markdownify }}

</div>

</details>



{% endcapture %}
{{ modelcollapse-content | markdownify }}


</div>

<div id="postmodelcollapse" >

{% capture postmodelcollapse-content %}

<div >

<details><summary>postmodel</summary>

<div id="inner" >
{% capture inner-content %}

```python
#Predict:

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)

#Assess Success of Prediction:

ROC AUC
TP/TN
F1
Confusion Matrix

#Tweak Parameters to Optimise Metrics:

#Select A new Model

#Repeat the process. 

#Final Showdown

Measure the performance of all models against the holdout set.
And pick the final model. 

```
{% endcapture %}
{{ inner-content | markdownify }}

</div>

</details>  

</div>

{% endcapture %}
{{ postmodelcollapse-content | markdownify }}


</div>
{% endcapture %}
{{ rnn-content | markdownify }}
</div>


</div>
{% endcapture %}
{{ code-content | markdownify }}

</div>

<div id="examples" class="tab-pane fade" markdown="1">
{% capture examples-content %}

[RNN Stock](https://github.com/Kulbear/stock-prediction)


{% endcapture %}
{{ examples-content | markdownify }}

</div>
</div>


<br>
<br>



### Regular Time-Series 

It is often necessary to predict the future value of an ostensibly random variable or data series. Advanced machine learning techiques can be used to identify patterns in the data that might not at first be relevant by looking at the associated plots

<ul class="nav nav-tabs">
<li class="active"><a data-toggle="tab" href="#problem1">Problem Sets</a></li>
<li><a data-toggle="tab" href="#data1">Data Types</a></li>
<li><a data-toggle="tab" href="#code1">Code Base</a></li>
<li><a data-toggle="tab" href="#examples1">Examples</a></li>
</ul>

<div class="tab-content">
<div id="problem1" class="tab-pane fade in active">
{% capture problem1-content %}

##### Asset Class Prediction 

Asset Class Prediction:
e.g. commodity, stocks and bonds. 
* HFT
* Next Day
* Long Term

{% endcapture %}
{{ problem1-content | markdownify }}
</div>
<div id="data1" class="tab-pane fade" markdown="1">
{% capture data1-content %}

| Data Types                                             | Description                                                     | Description                                                     |
|:-------------------------------------------------------|:----------------------------------------------------------------|:----------------------------------------------------------------|
| [Categorical](Categorical)               | Data that can be discretely classified.  | Country, Exchange, Currency, Dummy Variable, State, Industry.  |
| [Continuous ](Continuous )                 | Data that incrementally changes in values                          |Past Asset Price, Interest Rate, Competitors Price. |
| [Stepped](Stepped) | Similar to continuos but changes infrequently          | P/E, Quarterly Revenue,           |
| [Transformed Category  ](Transformed Category  )                 | A different datatype converted to categorical.                          |Traded inside standard deviation - yes, no. P/E above 10 - yes, no. |
| [Models](Models) | The prediction of additional models           | ARIMA, AR, MA.          |



{% endcapture %}
{{ data1-content | markdownify }}
</div>
<div id="code1" class="tab-pane fade" markdown="1">
{% capture code1-content %}
<br>

<ul class="nav nav-tabs">
<div style="float:left; padding-right:0.5cm" class="active" ><a data-toggle="tab" href="#arima">ARIMA</a></div>
<div  ><a data-toggle="tab" href="#prophet">Prophet</a></div>
</ul>

<div class="tab-content">



<div id="arima" class="tab-pane fade in active" markdown="1">
{% capture arima-content %}


<div id="premodelcollapse1" >

{% capture premodelcollapse1-content %}

<div style="margin-top:0.5cm" >

<details ><summary>premodel</summary>

<div id="premodelinner1" >
{% capture premodelinner1-content %}

```python
#Load Data:
import pandas as pd
train = pd.read_csv("../input/train_1.csv")

#Explore For Insights:
import matplotlib.pyplot as plt
plt.plot(mean_group)
plt.show()

#Split Data in Three Sets:
from sklearn.model_selection import train_test_split



```

```
X_holdout = X.iloc[:int(len(X),:]
X_rest = X[X[~X_holdout]]

y_holdout = y.iloc[:int(len(y),:]
y_rest = y[y[~y_holdout]]

X_train, X_test, y_train, y_test = train_test_split(X_rest, y, test_size = 0.3, random_state = 0)

#Add Additional Features:
mean = X_train[col].mean()

from statsmodels.tsa.arima_model import ARIMA 

Plot Autocorrelation and Partial Autocorrelation graphs to estimate the hyperparatmenters used in the ARIMA model:

from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

for key in sums:
    fig = plt.figure(1,figsize=[10,5])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    data = np.array(sums[key])
    autocorr = acf(data)
    pac = pacf(data)

    x = [x for x in range(len(pac))]
    ax1.plot(x[1:],autocorr[1:])

    ax2.plot(x[1:],pac[1:])
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')

    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    print(key)
    plt.show()
```


{% endcapture %}
{{ premodelinner-content | markdownify }}

</div>


</details>


</div>

{% endcapture %}
{{ premodelcollapse1-content | markdownify }}

</div>

<div id="modelcollapse1" >

{% capture modelcollapse1-content %}


<details open ><summary>model</summary>

<div id="modelinner1" >
{% capture modelinner1-content %}


<ul class="nav nav-tabs">
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#arimadefault">default</a></div>
</ul>

<div class="tab-content">
<div id="arimadefault" class="tab-pane fade in active" markdown="1">
{% capture arimadefault-content %}

```python
params = {'en': [4,1,0], 'ja': [7,1,1], 'de': [7,1,1], 'na': [4,1,0], 'fr': [4,1,0], 'zh': [7,1,1], 'ru': [4,1,0], 'es': [7,1,1]}

for key in sums:
    data = np.array(sums[key])
    result = None
    arima = ARIMA(data,params[key])
    result = arima.fit(disp=False)
    #print(result.params)
    pred = result.predict(2,599,typ='levels')
    x = [i for i in range(600)]
    i=0
    
    print(key)
    plt.plot(x[2:len(data)],data[2:] ,label='Data')
    plt.plot(x[2:],pred,label='ARIMA Model')
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()
    

# Naive decomposition of our Time Series as explained above
decomposition = sm.tsa.seasonal_decompose(df_date_index, model='multiplicative',freq = 7)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
rcParams['figure.figsize'] = 30, 20
```


{% endcapture %}
{{ arimadefault-content | markdownify }}
</div>

</div>

{% endcapture %}
{{ modelinner1-content | markdownify }}

</div>

</details>



{% endcapture %}
{{ modelcollapse1-content | markdownify }}


</div>

<div id="postmodelcollapse1" >

{% capture postmodelcollapse1-content %}

<div >

<details><summary>postmodel</summary>

<div id="inner1" >
{% capture inner1-content %}

```python
#Predict:

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)

#Assess Success of Prediction:

ROC AUC
TP/TN
F1
Confusion Matrix

#Tweak Parameters to Optimise Metrics:

#Select A new Model

#Repeat the process. 

#Final Showdown

Measure the performance of all models against the holdout set.
And pick the final model. 

```
{% endcapture %}
{{ inner1-content | markdownify }}

</div>

</details>  

</div>

{% endcapture %}
{{ postmodelcollapse1-content | markdownify }}


</div>
{% endcapture %}
{{ arima-content | markdownify }}
</div>



<div id="prophet" class="tab-pane fade" markdown="1">
{% capture prophet-content %}


<div id="premodelcollapse1" >

{% capture premodelcollapse1-content %}

<div style="margin-top:0.5cm" >

<details ><summary>premodel</summary>

<div id="premodelinner1" >
{% capture premodelinner1-content %}


```python
#Load Data:
import pandas as pd
train = pd.read_csv("../input/train_1.csv")

#Explore For Insights:
import matplotlib.pyplot as plt
plt.plot(mean_group)
plt.show()

#Split Data in Three Sets:
from sklearn.model_selection import train_test_split

X_holdout = X.iloc[:int(len(X),:]
X_rest = X[X[~X_holdout]]

y_holdout = y.iloc[:int(len(y),:]
y_rest = y[y[~y_holdout]]

X_train, X_test, y_train, y_test = train_test_split(X_rest, y, test_size = 0.3, random_state = 0)

#Add Additional Features:
mean = X_train[col].mean()

```

{% endcapture %}
{{ premodelinner1-content | markdownify }}

</div>


</details>


</div>

{% endcapture %}
{{ premodelcollapse1-content | markdownify }}

</div>

<div id="modelcollapse1" >

{% capture modelcollapse1-content %}


<details open ><summary>model</summary>

<div id="modelinner1" >
{% capture modelinner1-content %}


<ul class="nav nav-tabs">
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#prophetdefault">Default</a></div>
</ul>

<div class="tab-content">
<div id="prophetdefault" class="tab-pane fade in active" markdown="1">
{% capture prophetdefault-content %}

```python
from fbprophet import Prophet

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=10)
forecast = m.predict(future)
```


{% endcapture %}
{{ prophetdefault-content | markdownify }}
</div>

</div>

{% endcapture %}
{{ modelinner1-content | markdownify }}

</div>

</details>



{% endcapture %}
{{ modelcollapse1-content | markdownify }}


</div>

<div id="postmodelcollapse1" >

{% capture postmodelcollapse1-content %}

<div >

<details><summary>postmodel</summary>

<div id="inner1" >
{% capture inner1-content %}

```python
#Predict:

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)

#Assess Success of Prediction:

ROC AUC
TP/TN
F1
Confusion Matrix

#Tweak Parameters to Optimise Metrics:

#Select A new Model

#Repeat the process. 

#Final Showdown

Measure the performance of all models against the holdout set.
And pick the final model. 

```
{% endcapture %}
{{ inner1-content | markdownify }}

</div>

</details>  

</div>

{% endcapture %}
{{ postmodelcollapse1-content | markdownify }}


</div>
{% endcapture %}
{{ prophet-content | markdownify }}
</div>


</div>
{% endcapture %}
{{ code1-content | markdownify }}

</div>

<div id="examples1" class="tab-pane fade" markdown="1">
{% capture examples1-content %}

[RNN Stock](https://github.com/Kulbear/stock-prediction)


{% endcapture %}
{{ examples1-content | markdownify }}

</div>
</div>


