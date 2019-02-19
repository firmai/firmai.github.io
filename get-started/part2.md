---
title: "Prediction, Continuous Values"
keywords: continuous, values, regression, prediction, model, gbt,  keras, concepts, supervised, learning
description: Learn how to write, build, and run a simple app -- the Docker way.
---

{% include_relative nav.html selected="2" %}

### Continuous Values 

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

##### Comparative Prediction Tasks

* Predict how many times a cutomer would call customer service in the next year.
* Corporate Valuation.
* Salary Prediction and Recommendation.
* Predict the success of a product at launch.
* Predict probabilistic distribution of hourly rain using polarimetric radar measurements.
* Predict the sale price at auction.
* Predict census retrurn rates.
* Predict Customer Value.
* Predict Severity of Claims/Final Cost.
* Clicks how many clicks/interest will something receives based on its charactersitcs.
* House price valuations.
* Predict duration for process.
* Predict Perscription Volume.

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
    conv.compile(loss = 'mse', optimizer = sgd, metrics = ['accuracy'])
    return conv
    
model = KerasRegressor(build_fn=create_model,  batch_size = 500, epochs = 20, verbose = 1,class_weight=class_weight)
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

