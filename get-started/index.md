---
title: "Prediction, Classification"
keywords: get started, prediction, classification, model, keras, concepts, supervised, learning
description: Get oriented on some basics of Docker before diving into the walkthrough.
redirect_from:
- /getstarted/
- /get-started/part1/
- /engine/getstarted/
- /learn/
- /engine/getstarted/step_one/
- /engine/getstarted/step_two/
- /engine/getstarted/step_three/
- /engine/getstarted/step_four/
- /engine/getstarted/step_five/
- /engine/getstarted/step_six/
- /engine/getstarted/last_page/
- /engine/getstarted-voting-app/
- /engine/getstarted-voting-app/node-setup/
- /engine/getstarted-voting-app/create-swarm/
- /engine/getstarted-voting-app/deploy-app/
- /engine/getstarted-voting-app/test-drive/
- /engine/getstarted-voting-app/customize-app/
- /engine/getstarted-voting-app/cleanup/
- /engine/userguide/intro/
- /mac/started/
- /windows/started/
- /linux/started/
- /getting-started/
- /mac/step_one/
- /windows/step_one/
- /linux/step_one/
- /engine/tutorials/dockerizing/
- /mac/step_two/
- /windows/step_two/
- /linux/step_two/
- /mac/step_three/
- /windows/step_three/
- /linux/step_three/
- /engine/tutorials/usingdocker/
- /mac/step_four/
- /windows/step_four/
- /linux/step_four/
- /engine/tutorials/dockerimages/
- /userguide/dockerimages/
- /engine/userguide/dockerimages/
- /mac/last_page/
- /windows/last_page/
- /linux/last_page/
- /mac/step_six/
- /windows/step_six/
- /linux/step_six/
- /engine/tutorials/dockerrepos/
- /userguide/dockerrepos/
- /engine/userguide/containers/dockerimages/
---

{% include_relative nav.html selected="1" %}

Welcome! This section highlights important business machine learning models. Many of these models are not code-complete and simply provides excerpted pseudo-like code. When in doubt, 90% of the code mentions on this website are those of the [Python](https://www.python.org/) programming language. 

This six-part documentation identifies:

1. Some of the best classificaiton models, on this page.
2. [Continuous value prediction problems](part2.md)
3. [The use of Natural Language Processing](part3.md)
4. [Important time series solutions](part4.md)
5. [The core principles of recommender systems](part5.md)
6. [And experimental image and voice technologies](part6.md)

With the prerequisite knowledge of python and expecially the data science components of data science, these sections
are easily explorable. If you need any help with the models feel free to get in touch for a consultation. 



### Binary Classification

A lot of classification problems are binary in nature such as predicting whether the stock price will go up or down in the future, predicting gender and predicting whetehr a prospective client will buy your product. 

<ul class="nav nav-tabs">
<li class="active"><a data-toggle="tab" href="#problem">Problem Sets</a></li>
<li><a data-toggle="tab" href="#data">Data Types</a></li>
<li><a data-toggle="tab" href="#code">Code Base</a></li>
<li><a data-toggle="tab" href="#examples">Examples</a></li>
</ul>

<div class="tab-content">
<div id="problem" class="tab-pane fade in active">
{% capture problem-content %}

##### Binary Business Prediction:

* Future direction of commodity, stocks and bonds prices. 
* Predicting a customer demographic.
* Predict wheteher customers will respond to direct mail.
* Predict the pobabilitiy of damage in a home inspection 
* Predict the liklihood that a grant application will succeed.
* Predict job success using a 10 part questionaire. 
* Predict those most likley to donate to a cause. 

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
<div  ><a data-toggle="tab" href="#mlp">MLP</a></div>
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
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
          "feature_fraction": feature_fraction, 
          "bagging_freq": bagging_freq,
          "verbosity": 0,
          "metric": "binary_logloss",
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
import xgboost as XGB
model = xgb.XGBClassifier(objective='binary:logistic',
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
<div  ><a data-toggle="tab" class="active" href="#cnn1">1D CNN</a></div>
</ul>

<div class="tab-content">
<div id="cnn1" class="tab-pane fade in active" markdown="1">
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
    conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
    return conv
            
model = KerasClassifier(build_fn=create_model,  batch_size = 500, epochs = 20, verbose = 1,class_weight=class_weight)
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




<div id="mlp" class="tab-pane fade" markdown="1">
{% capture mlp-content %}


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
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#mlpdefault">Default</a></div>
</ul>

<div class="tab-content">
<div id="mlpdefault" class="tab-pane fade in active" markdown="1">
{% capture mlpdefault-content %}

```python
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


{% endcapture %}
{{ mlpdefault-content | markdownify }}
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
{{ mlp-content | markdownify }}
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





### Multi-class Classification

This section relates to predictions for multiple classes. Machine learning has improved the quality of these types of predictions.

<ul class="nav nav-tabs">
<li class="active"><a data-toggle="tab" href="#problem1">Problem Sets</a></li>
<li><a data-toggle="tab" href="#data1">Data Types</a></li>
<li><a data-toggle="tab" href="#code1">Code Base</a></li>
<li><a data-toggle="tab" href="#examples1">Examples</a></li>
</ul>

<div class="tab-content">
<div id="problem1" class="tab-pane fade in active">
{% capture problem1-content %}

##### Multi-class Prediction 

* Item specific sales prediction i.e. unit of sales
* Predicting store sales.
* Predict the unit of sales fro multiple items.
* Predicting the liklihood of certain crimes occuring at different points geographically and at different times.
* What when, where and at what severity will the flu strike.
* New empoyees predict the level of access and what access they require.
* Predict the most pressing community issue.
* What customers wil purcahse what policy.
* Predict which shoppers are most likely to repeat purchase.
* Predict which blog post from a selcetion would be most popular
* Predict destination of taxi with initial partial trajectories.


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
<div style="float:left; padding-right:0.5cm" class="active" ><a data-toggle="tab" href="#gbm1">GBM</a></div>
<div  ><a data-toggle="tab" href="#mlp1">MLP</a></div>
</ul>

<div class="tab-content">



<div id="gbm1" class="tab-pane fade in active" markdown="1">
{% capture gbm1-content %}


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
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#lgbm1">LGBM</a></div>
<div style="float:left; padding-right:0.5cm" ><a data-toggle="tab" href="#xgb1">XGBoost</a></div>
</ul>

<div class="tab-content">
<div id="lgbm1" class="tab-pane fade in active" markdown="1">
{% capture lgbm1-content %}

```python
import lightgbm as lgbm

learning_rate = 0.8
num_leaves =128
min_data_in_leaf = 1000
feature_fraction = 0.5
bagging_freq=1000
num_boost_round = 1000
params = {"objective": "multiclass",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
          "feature_fraction": feature_fraction, 
          "bagging_freq": bagging_freq,
          "verbosity": 0,
          "metric": "multi_logloss",
          "nthread": 4,
          "subsample": 0.9
          }

    dtrain = lgbm.Dataset(X_train, y_train)
    dvalid = lgbm.Dataset(X_validate, y_test, reference=dtrain)
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100,early_stopping_rounds=100)
    bst.predict(X_test, num_iteration=bst.best_iteration)
```


{% endcapture %}
{{ lgbm1-content | markdownify }}
</div>

<div id="xgb1" class="tab-pane fade" markdown="1">
{% capture xgb1-content %}

```python
import xgboost as XGB
model = xgb.XGBClassifier(objective='multi:softmax',
                     learning_rate=0.037, max_depth=5, 
                     min_child_weight=20, n_estimators=180,
                     reg_lambda=0.8,booster = 'gbtree',
                     subsample=0.9, silent=1,
                     nthread = -1)

model.fit(train[feature_names], target)

pred = model.predict(test[feature_names])
```


{% endcapture %}
{{ xgb1-content | markdownify }}
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
{{ gbm1-content | markdownify }}
</div>



<div id="mlp1" class="tab-pane fade" markdown="1">
{% capture mlp1-content %}


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
<div style="float:left; padding-right:0.5cm" class="active"  ><a data-toggle="tab" href="#mlpdefault">Default</a></div>
</ul>

<div class="tab-content">
<div id="mlpdefault" class="tab-pane fade in active" markdown="1">
{% capture mlpdefault-content %}

```python
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```


{% endcapture %}
{{ mlpdefault-content | markdownify }}
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
{{ mlp1-content | markdownify }}
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










[On to Part 2 >>](part2.md){: class="button outline-btn" style="margin-bottom: 30px; margin-right: 100%"}
