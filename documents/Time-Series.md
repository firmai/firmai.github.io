# TS - Time Series
In modelling generally power, linear, exponential and log function reappear. -> What is clear is that deeplearning can better substitute all of these models. 

A model is a simlification of a complex process. We don't want the model to be overly complex or oveerly simple, but that is only true for the base models, replication of a simple base model should not be viewed as complex, it is generally a better way to simplify. 

Once you have  model you can do:
Prediction, Forecasting, Optimisation, Ranking/Targeting, Scenario Planning, Interpreting the Coefficient, Sensitivity Analysis 

1. Predicting Price of Diamond Given Weight. 
2. Predicting the future price of diamonds using historical data.
3. Predicting the best price to sell the diamonds to optimise profit.
4. Given a set of diamonds, what would customers love purcahsing.
  1. We can sort predictions to create a ranking. 
5. If the interest rate changes in the future wht would be the effect on profit. 
6. Coefficient is the power, sometime they are helpfull, i.e. the elasticity coefficient for the sale of diamonds.
7. How sensitive the model is to certain assumptions such as the future interest rate, helps us to understand what assumptions should be given a second look.

Forecasting is time series, prediction is less related to time series it can just be cross-sectional.
Models creates an instutional memory, that need not be known by any one person.
Models are also serendepitous insigh generators. 

Most models are somewhere between emperical and teoretical 
Someone have laid down a set of assumptions of relationships (i.e. markets is efficinet) - this is theoretical such as the options pricing model.    

Other models look at the data and try and approximate the underlying procecss (separated out profitable and unprofitable customers and trying to identify the driving charactersitcs) - that is a purely emperical model. 

Deterministic and probabilistic models:
Deterministic - give a fixed set of inputs the model alwasy gives the same output. - interest model, 8% for 10 years will always double your money. 
Probabilistic/Stochastic - Bough loteraty tickets instead, that fundamentally depends so there is an expected value. 

You get discrete and continous variables, i.e. you get discrete and continous processes as they are the input.

Static and dynamic models:

- Static: What is the probability that all the employees will show up today.
- Deterministic: If the company is predicted to be bankrupt, what is the liklihood that they would exit favourably.


All these functions are easy i.e. where are you throwing the X in. 
Log is the inverse of exponential.

Linear function has two parameters/coefficients the orgin or intercept and the slope or gradient. 

I think exponential functions are more flexible than power function. 

Log function diminishing returns of scale. 

Classical Optimisation requires calculas, non-classical, derivative-free or black-box optimisation does not require calculas.

Probabilistic Models - incorporate random variable and probability distribution - they go hand in hand. 

It incoprates uncertainty expelicty, then we are able to understand the uncertainty explicity, it is more realistic to do so, i.e.
confidence intervals and the like.

We don't know the future price of oil - so now we will create some realistic probability distribution:
Drugs in development probability of success and anticipated revenue.

Some specific probabilistic models:

1. Regression Models - Works on data, finding the best looking line and create a band with interval, the smaller amount of noise the smaller the bands of errors the bettter, smaller bands, closer confidence interval, better approximation
2. Probability Trees - also from data
3. Monte Carlo Simulation - Scenario Analysis Via a Probabilistic Scenario - propogate the unceratinty of equations like optimisation equations. - you can have multiple unkowns. Like a scenario analysis. 
4. Markov Models - Dynamic Model looking at a process. - individual performance style, is dicrete, i.e. certain states or classification, employed, unemployed looking, unemployed not looking. It models the probability between states. You can also stay in the state. Markov Property - Lack of memory. Transition probabilities only depend on the current state.Given the present the future does not depend on the past. 

Very True - A lot of things are driven by power laws and not the normal discibution, cities, size of animals, buildings, trees.

With continuous values we have probaiility density functions. 

We would like to **summarise the probability distributions**

Where is the center of the probability distribution, the spread of the dsitribution, i.e.  standard deviation - compared agaisnt a nomrla distribution. There is other probability distribution summaries like kurtosis. 

There is hundreds of different probability dsitributions, here we present a few.

Bernoilli distribution (l is not announced), an event that can take on of two outcomes. - yes or no. It still has mean and standard deviation. 

Binomial Distribution - Happens when you repeat the Bernoilli dsitribution. - binomial does not mean bimodal!! It is simply a repeated bernolli experiment. - so it is discrete, but looks like a continous normal distribution. 

Normal Distribution is often used as a  normallity assumption.

Logistic Regression - When the output value is dichotomous. 

R squared - the proportion of variability in Y explained by the regression model. It is also just the quare of the correlation r. It makes sense, it makes absolute sense the more the features are correlated with y the more they are likley to explain Y.

RMSE - measure the standard deviation of the residuals, it tells you about the noise in the system the width of the spread. We can
use rmse to predict confidence intervals. **Nice so you can recreate this with RMSE.**  

Forecast +/- 2 * RMSE is the prediction interval - I think it would be good to add this in everything you do. 

R squared is comparative in relative nature not good for absolute comparsion between different tasks. We like higher values to explain more variability.

Apply the log transform to create better prediction, do a transformation that creates linearity on the transform scale.


![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527910134062_file.png)



![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527910134078_file.png)


Presenting the same data on the different scales. On the right side is a log of log transform to present the data differently. When you backtransoform its better. They are presenting the same model on different scales. Good. 

Logistic regression, the ourcome can be viewed as Bernoulli random variables. - S cuveer, never go above 1 or below 0. 


**NB.. You often have to choose distributions to fit our data.**



Sales Forecast are there for multipe reasons.

- Calculate the quanttity and price of produts to manufactore.
- Calulate where the amount of sales would occur
- Track the amount of support staff orsuppliers necessary for the amount of sales 

Four Forecasting Methods - Time Series (Use Past) - Not as accurate, Causal Analytics (Find Characteristics) - Most Accurate, Trial Rate (Surveys, good for new) and Diffusion Models (Analogous Products). I would argue use Causal Analytics + Time Series +  Diffusion Models. - No human interaction. 

Maybe you should aslo track the number of locations over time: [This is causal trying to calculate the sales]

![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527910134115_file.png)







This good look forward models are called median of median models. 

With especially noisy data - median of median works very well. 

I have two types of predictions: using the previous year's visits as prediction and median of previous days. I switch between these two models based on their accuracy in the previous period. I think the code is short and readable, most probably reading it will explain itself better than me.

It's based on the observation that most of the time series are low-traffic, noisy and seemingly very unpredictable (figure 1) while some of them behave quite nicely (figure 2). My main idea was to use Kalman filters to predict well-behaved time series while falling back to a more robust median-of-medians for the bulk of the data.

*Kalman filtering*, also known as linear quadratic estimation (LQE), is an algorithm that uses a series of measurements observed over time, containing statistical noise and other inaccuracies, and produces estimates of unknown variables that tend to be more accurate than those based on a single measurement alone

https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43727


## It makes sense, why do you need the actual past values, instead use the median with some added time series information. There is no reason to run it with all the unique model values as it will pick up enough info.



As part of it I had to create a simple model: using just the median visits of the last 8 weeks, the site, the access method, the week of the day and the day in the week I get a score of 36.78 that would lead to 9th rank


Median Estimation for Future Time Series:

Fibonnaici Time Series for Median.



    # I'm gong to share a solution that I found interesting with you.
    # The idea is to compute the median of the series in different window sizes at the end of the series,
    # and the window sizes are increasing exponentially with the base of golden ratio.
    # Then a median of these medians is taken as the estimate for the next 60 days.
    # This code's result has the score of around 44.9 on public leaderboard, but I could get upto 44.7 by playing with it.
    
    # r = 1.61803398875
    # Windows = np.round(r**np.arange(0,9) * 7)
    Windows = [6, 12, 18, 30, 48, 78, 126, 203, 329]

I have two types of predictions: using the previous year's visits as prediction and median of previous days. I switch between these two models based on their accuracy in the previous period. I think the code is short and readable, most probably reading it will explain itself better than me.



Sequence to Sequence web time series solution - Worth Having a Look at once you have some more processsing power and Ram. 
Good Multi-Steps

https://github.com/sjvasquez/web-traffic-forecasting
https://github.com/Arturus/kaggle-web-traffic/blob/master/how_it_works.md
https://github.com/jfpuget/Kaggle/tree/master/WebTrafficPrediction

Good Single Steps:
https://www.kaggle.com/kcbighuge/predicting-sales-with-a-nested-lstm


ML Competition

This is key:

You described single step prediction model. It will work, but you have to predict 60 days, not one. You can predict one day, refit the model with previous days + predicted day, predict next day, etc, but it would be too slow and inefficient. Of course, you can train 60 RNN's to predict 60 different days, but this would be inefficient too (such model will leave out serial dependency between days).
The real power of RNN's is that you can build generative model, and predict all 60 days at once.

@Arthur, thank you for the clarification. Just wanted to understand this better - when you say predict all 60 days at once do you mean a sequence to sequence type of model where the the model takes in the past some number of days and then trains on the next 60 days in the sequence so that it ultimately outputs a sequence of 60 days?

@Gustavo De Mari, all 'time series' books have approximately the same content. Don't take them too seriously: ARIMA-type models are mostly theoretical and rarely applicable to real-world unstationary data. Bayesian modeling (like Prophet) is much more practical.
This is a good intro to time series prediction: [https://www.otexts.org/fpp](https://www.otexts.org/fpp)

What you are infact looking at is something called generative RNN.

Here is the trick you take as input previous values.

It seems like you are required to creat a sequence to sequence model of sorts.

**TL;DR** this is seq2seq model with some additions to utilize year-to-year and quarter-to-quarter seasonality in data.


There are two main information sources for prediction:

1. Local features. If we see a trend, we expect that it will continue (AutoRegressive model), if we see a traffic spike, it will gradually decay (Moving Average model), if wee see more traffic on holidays, we expect to have more traffic on holidays in the future (seasonal model).
2. Global features. If we look to autocorrelation plot, we'll notice strong year-to-year autocorrelation and some quarter-to-quarter autocorrelation.

The good model should use both global and local features, combining them in a intelligent way.
I decided to use RNN seq2seq model for prediction, because:

1. RNN can be thought as a natural extension of well-studied ARIMA models, but much more flexible and expressive.
2. RNN is non-parametric, that's greatly simplifies learning. Imagine working with different ARIMA parameters for 145K timeseries.
3. Any exogenous feature (numerical or categorical, time-dependent or series-dependent) can be easily injected into the model
4. seq2seq seems natural for this task: we predict next values, conditioning on joint probability of previous values, including our past predictions. Use of past predictions stabilizes the model, it learns to be conservative, because error accumulates on each step, and extreme prediction at one step can ruin prediction quality for all subsequent steps.  → This is of course the only reason sequence to sequence models work.

Instead of seq2seq models you can also train individual models to do the same, for example 60 different models, the issue of this approach is the you do away with the serial dependency between the days. Another way is to feed the predicted value into the testing frame while dropping the last value and iteratively predicting into the future, this is a cheaper exercise but the errors explode. Another approach is to output multiple future time-steps, however this generally also produces bad results as the past predictions are not used to stabilise the model at each step. But I have seen this perform well when seasonality is taken care off and a big enough set. (https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/39395)  For that reason out of these 4 approaches, the only valid model is a seq2seq model.

In saying that a normal LSTM model would perform perfectly well if we just want to predict one period into the future and then the use of a seq2seq model is redundant. 


## Feature engineering

I tried to be minimalistic, because RNN is powerful enough to discover and learn features on its own. Model feature list:

- *pageviews* (spelled as 'hits' in the model code, because of my web-analytics background). Raw values transformed by log1p() to get more-or-less normal intra-series values distribution, instead of skewed one.
- *agent*, *country*, *site* - these features are extracted from page urls and one-hot encoded
- *day of week* - to capture weekly seasonality
- *year-to-year autocorrelation*, *quarter-to-quarter autocorrelation* - to capture yearly and quarterly seasonality strength.
- *page popularity* - High traffic and low traffic pages have different traffic change patterns, this feature (median of pageviews) helps to capture traffic scale. This scale information is lost in a *pageviews* feature, because each pageviews series independently normalized to zero mean and unit variance.
- *lagged pageviews* - I'll describe this feature later
## Feature preprocessing

All features (including one-hot encoded) are normalized to zero mean and unit variance. Each *pageviews* series normalized independently.
Time-independent features (autocorrelations, country, etc) are "stretched" to timeseries length i.e. repeated for each day by `tf.tile()` command.
Model trains on random fixed-length samples from original timeseries. For example, if original timeseries length is 600 days, and we use 200-day samples for training, we'll have a choice of 400 days to start the sample.
This sampling works as effective data augmentation mechanism: training code randomly chooses starting point for each timeseries on each step, generating endless stream of almost non-repeating data.


Encoder is [cuDNN GRU](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/cudnn_rnn/CudnnGRU). cuDNN works much faster (5x-10x) than native Tensorflow RNNCells, at the cost of some inconvenience to use and poor documentation.
Decoder is TF `GRUBlockCell`, wrapped in `tf.while_loop()` construct. Code inside the loop gets prediction from previous step and appends it to the input features for current step.


## Working with long timeseries

LSTM/GRU is a great solution for relatively short sequences, up to 100-300 items. On longer sequences LSTM/GRU still works, but can gradually forget information from the oldest items. Competition timeseries is up to 700 days long, so I have to find some method to "strengthen" GRU memory.
My first method was to use some kind of [*attention*](https://distill.pub/2016/augmented-rnns). Attention can bring useful information from a distant past to the current RNN cell. The simplest yet effective attention method for our problem is a fixed-weight sliding-window attention. There are two most important points in a distant past (taking into account long-term seasonality): 1) year ago, 2) quarter ago.

![from_past](https://github.com/Arturus/kaggle-web-traffic/raw/master/images/from_past.png)


I can just take encoder outputs from `current_day - 365` and `current_day - 90` timepoints, pass them through FC layer to reduce dimensionality and append result to input features for decoder. This solution, despite of being simple, considerably lowered prediction error.
Then I averaged important points with their neighbors to reduce noise and compensate uneven intervals (leap years, different month lengths): `attn_365 = 0.25 * day_364 + 0.5 * day_365 + 0.25 * day_366`
Then I realized that `0.25,0.5,0.25` is a 1D convolutional kernel (length=3) and I can automatically learn bigger kernel to detect important points in a past.
I ended up with a monstrous attention mechanism, it looks into 'fingerprint' of each timeseries (fingerprint produced by small ConvNet), decides which points to attend and produces weights for big convolution kernel. This big kernel, applied to decoder outputs, produces attention features for each prediction day. This monster is still alive and can be found in a model code.
Note, I did'nt used classical attention scheme (Bahdanau or Luong attention), because classical attention should be recalculated from scratch on every prediction step, using all historical datapoints. This will take too much time for our long (~2 years) timeseries. My scheme, one convolution per all datapoints, uses same attention weights for all prediction steps (that's drawback), but much faster to compute.
Unsatisfied by complexity of attention mechanics, I tried to remove attention completely and just take important (year, halfyear, quarter ago) datapoints from the past and use them as an additional features for encoder and decoder. That worked surprisingly well, even slightly surpassing attention in prediction quality. My best public score was achieved using only lagged datapoints, without attention.


  **TL;DR** this is seq2seq model with some additions to utilize year-to-year and quarter-to-quarter seasonality in data. - Interesting, but he did not specifically engineer seasonality into his dataframe. 

Additional important benefit of lagged datapoints: model can use much shorter encoder without fear of losing information from the past, because this information now explicitly contained in features. Even 60-90 days long encoder still gives acceptable results, in contrast to 300-400 days required for previous models. Shorter encoder = faster training and less loss of information

Final predictions were rounded to the closest integer, negative predictions clipped at zero.

- Just notice that they would sometimes produce negative result. 


There are two ways to split timeseries into training and validation datasets:

1. *Walk-forward split*. This is not actually a split: we train on full dataset and validate on full dataset, using different timeframes. Timeframe for validation is shifted forward by one prediction interval relative to timeframe for training.
2. *Side-by-side split*. This is traditional split model for mainstream machine learning. Dataset splits into independent parts, one part used strictly for training and another part used strictly for validation.
![split](https://github.com/Arturus/kaggle-web-traffic/raw/master/images/split.png)


I tried both ways.
Walk-forward is preferable, because it directly relates to the competition goal: predict future values using historical values. But this split consumes datapoints at the end of timeseries, thus making hard to train model to precisely predict the future.
Let's explain: for example, we have 300 days of historical data and want to predict next 100 days. If we choose walk-forward split, we'll have to use first 100 days for real training, next 100 days for training-mode prediction (run decoder and calculate losses), next 100 days for validation and next 100 days for actual prediction of future values. So we actually can use only 1/3 of available datapoints for training and will have 200 days gap between last training datapoint and first prediction datapoint. That's too much, because prediction quality falls exponentially as we move away from a training data (uncertainty grows). Model trained with a 100 days gap (instead of 200) would have considerable better quality.
Side-by-side split is more economical, as it don't consumes datapoints at the end. That was a good news. Now the bad news: for our data, model performance on validation dataset is strongly correlated to performance on training dataset, and almost uncorrelated to the actual model performance in a future. In other words, side-by-side split is useless for our problem, it just duplicates model loss observed on training data.
Resume?
I used validation (with walk-forward split) only for model tuning. Final model to predict future values was trained in blind mode, without any validation.



## Hyperparameter tuning

There are many model parameters (number of layers, layer depths, activation functions, dropout coefficents, etc) that can be (and should be) tuned to achieve optimal model performance. Manual tuning is tedious and time-consuming process, so I decided to automate it and use [SMAC3](https://automl.github.io/SMAC3/stable/) package for hyperparameter search. Some benefits of SMAC3:

- Support for conditional parameters (e.g. jointly tune number of layers and dropout for each layer; dropout on second layer will be tuned only if n_layers > 1)
- Explicit handling of model variance. SMAC trains several instances of each model on different seeds, and compares models only if instances were trained on same seed. One model wins if it's better than another model on all equal seeds.

Contrary to my expectations, hyperparamter search did not found well-defined global minima. All best models had roughly the same performance, but different parameters. Probably RNN model is too expressive for this task, and best model score depends more on the data signal-to-noise ratio than on the model architecture.

Basically find that hyperparam tuning not too important. 

For NN, I didn't explicitly used day of week and week rank. I rather used them to define output indices: the output for the third Tuesday is at the same position, whatever the year I am in. This was key for good predictions as it enables the prediction of weekly patterns. Project and access/agent are one hot encoded, medians are the log1p transformed data. I did not normalize data as mean is close to 0 already, and variance close to 1.


There is a multiplicative and additive decomposition model.


    import statsmodels.api as sm
    # multiplicative
    res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
    #plt.figure(figsize=(16,12))
    fig = res.plot()
    #fig.show()



    # Additive model
    res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
    #plt.figure(figsize=(16,12))
    fig = res.plot()
    #fig.show()


we assume an additive model, then we can write

> yt=St+Tt+Et

where yt is the data at period t, St is the seasonal component at period t, Tt is the trend-cycle component at period tt and Et is the remainder (or irregular or error) component at period t Similarly for Multiplicative model,

> yt=St x Tt x Et

There are multiple tests that can be used to check stationarity.

- ADF( Augmented Dicky Fuller Test)
- KPSS
- PP (Phillips-Perron test)

Let's just perform the ADF which is the most commonly used one.

This removes trends and seasonality, 


    # to remove trend
    from pandas import Series as Series
    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)
    
    # invert differenced forecast
    def inverse_difference(last_ob, value):
        return value + last_ob




# Hierarchical time series:

The [Forecasting: principles and practice](https://www.otexts.org/fpp/9/4) , is the ultimate reference book for forecasting by Rob J Hyndman.
He lays out the fundamentals of dealing with grouped or Hierarchical forecasts. Consider the following simple scenario.

![](https://www.otexts.org/sites/default/files/resize/fpp/images/hts1-550x274.png)


Hyndman proposes the following methods to estimate the points in this hierarchy. I've tried to simplify the language to make it more intuitve.
**Bottom up approach:**

- Predict all the base level series using any method, and then just aggregate it to the top.
- Advantages: Simple , No information is lost due to aggregation.
- Dis-advantages: Lower levels can be noisy

**Top down approach:**

- Predict the top level first. (Eg: predict total sales first)
- Then calculate **weights** that denote the proportion of the total sales that needs to be given to the base level forecast(Eg:) the contribution of the item's sales to the total sales
- There are different ways of arriving at the "weights".
  - **Average Historical Proportions** - Simple average of the item's contribution to sales in the past months
  - **Proportion of historical averages** - Weight is the ratio of average value of bottom series by the average value of total series (Eg: Weight(item1)= mean(item1)/mean(total_sales))
  - **Forecasted Proportions** - Predict the proportion in the future using changes in the past proportions
- Use these weights to calcuate the base -forecasts and other levels

**Middle out:**

- Use both bottom up and top down together.
- Eg: Consider our problem of predicting store-item level forecasts.
  - Take the middle level(Stores) and find forecasts for the stores
  - Use bottoms up approach to find overall sales
  - Dis-integrate store sales using proportions to find the item-level sales using a top-down approach

**Optimal combination approach:**

- Predict for all the layers independently
- Since, all the layers are independent, they might not be consistent with hierarchy
  - Eg: Since the items are forecasted independently, the sum of the items sold in the store might not be equal to the forecasted sale of store or as Hyndman puts it “aggregate consistent”
- Then some matrix calculations and adjustments happen to provide ad-hoc adjustments to the forecast to make them consistent with the hierarchy

Use medians as features.
Use `log1p` to transform data, and `MAE` as the evaluation metric.
XGBoost and deep learning models such as MLP, CNN, RNN work. However, the performance hugely depends on how we create and train models.
For these deep learning models, skip connection works.
**Best trick to me**: clustering these time-series based on the performance of the best model. Then training different models for each cluster.


1. ~450 days to predict 60
2. No one uses relu in recurrent (or similar autoregressive structure) networks, it explodes quickly. Usual activation in this case is tanh(), it's bounded.

RNN readily overfits even on full dataset. Considering this, one RNN for for each timeseries looks like a nonsense. Deep learning usually wants a lot of data, the more data, the better the results. RNN usually trained by batches, one batch = many parallel timeseries, I don't see a problem here.





https://www.drivendata.org/competitions/

You can buy an external GPU at some point.

https://apple.stackexchange.com/questions/277356/machine-learning-on-external-gpu-with-cuda-and-late-mbp-2016?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

SMAPE ignores outliers. It is also invariant if you linearly rescale data.



## Sliding Window Method[¶](https://www.kaggleusercontent.com/kf/3308169/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kEzBvspJ39abi240qg_DAQ.elaZdUQ0IgjbT1hOEYRgDTc3_1eP-vZK_-f-HF-91nwjpOHpo0rxTq_F4lpjt1diIEtb_lpU6y7pQruNPQZOHHpfYjLWJWqA5HjLk8TojqyfLP4hxWpJay7Ls0AvsOWB_RCj9onTIoT8d4xBc9ZPpHqEPlIdhmAvqbq51OzUbpi5z3JG8Jt75AgHI7yRHkes.rzPh39BnBh00YjOXIRuivg/__results__.html#Sliding-Window-Method)

This method can be used to transform what would be a time series forecast into a supervised learning prediction. The key thing we want here is to use the value of `item_price` on the previous date as the predictor variable to the current date. For simplicity, I'll call the previous `item_price` as `item_price_x`.

    revenueByMonth["item_price_x"] = revenueByMonth["item_price"].shift(1)
    
    slideRevenueByMonth = revenueByMonth.drop(columns = "monthlyDate")
    
    sns.lmplot(data = slideRevenueByMonth, x = "item_price_x", y = "item_price", ci=False);
    
    slideRevenueByMonth.head()


LSTM only just normally beats the persistence models, you should also try median and a few quick eqasy ones, before you try anythin fancy. 

You have actually seen multiple graps doing just that - Predicting 3 days ahead for LSTM then it has to regroup. The problem with using machine learning models is that you go from using a time series to a suervised learning method - that is what I have headr people quote. 

mean absolute error (MAE). Unlike other top solutions, my main tool was polynomial autoregression. (Rather than least squares regression, I used median regression, which minimizes MAE.)


RNN and CNN will give you the best solution in a large time sereis. 
LTSM Multivariate - There is things you can do to further improve your LTSM preprocessing.


- Making all series stationary with differencing and seasonal adjustment.
- Providing more than 1 hour of input time steps.

This last point is perhaps the most important given the use of Backpropagation through time by LSTMs when learning sequence prediction problems.


I would add that the LSTM [does not appear to be suitable for autoregression type problems](https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/) and that you may be better off exploring an MLP with a large window.

I model the problem with no timesteps and lots of features (multiple obs at the same time).
I found that if you frame the problem with multiple time steps for multiple features, performance was worse. Basically, we are using the LSTM as an MLP type network here.
LSTMs are not great at autoregression, but this post was the most requested I’ve ever had.

Correct me if I am wrong but the whole point of RNN+LSTM learning over time(hidden states depending on past values) goes moot here.
Essentially, this is just an autoregressive neural network. There is no storage of states over time. - Correct. 

The predictions look like persistence.


Jason, what am I missing, looking at your plot of the most recent 100 time steps, it looks like the predicted value is always 1 time period after the actual? If on step 90 the actual is 17, but the predicted value shows 17 for step 91, we are one time period off, that is if we shifted the predicted values back a day, it would overlap with the actual which doesn’t really buy us much since the next hour prediction seems to really align with the prior actual. Am I missing something looking at this chart?


This is what a persistence forecast looks like, that value(t) = value(t-1).

The next step is to make persistence forecasts.
We can implement the persistence forecast easily in a function named *persistence()* that takes the last observation and the number of forecast steps to persist. This function returns an array containing the forecast.
# make a persistence forecast def persistence(last_ob, n_seq): return [last_ob for i in range(n_seq)]

| 1
2
3 | # make a persistence forecast
def persistence(last_ob, n_seq):
return [last_ob for i in range(n_seq)] |

We can then call this function for each time




https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM


You can also build a LSTM nework for regression and classification model.
Prophet.


We predict values with Prophet by passing in the dates for which we want to create a forecast. If we also supply the historical dates (as in our case), then in addition to the prediction we will get an in-sample fit for the history. Let's call the model's predict method with our future dataframe as an input:

In the resulting dataframe you can see many columns characterizing the prediction, including trend and seasonality components as well as their confidence intervals. The forecast itself is stored in the yhat column.




What is so nice about arima it is so shit it can’t overfit. 

Types.
One Input

- Holt-Winters
- SARIMA
- LTSM
- Tree Ensembles
- Prophet

Multiple Inputs

- SARIMAX
- LSTM
- Tree Ensembles
- Propher

I fundamentally want to cover four models,  ARIMA, Holt Winters, LTSM, Prophet and CART models. You will implement these models and I identify the best performance you can get from them. 

**Good Anomaly Detection - I like what I see**
https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection



![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527306265875_file.png)



See I knew there was such a thing:

The **TBATS model** is a time series **model** for series exhibiting multiple complex seasonalities.

Holt Winters is not really the econometrics appraoch, their approach starts with knowledge of stationarity

x(t)=ρ*x(t−1)+e(t) Increasing the value of ρ and recursively accumulating or running, it becomes more autocorrelated. 

Finally, the value of ρ equal to 1 gives us a random walk process — non-stationary time series.

This happens because after reaching the critical value the series x(t)=ρ*x(t−1)+e(t) does not return to its mean value.

You can doe a Dickey-Fuller unit root tests - testing how significant the stationarity it. It becomes fractalised then it is not stationary anymore. 

If ρ=1 then the first difference gives us stationary white noise e(t). This fact is the main idea of the Dickey-Fuller test for the stationarity of time series (presence of a unit root) Of course because the x term falls away and stationarity remains. If you take the first difference and it is still not stationary there migth be some higher effects involve. 

You can basically just keep on finding higher order derivatives untill just the e error compnent remain. 


A lot on ARIMA:

There is clearly seasonality by the hour, but there is seasonality by the month too. However, we will not be dealing with that.
Surprisingly, initial series are stationary, Dickey-Fuller test rejected null hypothesis that a unit root is present. Actually, it can be seen on the plot itself — we don’t have a visible trend, so mean is constant, variance is pretty much stable throughout the series. The only thing left is seasonality which we have to deal with before modelling.
Shit what a nice calucaltion, then only the supposed trend will remain the level and season differnce will dissapear.
To do so let’s take “seasonal difference” which means a simple subtraction of series from itself with a lag that equals the seasonal period.
NB!! First differnces is not first derivative, first difference is a lagged formulaic deduction to try and reengineer the error.! The thing is you should do the dicky fuller on the normal time series, howver the dicky fuller is your trend identifier and autocorrelation identifer, it is not going to help you spot seasonality and once you have infact remove seasonality by substracting the difference in expected seasonal lags, then you redo the test over that series. Normally then you would see some more pronounced trends, in that case you can simply take first differences.
ads_diff = ads.Ads - ads.Ads.shift(24)
tsplot(ads_diff[24:], lags=60)
That’s better, visible seasonality is gone, however autocorrelation function still has too many significant lags. To remove them we’ll take first differences — subtraction of series from itself with lag 1
ads_diff = ads_diff - ads_diff.shift(1)
tsplot(ads_diff[24+1:], lags=60)
Perfect! Our series now look like something undescribable, oscillating around zero, Dickey-Fuller indicates that it’s stationary and the number of significant peaks in ACF has dropped. We can finally start modelling!. ACF is probably autocorrelation.
The autocorrelation function (ACF) measures how a series is correlated with itself at different lags.
Once your series is stationary, the ACF can help guide your choice of moving average lags. Also it's a good way to confirm any trend, for a positive trend you'll see the ACF taking ages to die out.
The partial autocorrelation function can be interpreted as a regression of the series against its past lags. It helps you come up with a possible order for the auto regressive term. The terms can be interpreted the same way as a standard linear regression, that is the contribution of a change in that particular lag while holding others constant.
Naturally 0 is important as it provides the level of start.
AR(p) — autoregression model, i.e., regression of the time series onto itself. Basic assumption — current series values depend on its previous values with some lag (or several lags). The maximum lag in the model is referred to as p. To determine the initial p you need to have a look at PACF plot — find the biggest significant lag, after which most other lags are becoming not significant.
MA(q) — moving average model. Without going into detail it models the error of the time series, again the assumption is — current error depends on the previous with some lag, which is referred to as q. Initial value can be found on ACF plot with the same logic.
AR(p) + MA(q) = ARMA(p,q)
What we have here is the Autoregressive–moving-average model! If the series is stationary, it can be approximated with those 4 letters. Shall we continue?
I(d)— order of integration. It is simply the number of nonseasonal differences needed for making the series stationary. In our case it’s just 1, because we used first differences.
Adding this letter to four previous gives us ARIMA model which knows how to handle non-stationary data with the help of nonseasonal differences. Awesome, last letter left!
S(s) — this letter is responsible for seasonality and equals the season period length of the series
After attaching the last letter we find out that instead of one additional parameter we get three in a row — (P,D,Q)

    P — order of autoregression for seasonal component of the model, again can be derived from PACF, but this time you need to look at the number of significant lags, which are the multiples of the season period length, for example, if the period equals 24 and looking at PACF we see 24-th and 48-th lags are significant, that means initial P should be 2.
    Q — same logic, but for the moving average model of the seasonal component, use ACF plot
    D — order of seasonal integration. Can be equal to 1 or 0, depending on whether seasonal differences were applied or not
    
        p is most probably 4, since it’s the last significant lag on PACF after which most others are becoming not significant.
    d just equals 1, because we had first differences
    q should be somewhere around 4 as well as seen on ACF
    P might be 2, since 24-th and 48-th lags are somewhat significant on PACF
    D again equals 1 — we performed seasonal differentiation
    Q is probably 1, 24-th lag on ACF is significant, while 48-th is not
# Shit I can do SARIMA no issue
## Using the above Heuristics, you can test multiple values and see what is better, unlike HOlt-Winters these
## Values can differ tremendously hence the previous deliveration.


Prophet

So far we have used Prophet with the default settings and the original data. We will leave the parameters of the model alone. But despite this we still have some room for improvement. In this section, we will apply the Box–Cox transformation to our original series. Let’s see where it will lead us.
A few words about this transformation. This is a monotonic data transformation that can be used to stabilize variance. We will use the one-parameter Box–Cox transformation, which is defined by the following expression:

    We will need to implement the inverse of this function in order to be able to restore the original data scale. It is easy to see that the inverse is defined as:

**This COX is just worth it if there is a lot of variation differences**
#i.e. irregular amplitudes over time, which was not my case
def inverse_boxcox(y, lambda_):
return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
train_df2 = train_df.copy().set_index('ds')
from scipy import stats
train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])
train_df2.reset_index(inplace=True)
m2 = Prophet()
m2.fit(train_df2)
future2 = m2.make_future_dataframe(freq='H',periods=prediction_size)
forecast2 = m2.predict(future2)
for column in ['yhat', 'yhat_lower', 'yhat_upper']:
forecast2[column] = inverse_boxcox(forecast2[column],
lambda_prophet)
cmp_df2 = make_comparison_dataframe(ads2, forecast2)
for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():
print(err_name, err_value)
show_forecast(cmp_df, prediction_size, len(cmp_df), 'No transformations')
show_forecast(cmp_df2, prediction_size, len(cmp_df), 'Box–Cox transformation')
We have taken a look at Prophet, an open-source forecasting library that is specifically targeted at business time series. We have also done some hands-on practice in time series prediction.
As we have seen, the Prophet library does not make wonders, and its predictions out-of-box are not ideal. It is still up to the data scientist to explore the forecast results, tune model parameters and transform data when necessary.





**Tree based models don't extrapolate.** Tree models don't extrapolate, so if you want to use them for extrapolative forecasts, you need to create features in which the extrapolation is implicit. For example, you can fit a simple linear model with time variables and use the predictions as a feature. Or use the residuals as your target. Or use rates of change as features. And so on. With appropriate feature engineering, I think tree models can still be quite helpful. As often, feature engineering is critical.


This  document contains the primary tools an organisation might want to use for time series prediction. Inherent in time series forecasting is questions as to what will happen with an underlying metric a day, week month in advance. Depending on the context and question, certain tools will be better for the job. 

A quick recap of some important terms, the R-squared (R^2) is the percentage of variance explained by the model.

Errors: 
First do not get throw over about error metrics, i.e. measuring the difference between the model predicted and actual values. As you can imagine there is endless possibilities, should you take the absolute differences, squared differences, logarithmically squared differences etc. You can pretty much make any error metrics up such as the root of the cubed logarithm and call it mean root cubed logarithmic error (MRCLE). The essence hiding behind these error metrics is that the larger the measure the worse your prediction model. Without being to flippant, different error measure do have some justifiable purpose, sometimes. 

The mean absolute error (MAR) is the average difference between the predicted units versus the actual units across all predictions. Similarly you can use the median absolute error (MedAR), which is robust to outliers. Then  of course get the case where you don’t want to ignore outliers and instead penalise them, this is the mean squared error (MSE). The errors get penalised as a consequence of being squared and we know that when a higher value gets squared it becomes exponentially larger than the smaller value counterparts, hence penalisation of outliers. 

Two other errors worth mentioning is the mean squared logarithmic error (MSLE). The difference between this method and before mentioned methods is that this measure tries to decrease the inherent error disadvantage experiened in large number comparisons. You don’t necessarily want to penalise differences just because the constituents to the difference equation are large so that even a small deviation lead to an inevitably large value. And another value you might hear of is any of the above measures in percentage terms. Such as the mean absolute error in percentage terms (MAPE). A measure you see nowhere is mean logarithmic error (MLE) which is a shame, as I think it is one of the most easily understood and effective error measures. 


    # Importing everything from above
    
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
    from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


Moving averages are another time series concept. Moving averages (MA) can best be understood through the concept of momentum. Do you think yesterday’s price will be the same as today, well then simply create a Moving averages model with a window of one day. Moving averages is calculated on a time series of the actual values and not on predicted values. So the predicted value or Moving averages will always be yesterdays actual price, if instead you used past Moving averages predictions to inform future Moving averages predictions you would always predict the same value with a Moving averages window of one day. For what reason you might want to incorporate such recursive formula, I don’t know. 

You can of course extend the number of days backwards as far as is possible. The moving average technique is not really suitable to prediction as it under represents linear growth trends as it assumes the past is more important than an extrapolation of the future. The Moving average’s intended purpose is react slowly and conservatively to underlying changes in the data.  But moving averages do serve an important purpose and that they smooth time series to get rid of noise which helps highlights trends. The wider the window of days the smoother the trend. 

With Moving averages you can create a basic anomaly detection tool by creating confidence bands around the moving average. Anomaly detection is nothing but unsuspecting outlier detection, not to much too much thought should be put into it. The problem with such simple anomaly detection approach is that is does not account for seasonality. And stay clear of thinking seasonality is season based, instead seasonality occurs where there is a systematic difference in between categories of whichever time units, be it months, hours, days or minutes. And moving averages while highlighting the overall trend, simply ignores these systematic differences as it averages over them i.e. actually tries to ignore it best it can. 

So moving averages can simply be extended to weighted averages which is a simple modifications by which observation have different weights, such that more recent observation have greater weights, this would ensure that some of the seasonal trends get picked up, it still doesn’t fix the issue completely. A functionally simple way of creating weighted averages is using an exponential function, where we decrease the weights of observations when we move further back in history, called exponential smoothing or exponential moving average (EMA). The exponentiality is hiding in the recursivity of the function. Within this function there is a parameter alpha that you can adjust, the higher this value the more it discounts historical values in favour of current values. Unlike the simple moving average, the exponential moving average tends to take all observations into account. So for the simple moving average we adjust the window in days and in the exponential moving average, we adjust the alpha in percentage terms. For daily use all you really have to know is that exponential moving averages are more predictive in nature and out more likely to capture the underlying seasonal differences. 

So it is great that we can predict one time point in the future and obtain a nice smooth function, but we might want to predict two periods in the future without simply relying on a two period underlying series or recursive predictions. Instead what we can do is decompose a series into two components a level and a trend. The level is the intercept and the trend is the slope. The final prediction is the sum of the model values, the model that predicts the level and the model that predicts the trend. This is called double exponential smoothing (DEMA), it is a function that uses two constants as to be better at handling trends. In this scenario, you have two parameters the alpha and the beta. Sometimes I feel that these moving averages just have to be shifted to the left. There is a need for a another parameter that seeks to decrease the difference between the generated moving average series to perfectly sit under the actual pattern. The issue is that this would not provide a timely measure for prediction and anomaly detection but would instead be a good measure to retrospectively identify anomalies with the addition of confidence bands. Back to the parameters alpha and beta, the bigger either of these values the more weight is given to recent observations. The previous shift is actually not a bad way to describe the beta parameter, it conceptually works. The issue is there is still stystematic differences accross usnits and this can be removed using some seasonal adjustements. Therefore less smoothing. A lower alpha creates a smoother trend and a lower beta creates a smoother smoothed trend. Both are smoothing parameters, that is all you have to know ever. Combination of these values can produce awkward results, so it is better to choose the parameters automatically and to forget that they exists apart for this task. The only difference between single and double exponential averages is that DEMA helps solve for the lagging issue brining the moving average line closer to the current fluctuations in price. DEMA = 2*EMA - EMA(EMA), where the current EMA is a function of the EMA factor. DEMA essentially places even more weight to the recent data, bringing the DEMA line into closer correlation with the current price. 

Again, just like the different error measures, do not get too confused with moving averages the concept is simple, what is the best generalisable time series that can be extended into the future to decrease the error on average. These different versions is simply the acknowledgement of the actual existence of different constituents  to the underlying data. The double exponential smoothing technique is the best technique to approximate the level and trend, but it still absorbs the seasonality event though it is better with seasonality than the previous techniques. Again anyone can create their own moving averages with their own specifications. 


![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527292180423_file.png)


God, if that isn’t enough, there is something like triple exponential smoothing. Also known as Holt-Winters. The problem with the double exponential smoothing is that the seasonality still exists but it is better accounted for in the trends than normal exponential smoothing and simple moving average smoothing. The idea with Holt-Winters is to add another component, seasonality. Some time series simply don’t have seasonality and in that case Holt-Winters would over-extend your use case. The seasonal component is quite smart as it explain the repeated variations around the intercept and trend. I wonder if such process can be repeated recursively until all repeated variations gets removed, as I am sure the first go will not remove all repeated variation. No this can’t be done as you are bounded by a unit. However, if you have increasingly smaller units, from month to second, you can create a more reliable model on the second level, but not on the month level as it doesn’t add information to that unit of measurement. I do feel that you can repeat the intercept, trend, seasonality process again on the remaining residuals until the errors are fully random. I haven’t see this done before. 

The seasonal component simply increases or decreases on the intercept and the trend series remain the same. The seasonal component is smoothed through all the available seasons, for example, if we have a Monday component then it will only be averaged with other Mondays. To predict one step ahead you need a simple or exponential moving average, to predict two periods ahead you can use double exponential average, to predict n units ahead you can now use all three components of the triple exponential smoothing, this allows you to essentially do long term prediction. You always have to give the number of seasonal units, quarters, 4, day, 7, month, 12. 

And then of course we can use time-series cross-validation to select parameters - There’s nothing unusual here, as always we have to choose a loss function suitable for the task, that will tell us how close the model approximates data. Then using cross-validation we will evaluate our chosen loss function for given model parameters, calculate gradient, adjust model parameters and so forth, bravely descending to the global minimum of error. The best technique is to use cross validation on a rolling basis. 


![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527295758635_file.png)


In the Holt-Winters model, as well as in the other models of exponential smoothing, there’s a constraint on how big smoothing parameters could be, each of them is in the range from 0 to 1, therefore to minimize loss function we have to choose an algorithm that supports constraints on model parameters, in our case — Truncated Newton conjugate gradient. 

Good I fundamentally like these model approaches. Stationarity means that the mean and variance i.e. statistical properties do not change over time. A stationary series, is a series of random errors.


![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527296288365_file.png)



![](https://d2mxuefqeaa7sj.cloudfront.net/s_FDBE1F8DF66BBB02781D639B3B0331BBCBC5F95BF95BD9CA94C500F01238FDA6_1527296448267_file.png)


So why stationarity is so important? Because it’s easy to make predictions on the stationary series as we assume that the future statistical properties will not be different from the currently observed. Weak-sense stationarity or second-order means only two of the three components are constant. 

Random walk is not white noise, random walk is initialised by random noise or randomness after which the new value in the generation process depends on the previous value, so that it recursively gets generated, however, it still doesn’t follow any sicernable pattern. The first difference of a random walk will give us white noise. And that is essentially the idea of the dickey-fuller test, the identification of  a unit root. 

iid means two things even though it has an acronym that makes it look like two three things. iid says that to be truly random, a sequence of variables have to be independent - i.e. no autocorrelation and should be identically distributed, i.e. same mean and variance. An iid time series is stationary.  A random walk is the recursive accumulation of white noise. This is a random walk process, *xt*+1=*xt*+*εt*+1, whereas *εt is the random noise. White noise can’t be a random walk and a random walk can’t be white noise because* *Var*(*xt*+1)=*Var*(*xt*)+*Var*(*εt*+1) is stricly increasing while the variance of a white noise is constant. Weak white noise is stationary up to order two, strong form white noise is strongly stationary. Regarding the relationship between white noise and a random walk, I would put it this way: a random walk is integrated white noise. Or to put it in quant finance terms: white noise is like the daily changes in the S&P in points, a random walk is the S&P daily level itself. What is interesting is that stationarity can have autocorrelation,  as long as the autocorrelation coefficient is random to the previous value.

If we can get stationary series from non-stationary using the first difference we call those series integrated of order 1. We’ve got to say that the first difference is not always enough to get stationary series. In such cases the augmented Dickey-Fuller test is used that checks multiple lags at once.

It is important to note that we can fight non-stationarity using different approaches — various order differences, trend and seasonality removal, smoothing, also using transformations like Box-Cox or logarithmic.

Well there is a fundamental difference between regression and double exponential smoothing. Regression is identifying the relationship of the variable with some other exogenous variable, whereas exponential smoothing is using the variable itself to predict the future. 

Surprisingly, initial series are stationary, Dickey-Fuller test rejected null hypothesis that a unit root is present.

The only thing left is seasonality which we have to deal with before modelling. To do so let’s take “seasonal difference” which means a simple subtraction of series from itself with a lag that equals the seasonal period.

A few words about the model. Letter by letter we’ll build the full name — **SARIMA(*****p,d,q*****)(*****P,D,Q,s*****)**, Seasonal Autoregression Moving Average model:


- **AR(*****p*****)** — autoregression model, i.e., regression of the time series onto itself. Basic assumption — current series values depend on its previous values with some lag (or several lags). The maximum lag in the model is referred to as ***p***. To determine the initial ***p*** you need to have a look at PACF plot — find the biggest significant lag, after which **most** other lags are becoming not significant.
- **MA(*****q*****)** — moving average model. Without going into detail it models the error of the time series, again the assumption is — current error depends on the previous with some lag, which is referred to as ***q***. Initial value can be found on ACF plot with the same logic.

Let’s have a small break and combine the first 4 letters:
**AR(*****p*****) + MA(*****q*****) = ARMA(*****p,q*****)**
What we have here is the Autoregressive–moving-average model! If the series is stationary, it can be approximated with those 4 letters. Shall we continue?

**I(*****d*****)**— order of integration. It is simply the number of nonseasonal differences needed for making the series stationary. In our case it’s just 1, because we used first differences.

Adding this letter to four previous gives us **ARIMA** model which knows how to handle non-stationary data with the help of nonseasonal differences. Awesome, last letter left!

- **S(*****s*****)** — this letter is responsible for seasonality and equals the season period length of the series

After attaching the last letter we find out that instead of one additional parameter we get three in a row — ***(P,D,Q)***


- ***P*** — order of autoregression for seasonal component of the model, again can be derived from PACF, but this time you need to look at the number of significant lags, which are the multiples of the season period length, for example, if the period equals 24 and looking at PACF we see 24-th and 48-th lags are significant, that means initial ***P*** should be 2.
- ***Q*** — same logic, but for the moving average model of the seasonal component, use ACF plot
- ***D*** — order of seasonal integration. Can be equal to 1 or 0, depending on whether seasonal differences were applied or not

Now, knowing how to set initial parameters, let’s have a look at the final plot once again and set the parameters:

    tsplot(ads_diff[24+1:], lags=60)


- ***p*** is most probably 4, since it’s the last significant lag on PACF after which most others are becoming not significant.
- ***d*** just equals 1, because we had first differences
- ***q*** should be somewhere around 4 as well as seen on ACF
- ***P*** might be 2, since 24-th and 48-th lags are somewhat significant on PACF
- ***D*** again equals 1 — we performed seasonal differentiation
- ***Q*** is probably 1, 24-th lag on ACF is significant, while 48-th is not


Small lyrical digression again. Often in my job I have to build models with the only principle guiding me known as [*fast, good, cheap*](http://fastgood.cheap/). That means some of the models will never be “production ready” as they demand too much time for the data preparation (for example, SARIMA), or require frequent re-training on new data (again, SARIMA), or are difficult to tune (good example — SARIMA), so it’s very often much easier to select a couple of features from the existing time series and build a simple linear regression or, say, a random forest. Good and cheap.

Maybe this approach is not backed up by theory, breaks different assumptions (like, Gauss-Markov theorem, especially about the errors being uncorrelated), but it’s very useful in practice and quite frequently used in machine learning competitions.

**Feature exctraction**
Alright, model needs features and all we have is a 1-dimentional time series to work with. What features can we exctract?
**Lags of time series, of course**
**Window statistics:**

- Max/min value of series in a window
- Average/median value in a window
- Window variance
- etc.

**Date and time features:**

- Minute of an hour, hour of a day, day of the week, you get it
- Is this day a holiday? Maybe something special happened? Make it a boolean feature

**Target encoding**
**Forecasts from other models** (though we can lose the speed of prediction this way)
Let’s run through some of the methods and see what we can extract from our ads series
**Lags of time series**
Shifting the series ***n*** steps back we get a feature column where the current value of time series is aligned with its value at the time ***t−n***. If we make a 1 lag shift and train a model on that feature, the model will be able to forecast 1 step ahead having observed current state of the series. Increasing the lag, say, up to 6 will allow the model to make predictions 6 steps ahead, however it will use data, observed 6 steps back. If something fundamentally changes the series during that unobserved period, the model will not catch the changes and will return forecasts with big error. So, during the initial lag selection one has to find a balance between the optimal prediction quality and the length of forecasting horizon.

Well, simple lags and linear regression gave us predictions that are not that far from SARIMA in quality. There are lot’s of unnecessary features, but we’ll do feature selection a bit later. Now let’s continue engineering!

Why not try XGBoost now?

Here is the winner! The smallest error on the test set among all the models we’ve tried so far.

Yet this victory is deceiving and it might not be the brightest idea to fit xgboost as soon as you get your hands over time series data. Generally tree-based models poorly handle trens in data, compared to linear models, so you have to detrend your series first or use some tricks to make the magic happen. Ideally — make the series stationary and then use XGBoost, for example, you can forecast trend separately with a linear model and then add predictions from xgboost to get final forecast.

Difference between Holt-Winters and ARIMA. Holt-Winters has three parameters, so it's simple, but they're basically smoothing factors so it doesn't tell you much if you know them. ARIMA has more parameters, and some of them have some intuitive meaning, but it still doesn't tell you much. I have seen people with different data sets compare results from both algorithms and get different results. In some cases the Holt-Winters algorithms gives better results than the ARIMA and in others cases it is the other way around. I don't think you will find and an explicit answer on when to use one over the other.


I will have this as my belief for now.

They're different things. ARIMA is used for modeling the level of the series (given the past data, what's my forecast for the variable itself next period?). ARCH models (GARCH is just a more general variant of ARCH) are used for modeling volatility of shocks to the series (given the past data, what's my forecast of the shock volatility next period?). Since more volatile shock also mean more uncertainty in my forecast of the variable level, ARCH part is important if we suspect volatility changes over time. It's also important for applications such as VaR obviously, since you need to estimate the dispersion of the distribution of your losses.

You can have ARIMA model without ARCH part (then your volatility is assumed constant), you could also have ARCH without ARIMA (so there would be no dynamics in the level of your variable, just uncorrelated shocks but with varying volatility), or you can combine both to model jointly both the first and second moment.

MAPE is widely used as a measure of prediction accuracy because it expresses error as a percentage and thus can be used in model evaluations on different datasets.


    snaive (seasonal naive) is a model that makes constant predictions taking into account information about seasonality. For instance, in the case of weekly seasonal data for each future Monday, we would predict the value from the last Monday, and for all future Tuesdays we would use the value from the last Tuesday and so on.


I like this idea, you dont have to smooth you can just bucker resampe. 

To reduce the noise, we will resample the post counts down to weekly bins. Besides *binning*, other possible techniques of noise reduction include [Moving-Average Smoothing](https://en.wikipedia.org/wiki/Moving_average) and [Exponential Smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing), among others.
We save our downsampled dataframe in a separate variable because further in this practice we will work only with daily series:

    weekly_df = daily_df.resample('W').apply(sum)

This downsampled chart proves to be somewhat better for an analyst’s perception.


Now, we’re going to omit the first few years of observations, up to 2015. First, they won’t contribute much into the forecast quality in 2017. Second, these first years, having very low number of posts per day, are likely to increase noise in our predictions, as the model would be forced to fit this abnormal historical data along with more relevant and indicative data from the recent years.

To sum up, from visual analysis we can see that our dataset is non-stationary with a prominent growing trend. It also demonstrates weekly and yearly seasonality and a number of abnormal days in each year.



LSTMs can almost seamlessly model problems with multiple input variables. All we need is a 3D input vector that needs to be fed into the input shape of the LSTM. So long as we figure out a way to convert all our input variables to be represented in a 3D vector form, we are good use LSTM. This adds a great benefit in time series forecasting, where classical linear methods can be difficult to adapt to multivariate or multiple input forecasting problems (A side note here for multivariate forecasting — keep in mind that when we use multivariate data for forecasting, then we also need “future multi-variate” data to predict the future outcome!)


In general, while using LSTM’s, I found that they offer lot of flexibility in modelling the problem — meaning we have a good control over several parameters of the time series. In particular we can —
Flexibility to use several combinations of seq2seq LSTM models to forecast time-series — many to one model(useful when we want to predict at the current timestep given all the previous inputs), many to many model (useful when we want to predict multiple future time steps at once given all the previous inputs) and several other variations on these. We can customize several things for example — the size of look-back window to predict at the current step, the number of time steps we want to predict into the future, feeding the current prediction back into the window to make prediction at the next time step (this technique also known as moving-forward window) and so on.

However, the above is usually not a realistic way in which predictions are done, as we will not have all the future window sequences available with us.
2. So, if we want to predict multiple time steps into the future, then a more realistic way is to predict one time step at a time into the future and feed that prediction back into the input window at the rear while popping out the first observation at the beginning of the window (so that the window size remains same). Refer to the below code snippet that does this part — (the comments in the code are self explanatory if you go through the code in my github link that I mentioned above )

However, the issue is that you would be running up an accumulation of errors. 

So there is two ways for LTSM - you either eat the prediction or you create new models. 


Different Type of Forecasts

## 1. Direct Multi-step Forecast Strategy

The direct method involves developing a separate model for each forecast time step.


## 2. Recursive Multi-step Forecast

The recursive strategy involves using a one-step model multiple times where the prediction for the prior time step is used as an input for making a prediction on the following time step


## 3. Direct-Recursive Hybrid Strategies

The direct and recursive strategies can be combined to offer the benefits of both methods.


## 4. Multiple Output Strategy

The multiple output strategy involves developing one model that is capable of predicting the entire forecast sequence in a one-shot manner.
In the case of predicting the temperature for the next two days, we would develop one model and use it to predict the next two days as one operation.


LSTM works better if we are dealing with huge amount of data and enough training data is available, while ARIMA is better for smaller datasets (is this correct?)
ARIMA requires a series of parameters `(p,q,d)` which must be calculated based on data, while LSTM does not require setting such parameters. However, there are some hyperparameters we need to tune for LSTM.

**ARIMA**
You are incorrect in your assessment that ARIMA requires stationary time series to forecast on. [Non-seasonal ARIMA has three input values to help control for smoothing, stationarity, and forecasting](http://www.slideshare.net/21_venkat/arima-26196965) ARIMA(p,d,q), where:

- p is the number of autoregressive terms,
- d is the number of nonseasonal differences needed for stationarity, and
- q is the number of lagged forecast errors in the prediction equation.

By contrast [seasonal ARIMA has six input values](http://people.duke.edu/~rnau/411arim.htm#pdq) ARIMA(p,d,q,P,D,Q), where:

- P is the number of seasonal autoregressive terms,
- D is the number of seasonal differences, and
- Q is the number of seasonal moving-average terms.

Subject to the qualifying statements above, I suggest playing with seasonal ARIMA to get a feel for the intricacies involved in smoothing, de-seasoning, de-trending, de-noiseing, and forecasting.

Mechanically, ARIMAX and ARIMA do not differ. In fact in StatsModels (and other software), we don’t even make a distinction between the two models. They are so similar that making a distinction is almost petty. Yet, the distinction exists, you can go look it up. So the question is why?
The answer is that by trying to combine two time-series in a regression opens you up to all kinds of new mistakes that you can make. Yeah, univariate time-series analysis has different things, like ensuring that your time-series is stationary. But multivariate time-series you start entering the weird world of causality bending. (Causality bending is my own term for what is going on here). Let’s point out the basic rules of causality.

prophet multiple.


https://github.com/facebook/prophet/issues/101


[facebook/prophet#101](https://github.com/facebook/prophet/issues/101)


This is actual SARIMA with actual datess:


def plotSARIMA(series, model, n_steps, plot_intervals=False):
"""
Plots model vs predicted values

        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
    
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    #preds, stderr, ci  = best_model.forecast(start = data.shape[0], end = data.shape[0]+n_steps)
    
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    
    forecast = data.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
    
    plt.figure(figsize=(15, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    
    plt.plot(forecast, color='r', label="model")
    
    if plot_intervals:
        plt.plot(forecast + np.sqrt(model.conf_int().loc["sigma2",0]), "g--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(forecast - np.sqrt(model.conf_int().loc["sigma2",0]), "g--", alpha=0.5) 
    
    
    plt.axvspan(data.index[-30], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True);

plotSARIMA(ads, best_model, (50))
plotSARIMA(ads, best_model, (24), plot_intervals=True)


  Some More LTSM Time Series Stuff:

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib
import numpy
from numpy import concatenate

# frame a sequence as a supervised learning problem

def timeseries_to_supervised(data, lag=1):
df = DataFrame(data)
columns = [df.shift(i) for i in range(1, lag+1)]
columns.append(df)
df = concat(columns, axis=1)
return df

# create a differenced series

def difference(dataset, interval=1):
diff = list()
for i in range(interval, len(dataset)):
value = dataset[i] - dataset[i - interval]
diff.append(value)
return Series(diff)

# invert differenced value

def inverse_difference(history, yhat, interval=1):
return yhat + history[-interval]

# scale train and test data to [-1, 1]

def scale(train, test):
# fit scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
# transform train
train = train.reshape(train.shape[0], train.shape[1])
train_scaled = scaler.transform(train)
# transform test
test = test.reshape(test.shape[0], test.shape[1])
test_scaled = scaler.transform(test)
return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value

def invert_scale(scaler, X, yhat):
new_row = [x for x in X] + [yhat]
array = numpy.array(new_row)
array = array.reshape(1, len(array))
inverted = scaler.inverse_transform(array)
return inverted[0, -1]

# fit an LSTM network to training data

def fit_lstm(train, batch_size, nb_epoch, neurons, timesteps):
X, y = train[:, 0:-1], train[:, -1]
X = X.reshape(X.shape[0], timesteps, 1)
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(nb_epoch):
model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
model.reset_states()
return model

# make a one-step forecast

def forecast_lstm(model, batch_size, X):
X = X.reshape(1, len(X), 1)
yhat = model.predict(X, batch_size=batch_size)
return yhat[0,0]

# run a repeated experiment

def experiment(repeats, series, timesteps):
# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, timesteps)
supervised_values = supervised.values[timesteps:,:]
# split data into train and test-sets
train, test = supervised_values[0:-12, :], supervised_values[-12:, :]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
# run experiment
error_scores = list()
for r in range(repeats):
# fit the base model
lstm_model = fit_lstm(train_scaled, 1, 500, 1, timesteps)
# forecast test dataset
predictions = list()
for i in range(len(test_scaled)):
# predict
X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
yhat = forecast_lstm(lstm_model, 1, X)
# invert scaling
yhat = invert_scale(scaler, X, yhat)
# invert differencing
yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
# store forecast
predictions.append(yhat)
# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('%d) Test RMSE: %.3f' % (r+1, rmse))
error_scores.append(rmse)
return error_scores
def parser(x):
return datetime.strptime('190'+x, '%Y-%b')
series = read_csv('data/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series.head()

# execute the experiment

def run():
# load dataset
series = pd.read_csv('data/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# experiment
repeats = 10
results = pd.DataFrame()
# run experiment
timesteps = 1
#series = ads2
results['results'] = experiment(repeats, series, timesteps)
# summarize results
print(results.describe())
# save results
results.to_csv('experiment_timesteps_1.csv', index=False)
# entry point
run()

## LTSM is not going to work on this small dataset I have tried.
    Transform the time series data so that it is stationary. Specifically, a lag=1 differencing to remove the increasing trend in the data.
    Transform the time series into a supervised learning problem. Specifically, the organization of data into input and output patterns where the observation at the previous time step is used as an input to forecast the observation at the current time timestep
    Transform the observations to have a specific scale. Specifically, to rescale the data to values between -1 and 1 to meet the default hyperbolic tangent activation function of the LSTM model.

These transforms are inverted on forecasts to return them into their original scale before calculating and error score.
A batch size of 1 means that the model will be fit using online training (as opposed to batch training or mini-batch training). As a result, it is expected that the model fit will have some variance.
Ideally, more training epochs would be used (such as 1000 or 1500), but this was truncated to 500 to keep run times reasonable.
Each experimental scenario will be run 10 times.
The reason for this is that the random initial conditions for an LSTM network can result in very different results each time a given configuration is trained.
Let’s dive into the experiments.
The univariate time series is converted to a supervised learning problem before training the model. The specified number of time steps defines the number of input variables (X) used to predict the next time step (y). As such, for each time step used in the representation, that many rows must be removed from the beginning of the dataset. This is because there are no prior observations to use as time steps for the first values in the dataset.
The expectation of increased performance with the increase of time steps was not observed, at least with the dataset and LSTM configuration used. - But one time step with one neuron outperformed.
This raises the question as to whether the capacity of the network is a limiting factor. We will look at this in the next section.
We can repeat the above experiments and increase the number of neurons in the LSTM with the increase in time steps and see if it results in an increase in performance.
The results tell a similar story to the first set of experiments with a one neuron LSTM. The average test RMSE appears lowest when the number of neurons and the number of time steps is set to one.
**You know this is for multiple features - So I wont be looking at it for now - But for the futreaaa**
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
fname="data/stock_data.csv"
data_csv = pd.read_excel (fname)
#how many data we will use

# (should not be more than dataset length )

data_to_use= 100

# number of training data
# should be less than data_to_use

train_end =70
total_data=len(data_csv)
#most recent data is in the end 
#so need offset
start=total_data - data_to_use
#currently doing prediction only for 3 steps ahead
steps_to_predict = 3
train_mse=[]
test_mse=[]
forecast=[]
for i in range(steps_to_predict):
train_mse.append(0)
test_mse.append(0)
forecast.append(0)
yt = data_csv.iloc [start:total_data ,4] #Close price
yt1 = data_csv.iloc [start:total_data ,1] #Open
yt2 = data_csv.iloc [start:total_data ,2] #High
yt3 = data_csv.iloc [start:total_data ,3] #Low
vt = data_csv.iloc [start:total_data ,6] # volume
for i in range(steps_to_predict):

    if i==0:
        units=20
        batch_size=1
    if i==1:
        units=15
        batch_size=1
    if i==2:
         units=80
        batch_size=1
    
    
    
    yt_ = yt.shift (-i - 1  )   
    
    data = pd.concat ([yt, yt_, vt, yt1, yt2, yt3], axis =1)
    data. columns = ['yt', 'yt_', 'vt', 'yt1', 'yt2', 'yt3']
    
    data = data.dropna()
# target variable - closed price
    y = data ['yt_']
# closed, volume, open, high, low
    cols =['yt',    'vt',  'yt1', 'yt2', 'yt3']
    x = data [cols]
    
    
    
    scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
    x = np. array (x).reshape ((len( x) ,len(cols)))
    x = scaler_x.fit_transform (x)
    
    
    scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
    y = np.array (y).reshape ((len( y), 1))
    y = scaler_y.fit_transform (y)
    
    
    
    
    x_train = x [0: train_end,]
    
    
    x_test = x[ train_end +1:len(x),]    
    y_train = y [0: train_end] 
    
    
    
    y_test = y[ train_end +1:len(y)]  
    
    if (i == 0) :     
        prediction_data=[]
        for j in range (len(y_test) - 0 ) :
               prediction_data.append (0)       
    
    
    
    x_train = x_train.reshape (x_train. shape + (1,)) 
    x_test = x_test.reshape (x_test. shape + (1,))
    
    
    
    
    
    
    
    seed =2018
    np.random.seed (seed)

##############

## i=0

##############
if i == 0 :
fit0 = Sequential ()
fit0.add (LSTM ( units , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
fit0.add(Dropout(0.2))
fit0.add (Dense (output_dim =1, activation = 'linear'))
fit0.compile (loss ="mean_squared_error" , optimizer = "adam")

          fit0.fit (x_train, y_train, batch_size =batch_size, nb_epoch =25, shuffle = False)
          train_mse[i] = fit0.evaluate (x_train, y_train, batch_size =batch_size)
          test_mse[i] = fit0.evaluate (x_test, y_test, batch_size =batch_size)
          pred = fit0.predict (x_test) 
          pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))
             # below is just fo i == 0
          for j in range (len(pred) - 0 ) :
                   prediction_data[j] = pred[j] 
    
    
    
          forecast[i]=pred[-1]

#############

## i=1

#############
if i == 1 : 
fit1 = Sequential ()
fit1.add (LSTM ( units , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
fit1.add(Dropout(0.2))
fit1.add (Dense (output_dim =1, activation = 'linear'))
fit1.compile (loss ="mean_squared_error" , optimizer = "adam") 
fit1.fit (x_train, y_train, batch_size =batch_size, nb_epoch =25, shuffle = False)
train_mse[i] = fit1.evaluate (x_train, y_train, batch_size =batch_size)
test_mse[i] = fit1.evaluate (x_test, y_test, batch_size =batch_size)
pred = fit1.predict (x_test) 
pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))
forecast[i]=pred[-1]
#############

## i=2

#############
if i==2 :
fit2 = Sequential ()
fit2.add (LSTM ( units , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(len(cols), 1) ))
fit2.add(Dropout(0.2))
fit2.add (Dense (output_dim =1, activation = 'linear'))
fit2.compile (loss ="mean_squared_error" , optimizer = "adam") 
fit2.fit (x_train, y_train, batch_size =batch_size, nb_epoch =25, shuffle = False)
train_mse[i] = fit2.evaluate (x_train, y_train, batch_size =batch_size)
test_mse[i] = fit2.evaluate (x_test, y_test, batch_size =batch_size)
pred = fit2.predict (x_test) 
pred = scaler_y.inverse_transform (np. array (pred). reshape ((len( pred), 1)))

          forecast[i]=pred[-1]
    
    
    x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test), len(cols))))

prediction_data = np.asarray(prediction_data)
prediction_data = prediction_data.ravel()
for j in range (len(prediction_data) - 1 ):
prediction_data[len(prediction_data) - j - 1 ] = prediction_data[len(prediction_data) - 1 - j - 1]
prediction_data = np.append(prediction_data, forecast)
x_test_all = yt[len(yt)-len(prediction_data)-1:len(yt)-1]
x_test_all = x_test_all.ravel()
plt.plot(prediction_data, label="predictions")
plt.plot( x_test_all, label="actual")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
fancybox=True, shadow=True, ncol=2)
import matplotlib.ticker as mtick
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)
ax = plt.axes()
ax.yaxis.set_major_formatter(tick)
plt.show()
print ("prediction data")
print ((prediction_data))
print ("x_test_all")
print ((x_test_all))
print ("train_mse")
print (train_mse)
print ("test_mse")
print (test_mse)


The below models picks up pricing information in 2 dimesions to make a prediction every day. i.e, for each observation look at pricing info 10 days in the past to make a prediction tomorrow. I have tested longer periods → Shorter periods performed better. This would be easier to discuss in person.  Other models that do not incorporate past info further than 1 days technical infor, achieved AUCs of 70-85%


Bad Performing Models:

LTSM (AUC  - 56%)


        def create_model():
            # Build the model
            K.set_learning_phase(0)
            model = Sequential()
       
    
            # model.add(LSTM(4, input_dim = input_dim, input_length = input_length))
    
            #model.add(LSTM(50,return_sequences=True, dropout=0.2, input_shape=(10,51)))
            #model.add(LSTM(50, dropout=0.2, input_shape=(10,51)))
    
            #model.add(LSTM(50,return_sequences=True, dropout=0.2, input_shape=(100,6)))
            #model.add(LSTM(50, dropout=0.2, input_shape=(100,51)))
    
            model.add(LSTM(20,return_sequences=True, dropout=0.2, input_shape=(input_length,input_dim)))
            model.add(LSTM(20, input_shape=(input_length,input_dim)))
            #model.add(LSTM(8, input_shape=(input_length,input_dim)))
    
            #model.add(LSTM(100, return_sequences=True))  # returns a sequence of vectors of dimension 32
            #model.add(LSTM(50)) # You can take this out, not lead to much
            # The max output value is > 1 so relu is used as final activation.
            model.add(Dense(output_dim, activation='relu'))
    
            model.compile(loss='mean_squared_error',
                          optimizer='sgd',
                          metrics=['accuracy'])
            return model


CNN 2D - (AUC - 61%)

        def create_model():
            conv = Sequential()
            conv.add(Conv2D(20, (1, 4), input_shape = PRED.shape[1:4], activation = 'relu'))
            conv.add(MaxPooling2D((1, 2)))
            conv.add(Flatten())
            conv.add(Dense(1, activation = 'sigmoid'))
            sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
            conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
            return conv

Flattened CNN1D - (AUC - 59%)


        def create_model():
            conv = Sequential()
            conv.add(Conv2D(20, (1, 4), input_shape = PRED.shape[1:4], activation = 'relu'))
            conv.add(MaxPooling2D((1, 2)))
            conv.add(Flatten())
            conv.add(Dense(1, activation = 'sigmoid'))
            sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
            conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
            return conv

