---
title: "Prediction, Event and Anomaly"
keywords: classification, detection, anomaly, event, time-series, deep,  keras, LTSM
description: Deploy your app to production using Docker CE or EE.
---
{% include_relative nav.html selected="6" %}


## Anomaly Detection Software


| Name          | Language       | Pitch     
| ------------- |:-------------: | :-------------:     
| Etsy's [Skyline](https://github.com/etsy/skyline)                        | Python |Skyline is a real-time anomaly detection system, built to enable passive monitoring of hundreds of thousands of metrics   
| Linkedin's [luminol](https://github.com/linkedin/luminol)                | Python |Luminol is a light weight python library for time series data analysis. The two major functionalities it supports are anomaly detection and correlation. It can be used to investigate possible causes of anomaly.      
| Ele.me's [banshee](https://github.com/eleme/banshee)                     | Mentat's [datastream.io](https://github.com/MentatInnovations/datastream.io)| Python |An open-source framework for real-time anomaly detection using Python, Elasticsearch and Kibana.

## Related Software

This section includes some time-series software for anomaly detection-related tasks, such as forecasting.


| Name          | Language       | Pitch     
| ------------- |:-------------: | :-------------:   
| Facebook's [Prophet](https://github.com/facebook/prophet) | Python/R | Prophet is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays.
| [PyFlux](https://github.com/RJT1990/pyflux) | Python | The library has a good array of modern time series models, as well as a flexible array of inference options (frequentist and Bayesian) that can be applied to these models.
| [Pyramid](https://github.com/tgsmith61591/pyramid) | Python | Porting of R's _auto.arima_ with a scikit-learn-friendly interface.
| [SaxPy](https://github.com/seninp/saxpy) | Python | General implementation of SAX, as well as HOTSAX for anomaly detection.


## Benchmark Datasets

- Numenta's [NAB](https://github.com/numenta/NAB)
-- NAB is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications. It is comprised of over 50 labeled real-world and artificial timeseries data files plus a novel scoring mechanism designed for real-time applications.
- Yahoo's [Webscope S5](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)
-- The dataset consists of real and synthetic time-series with tagged anomaly points. The dataset tests the detection accuracy of various anomaly-types including outliers and change-points. 


## Time Series Anomaly Detection
   * SAX
        * HOT SAX: Finding the Most Unusual Time Series Subsequence: Algorithms and Applications, Eamonn Keogh, Jessica Lin, Ada Fu, 2005 - [Paper](http://www.cs.ucr.edu/~eamonn/discords/HOT%20SAX%20%20long-ver.pdf), [Materials](http://www.cs.ucr.edu/~eamonn/discords/)
    * LSTM
        * LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection, Pankaj Malhotra, Anusha Ramakrishnan, Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, Gautam Shroff, 2016 - [Paper](https://drive.google.com/file/d/0B8Dg3PBX90KNQWRwMElkVkQ0aFgzZGpzOGQtUU5DeWZYUlVV/view)
        * Credit Card Transactions, Fraud Detection, and Machine Learning: Modelling Time with LSTM Recurrent Neural Networks, Bénard Wiese and Christian Omlin, 2009 - [Springer](http://link.springer.com/chapter/10.1007%2F978-3-642-04003-0_10)
        * Long Short Term Memory Recurrent Neural Network Classifier for Intrusion Detection, Jihyun Kim, Jaehyun Kim, Huong Le Thi Thu, and Howon Kim - [Paper](https://www.google.ru/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=0ahUKEwiEoKCEpvjQAhUEQyYKHZG-AJYQFggoMAI&url=http%3A%2F%2Fisi-dl.com%2Fdownloadfile%2F32576&usg=AFQjCNHBXmuxdaQvw4f7ULSMEfUkuHX85g)
        * Deep Recurrent Neural Network-based Autoencoders for Acoustic Novelty Detection, Erik Marchi Fabio Vesperini, Stefano Squartini, and Bjo ̈rn Schuller - [Paper](http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/schuller/Publications/Marchi16-DRN.pdf)
        * A Novel Approach for Automatic Acoustic Novelty Detection Using a Denoising Autoencoder with Bidirectional LSTM Neural Networks, Erik Marchi, Fabio Vesperini, Florian Eyben, Stefano Squartini, Bjo ̈rn Schuller - [Paper](http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/schuller/Publications/Marchi15-ANA-UP_TUM_UK.pdf)
    * Transfer learning
        * Transfer Representation-Learning for Anomaly Detection, Jerone T. A. Andrews, Thomas Tanay, Edward J. Morton, Lewis D. Griffin, 2016 - [Paper](https://drive.google.com/file/d/0B8Dg3PBX90KNeFROU3BDT1ZhTXlSV3Rsb3JfVWNTWkpLTUhJ/view)
    * Anomaly Detection Based on Sensor Data in Petroleum Industry Applications, Luis Martí,1, Nayat Sanchez-Pi, José Manuel Molina, and Ana Cristina Bicharra Garcia - [Paper](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4367333/)
    * Anomaly detection in aircraft data using recurrent nueral networks (RNN), Anvardh Nanduri, Lance Sherry - [Paper](http://catsr.ite.gmu.edu/pubs/ICNS_2016_AnomalyDetectionRNN_01042015.pdf)
    * Bayesian Online Changepoint Detection, Ryan Prescott Adams, David J.C. MacKay - [Paper](http://hips.seas.harvard.edu/files/adams-changepoint-tr-2007.pdf)
    * Anomaly Detection in Aviation Data using Extreme Learning Machines, Vijay Manikandan Janakiraman, David Nielsen - [Paper](https://c3.nasa.gov/dashlink/static/media/publication/PID4205935.pdf)



