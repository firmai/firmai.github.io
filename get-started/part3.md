---
title: "Prediction, NLP"
keywords: text, nlp, regression, word, sentiment, deep,  keras, concepts, supervised, learning
description: Learn how to define load-balanced and scalable service that runs containers.
---
{% include_relative nav.html selected="3" %}
Introductions and Guides to NLP
* Ultimate Guide to [Understand & Implement Natural Language Processing](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
* [Introduction to NLP at Hackernoon](https://hackernoon.com/learning-ai-if-you-suck-at-math-p7-the-magic-of-natural-language-processing-f3819a689386) is for people who suck at math - in their own words
* [NLP Tutorial](http://www.vikparuchuri.com/blog/natural-language-processing-tutorial/)
* [Deep Learning for NLP with Pytorch](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)



Packages

* [Scikit-learn: Machine learning in Python](http://arxiv.org/pdf/1201.0490.pdf)
* [Natural Language Toolkit (NLTK)](http://www.nltk.org/)
* [Pattern](http://www.clips.ua.ac.be/pattern) - A web mining module for the Python programming language. It has tools for natural language processing, machine learning, among others.\
* [spaCy](https://github.com/spacy-io/spaCy) - Industrial strength NLP with Python and Cython.Text Summarization



Text Summarisation 

* [TextRank- bringing order into text](http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) by Mihalcea and Tarau is regarded as the first paper on text summarization. The code is available [here](https://github.com/ceteri/pytextrank)

* [Modelling compressions with Discourse constraints](http://jamesclarke.net/media/papers/clarke-lapata-emnlp07.pdf) by Clarke and Zapata provides a discourse informed model for summarization and subtitle generation.

* [Deep Recurrent Generative decoder model for abstractive text summarization](https://arxiv.org/pdf/1708.00625v1.pdf) by Li et al, 2017 uses a sequence-to-sequence oriented encoder-decoder model equipped with a deep recurrent generative decoder.

* [A Semantic Relevance Based Neural Network for Text Summarization and Text Simplification](https://arxiv.org/pdf/1710.02318v1.pdf) by Ma and Sun, 2017 uses a gated attention enocder-decoder for text summarization.

  

Text Classification

* [Convolutional Neural Networks for sentence classfication](https://arxiv.org/pdf/1408.5882v2.pdf) by Kim Yoon is now regarded as the standard baseline for text classification architecture. 
* [Using a CNN for text classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) by Denny Britz uses the same dataset as Kim Yoon's paper(mentioned above). The code implementation can be found [here](https://github.com/dennybritz/cnn-text-classification-tf).
* [Facebook's fasttext](https://github.com/facebookresearch/fastText) is a library for text embeddings and text classification
* [Brightmart's repo](https://github.com/brightmart/text_classification) has a list of all text classification models with their respective scores, trainings,explanations and their Python implementations.
* [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626v3.pdf) by Zhang et al uses CNN and compares them with the traditional text classification models. Its Lua implementation can be found [here](https://github.com/zhangxiangxiao/Crepe).



