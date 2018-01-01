# Predicting U.S. Business Cycle Recessions with rNNCapstone Project
> Capstone Project
>
> For the whole project, please see the [pdf file](term.pdf).
>
> Ben Chen, November 15, 2016

## I. Definition
### Project Overview
This capstone project attempt to use several type of neural network model to predict
when recession will occur in business cycle in U.S.. Using a monthly time series data
set covering from1963:M1 to 2016:M9,we investigate the impacts of some economic
variables and financial variables on business cycle. We are going to find a neural
network model which can predict recessions more precisely compared to regression,
classification or time series methods, and to find out which type of neural network
perform the best.

### Problem Statement
In this paper, we use a recurrent neural network to classify economic indicators
as arising from either expansion or recession regimes in real time. Recurrent neural
network (RNN) is a class of artificial neural network where connections between units
form a directed cycle. This creates an internal state of the network which allows it
to exhibit dynamic temporal behavior. Unlike feedforward neural networks, RNNs
can use their internal memory to process arbitrary sequences of inputs. This makes
them applicable to tasks such as unsegmented connected handwriting recognition or
speech recognition. rNN takes both historical data and its classification as an input,
which is then used to train the algorithm.

### Metrics
Since there’s quite difference between using deep learning and traditional econo-
metric models, we will consider neither ![AIC](assets/AIC.png) nor ![BIC](assets/BIC.png) to evaluate our models with
lag terms. In order to make our evaluation rigorous, we will choose ![R^2](assets/R^2.png) to evaluate
models on training set and ![F1](assets/F1.png) score to evaluate models on testing set. Though ![R^2](assets/R^2.png)
and ![F1](assets/F1.png) score are thought to be simple bench marks compared with ![AIC](assets/AIC.png) and ![BIC](assets/BIC.png) in
the scenario of predicting time series, ![R^2](assets/R^2.png) and ![F1](assets/F1.png) score are easier to interpret while
![AIC](assets/AIC.png) and ![BIC](assets/BIC.png) are not available in `scikit-learn` or `keras`. As a result, we will
use ![F1](assets/F1.png) score as first metric then the ![R^2](assets/R^2.png) in order to measure not only the predictive
power but how the classifiers depict variance of data. And also that’s why we use
mean square error in our loss function instead of other choices.

## II. Analysis
Data Exploration

Exploratory Visualization

Algorithms and Techniques

Benchmark

## III. Methodology
Data Preprocessing

Implementation

Refinement

## IV. Results
Model Evaluation and Validation

Justification

## V. Conclusion
Free-Form Visualization

Reflection

Improvement


