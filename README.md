# CMPUT466 Machine Learning Mini Proejct

## Environment
The required libraries are sklearn, pandas and numpy. To install the libraries

```
sudo pip install sklearn, pandas, numpy
```

The code was run and tested under python 2.7

To run the code

```
python main.py
```

## Test Data
#### Hold Out Test
These is a baseline algorithm, random prediction in place to serve as sanity checks.

<b>Logistic Regression with L2 regularizer Parameters Tunning</b>

|Regularizer Parameters|Average Accuracy(%)|Running time(secs)|
|---------|-------------|------------|
|0.01|88.28|25.18|
|0.1|88.28|25.377|
|1|88.28418|24.65|

Best parameters for Logistic Regression with l2 regularizer: regulzarizer value = 0.01

We can see Logistic Regression that three different regularization value give us a very close accuracy

<b>Neural Network Parameters Tunning</b>

|Neural Network Parameters|Average Acuracy(%)|Running time(secs)|
|---------|-------------|------------|
|(50,1)|88.01|0.84|
|(100,1)|88.06|0.83|
|(300,1)|88.16|1.79|
|(500,1)|88.08|5.32|
|(800,1)|88.09|7.54|

We could see the best parameters for neural network is with 1 hidden layer and 300 neurons

<b>Comparison</b>

|Algorithm|Average Accuracy(%)|Running time(secs)|
|---------|-------------|------------|
|Random|50.74|0.00409|
|Linear SVM|89.02|5.26|
|Logistic Regression with no regularizer|89.33|0.62|
|Logistic Regression with l2 regularizer with regularizer value = 0.01|88.69|22.64|
|Neural Network with parameters(300,) |88.16|1.79|

#### K-fold cross-validation
K-fold cross-validation with k = 10

<b>Result for Neural Network Parameters Tunning</b>

|Neural Network Parameters|Average Acuracy(%)|Running time(secs)|
|---------|-------------|------------|
|(50,1)| 88.26| 13.63|
|(100,1)| 88.18| 16.64|
|(300,1)| 88.02| 30.66|
|(500,1)| 88.06| 49.74|
|(800,1)| 88.15| 74.83|

We can see the best parameters for Neural Network is 1 layer with 50 neurons

<b>Final Results</b>

|Algorithm|Average Accuracy(%)|Running time(secs)|
|---------|-------------|------------|
|Linear SVM|86.04 +- 5.96|103.48|
|Logistic Regression with no regularizer|88.28 +- 1.96|10.37|
|Logistic Regression with l2 regularizer with regularizer value = 0.01|88.52 +- 0.91|340.25|
|Neural Network with parameters(50,)|88.26 +- 0.15|13.26|

## Reference
- https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- https://en.wikipedia.org/wiki/Default_(finance)
- http://scikit-learn.org/stable/modules/classes.html
- https://pandas.pydata.org/pandas-docs/stable/



