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

|Algorithm|Average Accuracy(%)|Running time(secs)|
|---------|-------------|------------|
|Random|49.8|0.00485|
|Linear SVM|88.33|4.50|
|Logistic Regression with no regularizer|88.96|0.515|
|Logistic Regression with l2 regularizer|88.59|20.99|
|Neural Network(nh = (300,1))|88.21|1.46|

#### K-fold cross-validation
K-fold cross-validation with k = 10

|Algorithm|Average Accuracy(%)|Running time(secs)|
|---------|-------------|------------|
|Linear SVM|82.33 +- 3.33e-09|65.07|
|Logistic Regression with no regularizer|88.28 +- 3.34e-09|7.85|
|Logistic Regression with l2 regularizer|88.52 +- 3.34e-09|311.9|
|Neural Network(nh = (300,1))|88.30 +- 3.34e-09|33.9|


## Reference
- https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- https://en.wikipedia.org/wiki/Default_(finance)
- http://scikit-learn.org/stable/modules/classes.html
- https://pandas.pydata.org/pandas-docs/stable/



