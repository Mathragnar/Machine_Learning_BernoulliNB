# Machine Learning Models for Classification

This repository contains two Python scripts demonstrating the use of Bernoulli Naive Bayes for classification tasks. 

## Script 1: `bernoulli_naive_bayes_basic.py`

This script implements a basic Bernoulli Naive Bayes model without hyperparameter tuning.

**Key features:**

* Data splitting into training, validation, and testing sets (70%-15%-15% split).
* Training a Bernoulli Naive Bayes model using `sklearn.naive_bayes.BernoulliNB`.
* Evaluation using accuracy score.
* Saving the trained model using `pickle`.

**Usage:**

1. Make sure you have the required libraries installed (`numpy`, `sklearn`, `pickle`).
2. Replace `X` and `y` with your feature matrix and target variable respectively.
3. Run the script: `python bernoulli_naive_bayes_basic.py`

**Output:**

* Accuracy score on the test set.
* Number of samples in each set (training, validation, testing).
* Class distribution in each set.
* A saved model file named `modelo_treinado_BernoulliNB.pkl`.


## Script 2: `bernoulli_naive_bayes_hyperparameter_tuning.py`

This script builds upon the first script by adding hyperparameter tuning using Grid Search and Random Search.

**Key features:**

* Data splitting into training, validation, and testing sets (70%-15%-15% split).
* Hyperparameter tuning using:
    * **Grid Search:** Explores a predefined grid of hyperparameters.
    * **Random Search:** Samples hyperparameters from a defined distribution.
* Training and evaluating Bernoulli Naive Bayes models using different hyperparameter combinations.
* Selection of the best model based on accuracy.
* Evaluation of the best model on the test set.
* Saving the best trained model using `pickle`.

**Usage:**

1. Make sure you have the required libraries installed (`numpy`, `sklearn`, `pickle`, `scipy`).
2. Replace `X` and `y` with your feature matrix and target variable respectively.
3. Run the script: `python bernoulli_naive_bayes_hyperparameter_tuning.py`

**Output:**

* Best hyperparameters found by Grid Search and Random Search.
* Accuracy scores for both search methods.
* Accuracy score of the best model on the test set.
* A saved model file named `modelo_treinado.pkl`.


## Conclusion

These scripts provide a basic example of using Bernoulli Naive Bayes for classification and demonstrate the benefits of hyperparameter tuning for improving model performance. Choose the script that best suits your needs and modify it according to your specific dataset and requirements. Remember to install the necessary libraries before running the scripts.
