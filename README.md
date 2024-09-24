# Titanic-survival-prediction

## Titanic Survival Prediction using Machine Learning

The sinking of the Titanic is one of the most infamous disasters in history. Predicting which passengers survived the tragedy is a well-known challenge in the field of data science and machine learning. In this project, we use various machine learning models to predict passenger survival based on available features such as age, gender, and class. This project emphasizes feature engineering, model optimization, and evaluation to create an accurate classification model.

Refer to [Data Analysis (Titanic Classification).ipynb](https://github.com/xiaozhu1110/Titanic-Shipwreck-Project/blob/main/Data%20Analysis%20(Titanic%20Classification).ipynb) for the code.

## 1. Problem Statement

Predicting whether a passenger survived the Titanic shipwreck based on features like age, gender, class, and ticket price.

## 2. Data Description

The data is obtained from the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data). The dataset contains the following characteristics:

- **Number of instances**: 891
- **Number of attributes**: 12
    - 11 features (input) and 1 target variable (output)

### Attribute Information:

**Inputs:**

- **PassengerId**: Unique identifier for each passenger.
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger in years.
- **SibSp**: Number of siblings/spouses aboard the Titanic.
- **Parch**: Number of parents/children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Passenger fare.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

All the features except *Fare* and *Age* are categorical variables.

**Output:**

- **Survived**: Binary classification (0 = Did not survive, 1 = Survived).

## 3. Modelling and Evaluation

### Algorithms used:
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

### Metrics:
Since the target variable (Survival) is binary, classification metrics have been used for evaluation. These metrics include:
- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: The ratio of true positives to the sum of true and false positives.
- **Recall (Sensitivity)**: The ratio of true positives to the sum of true positives and false negatives.
- **F1-Score**: The harmonic mean of Precision and Recall.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Evaluates the model's performance across different threshold settings.

## 4. Results

### 4.1 Model Performance

To predict survival on the Titanic dataset, several machine learning models were trained and evaluated using cross-validation. Below are the cross-validation accuracy results for each model:

- **Gaussian Naive Bayes**:
  - Mean Accuracy: 0.7248
- **Logistic Regression (with and without scaling)**:
  - Without scaling: 0.8104
  - With scaling: 0.8122
- **Decision Tree**:
  - Mean Accuracy: 0.7821
- **Random Forest**:
  - Without scaling: 0.8433
  - With scaling: 0.8454
- **Support Vector Machine (SVM)**:
  - Mean Accuracy: 0.8364
- **K-Nearest Neighbors (KNN)**:
  - Mean Accuracy: 0.8238
- **Voting Classifier (Combining multiple models)**:
  - Mean Accuracy: 0.8454

### 4.2 Feature Importance

The Random Forest model was used to rank feature importance. The chart below shows the relative importance of different features in predicting survival on the Titanic:

![Feature Importance](https://github.com/xiaozhu1110/Titanic-Shipwreck-Project/blob/main/Feature%20Importance%20Titanic.png)

The most important features included:
- **Sex**: Gender played a significant role in survival chances.
- **Pclass**: The class of ticket (1st, 2nd, or 3rd) was highly indicative of survival.
- **Fare**: The price paid for the ticket had a notable influence on survival rates.
- **Age**: Passenger age also impacted the probability of survival.

### 4.3 Hyperparameter Tuning

Grid Search was employed to fine-tune the hyperparameters of several models. After tuning, the best hyperparameters were identified as follows:
- **Logistic Regression**: C = 0.1, penalty = 'l1'
- **Random Forest**: max_depth = 10, n_estimators = 100
- **Support Vector Classifier (SVC)**: C = 1, kernel = 'rbf'

After tuning, the optimized models achieved the following scores:
- **Logistic Regression**: 0.8140
- **Random Forest**: 0.8464
- **SVC**: 0.8375


