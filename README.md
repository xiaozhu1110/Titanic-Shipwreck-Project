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
