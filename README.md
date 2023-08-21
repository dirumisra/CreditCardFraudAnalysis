

##### Credit Card Fraud Detection

In this project we will predict fraudulent credit card transactions with the help of Machine learning models.
In order to complete the project, we are going to follow below high level steps to build and select best model.

- Read the dataset and perform exploratory data analysis
- Building different classification models on the unbalanced data
- Building different models on 3 different balancing technique.

        - Random Oversampling
        - SMOTE
        - ADASYN


# Importing model building packages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score, classification_report

import warnings
warnings.filterwarnings("ignore")
