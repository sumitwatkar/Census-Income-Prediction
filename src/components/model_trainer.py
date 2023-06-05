import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_obj
from src.utils import evaluate_model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts/model_trainer", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the data into dependent and independent features")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Machine Learning Algorithms initialization")
            
            models = {
                "Random Forest" : RandomForestClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression()

            }

            logging.info("Machine Learning Parameters Initialization")



            params = {
                "Random Forest":{
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                },

                "Decision Tree":{
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features":["auto","sqrt","log2"]
                },

                "Logistic Regression":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                }
            }

            logging.info("Model Evaluation Started")

            model_report: dict = evaluate_model(self, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)

            logging.info("Model report initiation")

            # to get  best model from our report dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)

            ]

            best_model = models[best_model_name]

            logging.info(f"Best Model Found, Model Name : {best_model_name}, Accuracy Score : {best_model_score}")

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best model found, Model Name is {best_model_name}, accuracy Score: {best_model_score}")

            
            save_obj(file_path= self.model_trainer_config.train_model_file_path,
                     obj = best_model)

        except Exception as e:
            raise CustomException(e, sys)