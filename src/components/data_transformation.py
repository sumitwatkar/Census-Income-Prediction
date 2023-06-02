"""Handling Missing Values
Outliers Treatment
Handling Imbalanced Data
Converting categoral features into numerical features
"""

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



@dataclass
class DataTransformationConfig:
    preprocesor_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Started")

            numerical_features = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week', 'native_country']
            
            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            df.loc[(df[col]> upper_limit), col] = upper_limit
            df.loc[(df[col]< lower_limit), col] = lower_limit

            return df
        

        except Exception as e:
            logging.info("Outliers treatment done")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            numerical_features = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
            'hours_per_week', 'native_country']

            for col in numerical_features:
                self.remove_outliers_IQR(col = col, df = train_data)
            
            logging.info("Outliers capped on train data")


            for col in numerical_features:
                self.remove_outliers_IQR(col = col, df = test_data)
            
            logging.info("Outliers capped on test data")

            
            preprocessor_obj = self.get_data_transformation_obj()

            target_column = "income"
            drop_column = [target_column]

            logging.info('Splitting train data into dependent and independent features')
            input_feature_train_data = train_data.drop(target_column, axis=1)
            target_feature_train_data = train_data[target_column]

            logging.info('Splitting test data into dependent and independent features')
            input_feature_test_data = test_data.drop(target_column, axis=1)
            target_feature_test_data = test_data[target_column]

            # Apply transformation on train data and test data
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)


            # Apply preprocessor object on train data and test data
            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]


            save_obj (file_path= self.data_transformation_config.preprocesor_obj_file_path,
                      obj= preprocessor_obj)
            
            return (train_array, 
                    test_array,
                    self.data_transformation_config.preprocesor_obj_file_path)
    

        except Exception as e:
            raise CustomException(e, sys)