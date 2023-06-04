from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(self,X_train,y_train,X_test,y_test,models,params):
        try:
            report = {}

            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = models[model_name]
                para = params[model_name]

                grid = GridSearchCV(model, para, cv=2, n_jobs=-1, verbose=2)
                grid.fit(X_train, y_train)
                
                model.set_params(**grid.best_params_)
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)
                
                test_model_score = accuracy_score(y_test, y_test_pred)
                
                report[model_name] = test_model_score
            return report

        except Exception as e:
         raise CustomException(sys, e)