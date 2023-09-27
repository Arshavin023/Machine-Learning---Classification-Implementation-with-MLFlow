import os
import pandas as pd
from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories, save_json
from sklearn.metrics import (precision_score, recall_score, 
                             f1_score, accuracy_score, roc_auc_score,
                             roc_curve)
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.utils.common import feature_processor
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.config.configuration import ConfigurationManager

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self, actual, predicted):
        precision = precision_score(actual,predicted)
        recall = recall_score(actual, predicted)
        f1_score = (2*precision*recall)/(precision+recall)

        return precision, recall, f1_score
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        preprocessor = feature_processor()
        test_x = test_data.drop([self.config.target_column],axis=1)
        test_x = preprocessor.fit_transform(test_x)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            (precision, recall, f1_score) = self.eval_metrics(test_y,predicted_qualities)

            # Sending metrics as local

            scores = {"precision": precision, "recall":recall, "f1_score":f1_score}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1_score)

            # Model register does not work with file store

            if tracking_uri_type_store != 'file':

                mlflow.sklearn.log_model(model, 'model', registered_model_name = 'XGBClassifier')
            
            else:
                mlflow.sklearn.log_model(model,'model')
