import pandas as pd
import os
from mlProject import logger
from xgboost import XGBClassifier
import joblib
from sklearn.pipeline import Pipeline
from mlProject.utils.common import feature_processor
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.config.configuration import ConfigurationManager

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column],axis=1)
        test_x = test_data.drop([self.config.target_column],axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        preprocessor = feature_processor()
        # train_x_processed = preprocessor.fit_transform(train_x)
        
        xgb = XGBClassifier(learning_rate = self.config.learning_rate,
                            n_estimators = self.config.n_estimators,
                            max_depth = self.config.max_depth,
                            subsample = self.config.subsample,
                            colsample_bytree = self.config.colsample_bytree,
                            gamma = self.config.gamma,
                            reg_alpha = self.config.reg_alpha,
                            reg_lambda = self.config.reg_lambda,
                            min_child_weight = self.config.min_child_weight,
                            eval_metric = self.config.eval_metric,
                            # eval_stopping_rounds = self.config.early_stopping_rounds,
                            tree_method = self.config.tree_method,
                            scale_pos_weight = self.config.scale_pos_weight,
                            objective = self.config.objective,
                            random_state = 42)
        
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('model', xgb)
                                         ])

        model_pipeline.fit(train_x,train_y)

        joblib.dump(model_pipeline,os.path.join(self.config.root_dir, self.config.model_name))
