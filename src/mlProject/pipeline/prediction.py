import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject.utils.common import feature_processor

# Create class to load model
class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def predict(self, data):
        preprocessor = feature_processor()
        preprocessed_data = preprocessor.fit_transform(data)
        prediction = self.model.predict(preprocessed_data)

        return prediction

