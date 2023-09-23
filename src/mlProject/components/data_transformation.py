import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
    
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        # feature engineering for ordinal categorical data        

        # Custom mapping
        occupation_mapping  = {
            'service and sales': 0,
            'skilled trades and technical': 1,
            'manufacturing and production': 2,
            'professional and managerial': 3}

        type_of_apartment_mapping = {
            'studio apartment': 0,
            'one-bedroom apartment': 1,
            'two or multi-bedroom apartment': 2}

        # Sample data

        # Preprocess the data using the custom mapping
        apartment_preprocessed_data = [[type_of_apartment_mapping[category[0]]] for category in data[['type_of_apartment']].values]

        occupation_preprocessed_data = [[occupation_mapping[category[0]]] for category in data[['occupation']].values]

        combined_ordinal_categories = np.concatenate((apartment_preprocessed_data,occupation_preprocessed_data),axis=1)

        # Use OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        encoded_data = ordinal_encoder.fit_transform(combined_ordinal_categories)

        data[['type_of_apartment','occupation']] = combined_ordinal_categories

        train, test = train_test_split(data, test_size=0.20, random_state=42)
        
        # save train and test files as CSVs
        train.to_csv(os.path.join(self.config.root_dir,'train.csv'),index=False)
        test.to_csv(os.path.join(self.config.root_dir,'test.csv'),index=False)

        #Log information
        logger.info("Splitted data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)



