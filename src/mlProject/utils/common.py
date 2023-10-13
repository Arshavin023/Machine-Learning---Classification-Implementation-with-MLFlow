import os
from box.exceptions import BoxValueError
import yaml
from mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import pandas as pd
from pandas import DataFrame
from numpy import ndarray
import numpy as np

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def feature_processor():
        
    # Define preprocessing for numerical categorical features
    numeric_features = [0,1,2,3,4,5,6,7,8,9]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # Define preprocessing for nominal categorical features
    nominal_features = [10,11,12,13,14,15,16,17,18,19]
    nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Define preprocessing for ordinal categorical features
    ordinal_features = [16,17,18,19]
    ordinal_transformer = Pipeline(steps=[('labelencoder', OrdinalEncoder())])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

    return preprocessor

column_names = ['credit_duration(month)','credit_amount','duration_of_current_employment',
 'installment_percent','guarantors','duration_in_current_address','age','concurrent_credits',
 'no_of_credits_at_the_bank','no_of_dependents','account_type','payment_status_of_previous_loan',
 'loan_purpose','savings_type','marital_status','most_valuable_asset','type_of_apartment',
 'occupation','telephone','foreign_worker']

def numpy_to_pandas(data):
    df = pd.DataFrame(data,columns=column_names)
    return df

def ordinal_category_encode(data):
            # feature engineering for ordinal categorical data        

        # Custom mapping
        occupation_mapping  = {
            'service and sales': 3,
            'skilled trades and technical': 2,
            'manufacturing and production': 4,
            'professional and managerial': 1}

        type_of_apartment_mapping = {
            'studio apartment': 2,
            'one-bedroom apartment': 1,
            'two or multi-bedroom apartment': 3}

        telephone_mapping = {'yes':1, 'no':0}

        foreign_mapping = {'yes':0, 'no':1}
        # Sample data

        # Transforming ordinal categories with custom mapping
        apartment_preprocessed_data = [[type_of_apartment_mapping[category[0]]] for category in data[['type_of_apartment']].values]

        occupation_preprocessed_data = [[occupation_mapping[category[0]]] for category in data[['occupation']].values]

        telephone_preprocessed_data = [[telephone_mapping[category[0]]] for category in data[['telephone']].values]

        foreign_preprocessed_data = [[foreign_mapping[category[0]]] for category in data[['foreign_worker']].values]

        combined_ordinal_categories = np.concatenate((apartment_preprocessed_data,occupation_preprocessed_data,
                                                    telephone_preprocessed_data,foreign_preprocessed_data),axis=1)

        # Use OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        encoded_data = ordinal_encoder.fit_transform(combined_ordinal_categories)

        data[['type_of_apartment','occupation','telephone','foreign_worker']] = combined_ordinal_categories

        return data
