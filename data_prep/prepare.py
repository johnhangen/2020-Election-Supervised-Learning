#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

class PrepareData:
    """
        A class for preparing data for election analysis.

        Parameters:
        - data (pd.DataFrame): The input data for analysis.
        - y (str): The column name of the target variable. Default is '2020_winner'.

        Methods:
        - encode(): Encodes the target variable using LabelEncoder and returns the mapping of encoded labels.
        - split(test_size, random_state): Splits the data into training and testing sets and returns the split datasets.
        - compute_sample_weights(): Computes sample weights based on class imbalance.
        - scale_features(): Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self, data: pd.DataFrame, y: str = '2020_winner') -> None:
        self.data = data
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def encode(self) -> dict:
        """
        Encodes the target variable using LabelEncoder.

        Returns:
        - label_encoder_name_mapping (dict): A dictionary mapping the encoded labels to their original names.
        """
        self.data[self.y] = self.label_encoder.fit_transform(self.data[self.y])

        label_encoder_name_mapping = dict(zip(self.label_encoder.classes_,
                                         self.label_encoder.transform(self.label_encoder.classes_)))
        
        return label_encoder_name_mapping
    
    def split_test_train(self, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Splits the data into training and testing sets.

        Parameters:
        - test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
        - random_state (int): The seed used by the random number generator. Default is 42.

        Returns:
        - X_train, X_test, y_train, y_test (tuple): The split datasets.
        """
        X = self.data.drop(self.y, axis=1)
        y = self.data[self.y]

        X.astype('float64')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def compute_sample_weights(self) -> np.ndarray:
        """
        Computes sample weights based on class imbalance.

        Returns:
        - weights (np.ndarray): Array of sample weights.
        """
        train_weights = compute_sample_weight(class_weight='balanced', y=self.y_train)
        test_weights = compute_sample_weight(class_weight='balanced', y=self.y_test)
        
        return train_weights, test_weights
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes features by removing the mean and scaling to unit variance.

        Parameters:
        - X_train (pd.DataFrame): Training feature dataset.
        - X_test (pd.DataFrame): Testing feature dataset.

        Returns:
        - X_train_scaled, X_test_scaled (tuple): The scaled training and testing feature datasets.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled
    
    def one_hot_encode(self, column_name: str) -> None:
        """
        One-hot encodes the specified column and updates the dataset.

        Parameters:
        - column_name (str): The name of the column to one-hot encode.
        """
        if column_name in self.data.columns:
            one_hot = pd.get_dummies(self.data[column_name], prefix=column_name)
            self.data = self.data.drop(column_name, axis=1)
            self.data = pd.concat([self.data, one_hot], axis=1)
        else:
            print(f"Column '{column_name}' not found in the dataset.")


def main():
    path = 'data/final.csv'

    df = pd.read_csv(path) 

    prepare_data = PrepareData(df)

    label_encoder_name_mapping = prepare_data.encode()

    print(prepare_data.data['2020_winner'].head())


if __name__ == '__main__':
    main()