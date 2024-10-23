from src.logger import logging
from src.exception import CustomException
from abc import ABC,abstractmethod
from typing import Any
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self,X_train:pd.DataFrame,y_train:pd.Series) -> RegressorMixin:
        """
            Abstract method to build and train a model

            Parameters:
                X_train (pd.DataFrame): Training features
                y_train (pd.series): training data labels

            Returns:
                RegressorMixin: Trained model
        """
        pass



class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self,X_train:pd.DataFrame,y_train:pd.Series) -> Pipeline:
        """
            Build and train a linear regression model using scikit - learn

            Parameters:
                X_train (pd.DataFrame): Training features
                y_train (pd.Series): training data labels

            Returns:
                Pipeline: Trained model
        """

        if not isinstance(X_train,pd.DataFrame):
            raise CustomException("X_train must be a pandas dataframe")
        if not isinstance(y_train,pd.Series):
            raise CustomException("y_train must be a pandas series")
        
        logging.info("Initializing Linear regression model with training data")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        logging.info("Training linear regression model")
        pipeline.fit(X_train,y_train)

        logging.info("Model training completed")
        return pipeline
    



class LogisitcRegression(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """ 
        
            Build and train a logistic regression model using scikit - learn

            Parameters:
                X_train (pd.DataFrame): Training features
                y_train (pd.Series): training data labels

            Returns:
                Pipeline: Trained model
        """
        if not isinstance(X_train,pd.DataFrame):
            raise CustomException("X_train must be a pandas dataframe")
        if not isinstance(y_train,pd.Series):
            raise CustomException("y_train must be a pandas series")
        
        logging.info("Initializing Linear regression model with training data")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisitcRegression())
        ])
        logging.info("Training logistic regression model")
        pipeline.fit(X_train,y_train)

        logging.info("Model training completed")
        return pipeline
    

class ModelBuilder:
    def __init__(self,strategy:ModelBuildingStrategy):
        """
            Initialzing the model builder with a specific model building strategy

            Parameters:
                strategy (modelBuildingStrategy): the strategy to be used for model building
        """            
        self._strategy = strategy


    def set_strategy(self,strategy:ModelBuildingStrategy):
        """
            Set the model building strategy

            Parameters:
                strategy (ModelBuildingStrategy): the strategy to be used for model building
        """
        self._strategy = strategy

    def build_model(self,X_train:pd.DataFrame,y_train:pd.Series)->RegressorMixin:
        """
            executing the model building and training using current strategy

            Parameters:
                X_trian (pd.dataframe) -> training data features
                y_train (pd.series) -> training data label/target

            Return:
                RegressionMixin: Atrained scikit-learn model instance
        """
        logging.info(f"building and training the model using {self._strategy}")
        return self._strategy.build_model(X_train,y_train)
    

if __name__=="__main__":
    #example
    # df  = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    # model_builder = ModelBuilder(LinearRegressionStrategy())
    pass