from src.logger import logging
from src.exception import CustomException
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self,model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series)->dict:
        """
            abstract method to evaluate model

            Parameters:
                model (RegressorMixin): model to be evaluated
                X_test (pd.DataFrame): test data
                y_test (pd.Series): target variable

            Returns:
                dict: containing evaluation metrics
        """
        pass


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
            evaluates a regression model using mean-square-erro,mean-absolute-error and r2_score

            Paratmeters:
                model (RegressorMixin): model to be evaluated
                X_test (pd.DataFrame): test data
                y_test (pd.Series): target variable

            Returns:
                dict: containing evaluation metrics
        """

        logging.info("Evaluating model for test data")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics")
        mae = mean_absolute_error(y_pred=y_pred,y_true=y_test)
        mse = mean_squared_error(y_pred=y_pred,y_true=y_test)
        r2 = r2_score(y_pred=y_pred,y_true=y_test)
        metrics = {"mean squared error": mse, "mean absolute error": mae, "R_squared":r2}
        logging.info(f"model evaluation metrics : {metrics}")
        return metrics


class ModelEvaluator:
    def __init__(self,strategy:ModelEvaluationStrategy):
        """
            initializes ModelEvaluator with a specific model evaluation strategy

            Parameters:
                strategy (ModelEvaluationStrategy): strategy to be used for model evaluation
        """
        self._strategy = strategy

    def set_strategy(self,strategy:ModelEvaluationStrategy):
        """
            set a specific strategy for ModelEvaluator to perfom evaluation

            Parameters:
                strategy (ModelEvaluationStrategy): strategy to be used for model evaluation
        """
        self._strategy = strategy

    def evaluate(self,model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series)->dict:
        """
            this methods executes ModelEvaluator with selected strategy

            Parameters:
                model (RegressorMixin): model to be evaluated
                X_test (pd.DataFrame): test data
                y_test (pd.Series ): target variable

            Returns:
                dict: A dictionary containing evaluation metrics.
        """
        logging.info(f"evaluating model using the selected strategy {self._strategy}")
        return self._strategy.evaluate_model(model,X_test,y_test)
    
if __name__=="__main__":
    pass
