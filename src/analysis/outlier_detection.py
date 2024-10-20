from src.logger import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierDetectionStrategy(ABC):
    def detect_outliers(self,df :pd.DataFrame)->pd.DataFrame:
        """
            Abstract method to detect outliers in the given dataframe

            Parameters:
                df -> pd.dataframe => dataframe contianing features for outlier detection

            Returns:
                pd.dataframe: A booelan dataframe indicating where outliers are located
        """
        pass


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self,threshold=3):
        """
            initialize threshold for the Zscore outlier detection

            Parameters:
                threshold(int) -> initialize threshold
            
        """
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            detecting outliers using z-score method

            Parameters:
                df -> pd.dataframe => dataframe contianing features for outlier detection

            Returns:
                outliers -> returns boolean for outliers from the dataset
        """
        logging.info("Outliers detection using Z-score method")
        z_score = np.abs((df-df.mean())/df.std())
        outliers = z_score > self.threshold
        logging.info(f"outliers detected with z-schore threshold: {self.threshold}")
        return outliers
    

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            detecting outliers using inter quartile range(IQR) method

            Parameters:
                df -> pd.dataframe => dataframe contianing features for outlier detection

            Returns:
                outliers -> returns boolean for outliers from the dataset
        """
        logging.info("outlier detection using IQR method")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info(f"outliers detected with IQR method")
        return outliers
    


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        """
            initialize strategy for outlier detection

            Parameters:
                strategy -> OutlierDetectionStrategy => strategy for outlier detection
        """
        self._strategy = strategy

    def set_strategy(self,strategy:OutlierDetectionStrategy):
        """
            setting strategy to be used in outlier detection

            Parameters:
                strategy -> OutlierDetectionStrategy => setting strategy for outlier detection
        """
        logging.info("Switching outlier detection strategy")
        self._strategy = strategy

    def detect_outliers(self,df: pd.DataFrame) -> pd.DataFrame:
        """
            this method perform outlier detection function

            Parameters:
                df -> pd.DataFrame => dataframe containing features for outlier detection
            
            Returns:
                outliers -> returns boolean for outliers from the dataset
        """
        logging.info("execturing outlier detection strategy")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self,df:pd.DataFrame,method="remove",**kwargs) ->pd.DataFrame:
        """
            this methods deals with handling the outliers in the dataframe using various strategy

            Parameters:
                df -> pd.DataFrame => dataframe containing features for outlier detection
                method -> str => method to be used for handling outliers
                **kwargs => additional parameters for handling outliers

            Returns:
                cleaned df -> pd.dataframe => dataframe after removing outliers
        """

        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method=="cap":
            logging.info("capping outliers from the dataset")
            df_cleaned = df.clip(lower=df.quantile(0.01),upper=df.quantile(0.99),axis=1)
        else:
            logging.info("Invalid outlier handling method")
            return df
        logging.info("outlier handling completed")
        return df_cleaned
    
    def visualize_outliers(self,df:pd.DataFrame,features:list):
        """
            this method gives visualization of the outliers for selected features

            Parameters:
                df -> pd.DataFrame => dataframe containing features for outlier detection
                features -> list => list of features to be visualized

            Returns:
                None -> plots for visualizing outliers for various features
        """
        logging.info(f"visualizing the outliers for the features {features}")
        for feature in features:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot for {feature}")
            plt.show()
        logging.info("outlier visulizer completed")



if __name__ == '__main__':
    #example
    # df = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    # df_numeric = df.select_dtypes(include=[np.number]).dropna()
    # outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    # df_cleaned = outlier_detector.handle_outliers(df_numeric,method="remove")
    # print(df_cleaned.shape)
    # outlier_detector.visualize_outliers(df_numeric,["SalePrice","Gr Liv Area"])
    pass