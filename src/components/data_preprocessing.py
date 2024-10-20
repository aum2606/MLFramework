from src.logger import logging 
from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler


class PreprocessingStrategy(ABC):
    @abstractmethod
    def apply_data_transformation(self,df:pd.DataFrame)->pd.DataFrame:
        """
            Abstract method to apply data transformation(feature engineering) to the dataframe

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation

            Return:
                pd.dataframe -> transformed data
        """
        pass


class LogTransformation(PreprocessingStrategy):
    def __init__(self,features):
        """
            Initialize the LogTrasformation with specific feature to transform data.

            Parameters:
                features -> list of features to be transformed
        """
        self.features = features

    def apply_data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Applies log transformation to the specified features in the dataframe

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation

            Returns:
                pd.dataframe -> log transfered features dataframe
        """
        logging.info(f"applying log transformation to the following features {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("log transformation complete")
        return df_transformed
    


class StandardScaling(PreprocessingStrategy):
    def __init__(self,features):
        """
            Initialize the StandardScaling with specific feature to transform data

            Parameters:
                features -> list of features to be transformed
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Applies standard scaler transformation to the specified features in the dataframe

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation

            Returns:
                pd.dataframe -> scaled features dataframe
        """
        logging.info(f"appyling standard scaler to the dataframe features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("standard scaler transformation complete")
        return df_transformed
    

class MinMaxScaling(PreprocessingStrategy):
    def __init__(self,features,feature_range=(0,1)):
        """
            Initialize the StandardScaling with specific feature to transform data

            Parameters:
                features -> list of features to be transformed
                feature_range -> tuple of (min,max) to scale the data to
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range) 


    def apply_data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Applies standard scaler transformation to the specified features in the dataframe

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation

            Returns:
                pd.dataframe -> scaled features dataframe
        """
        logging.info(f"appyling MinMax scaler to the dataframe features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("MinMax scaling complete")
        return df_transformed
    

class OneHotEncoding(PreprocessingStrategy):
    def __init__(self,features):
        """
            Initialize the OneHotEncoing with specific feature to transform data

            Parameters:
                features -> list of features to be transformed
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False,drop='first')

    def apply_data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Applies One Hot Encoding to the categorical features in the dataframe

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation
            
            Returns:
                pd.dataframe -> one hot encoded features dataframe
        """
        logging.info(f"applying one hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(self.encoder.fit_transform(df[self.features]),columns=self.encoder.get_feature_names_out(self.features))
        df_transformed = pd.concat([df_transformed,encoded_df],axis=1)
        logging.info("One hot encoding complete")
        return df_transformed
    

class PreProcessor:
    def __init__(self,preprocessing_strategy: PreprocessingStrategy):
        """
            initializing the preprocessing strategy 

            Parameters:
                preprocessing_strategy -> PreprocessingStrategy instance
        """
        self.preprocessing_strategy = preprocessing_strategy

    
    def set_preprocessing_strategy(self,preprocessing_strategy:PreprocessingStrategy):
        """
            setting the preprocessing strategy for data preprocessing

            Parameters:
                preprocessing_strategy -> PreprocessingStrategy instance
        """
        logging.info("switching preprocessing strategy")
        self.preprocessing_strategy = preprocessing_strategy

    def apply_data_preprocessing(self,df:pd.DataFrame) ->pd.DataFrame:
        """
            applying data preprocessing using selected strategy

            Parameters:
                df -> pd.dataframe => the dataframe containing data for transformation
            
            Returns:
                pd.dataframe -> preprocessed dataframe
        """
        logging.info("applying preprocessing strategy")
        return self.preprocessing_strategy.apply_data_transformation(df=df)
    


if __name__ =="__main__":
    df = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    log_transformer = PreProcessor(LogTransformation(features=['SalePrice','Gr Liv Area']))
    df_log_transformed = log_transformer.apply_data_preprocessing(df=df)

    standard_scaler_transformer = PreProcessor(StandardScaling(features=['SalePrice','Gr Liv Area']))
    df_standard_scaled = standard_scaler_transformer.apply_data_preprocessing(df=df)

    minmax_scaler_transformer = PreProcessor(MinMaxScaling(features=['SalePrice','Gr Liv Area'],feature_range=(0,1)))
    df_minmax_scaled = minmax_scaler_transformer.apply_data_preprocessing(df=df)

    one_hot_encoder = PreProcessor(OneHotEncoding(features=['Neighborhood']))
    df_one_hot_encoded = one_hot_encoder.apply_data_preprocessing(df=df)