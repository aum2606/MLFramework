from src.logger import logging
from abc import ABC,abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self,df:pd.DataFrame,target_column:str):
        """
            this is abstract method tha splits the data into training and testing sets

            Parameters:
                df (pd.DataFrame): the data to be split
                target_column (str): the column that contains the target variable

            Returns:
            X_train, X_test, y_train, y_test -> training and testing split for different feature/columns
        """
        pass



class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self,test_size=0.2,random_state=43):
        """
            initialize the SimpleTrainTestSplitStrategy with specific parameters

            Parameters:
                test_size (float): the proportion of the data to be used for testing
                random_state (int): the seed for the random number generator
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
            splits the data into training and testing set

            Parameters:
                df (pd.DataFrame): the data to be split
                target_column (str): the column that contains the target variable

            Returns:
                X_train, X_test, y_train, y_test -> training and testing split for different feature
        """

        logging.info("Performing simple train test split")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
        logging.info("Simple train test split complete")
        return X_train,X_test,y_train,y_test
    


class DataSplitter:
    def __init__(self,strategy:DataSplittingStrategy):
        """
            initialize strategy for data splitting

            Parameters:
                strategy (DataSplittingStrategy): the strategy to be used for data splitting
        """
        self._strategy = strategy

    def set_strategy(self,strategy:DataSplittingStrategy):
        """
            this method sets the strategy for data splitting

            Parameters:
                strategy (DataSplittingStrategy): the strategy to be used for data splitting
        """
        self._strategy = strategy

    def split(self,df:pd.DataFrame,target_column:str):
        """
            It executes the data splitting with current strategy

            Parameters:
                df (pd.DataFrame): the data to be split
                target_column (str): the column that contains the target variable

            Returns:
                X_train, X_test, y_train, y_test: The training and testing splits for features and target
        """
        logging.info(f"Splitting the data using selected strategy {self._strategy}")
        return self._strategy.split_data(df,target_column)
    

if __name__=="__main__":
    #example
    # df  = pd.read_csv('D:/coding/ml/machine learning framework/src/components/extracted_data/AmesHousing.csv')
    # splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy(test_size=0.2,random_state=42))
    # X_train,X_test,y_train,y_test = splitter.split(df,target_column='SalePrice')
    # print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    pass