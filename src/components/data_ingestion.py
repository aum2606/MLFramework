import os
import zipfile
from abc import ABC,abstractmethod
from sqlalchemy import create_engine
import requests
import json
from bs4 import BeautifulSoup
from google.cloud import storage
import pandas as pd





class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,file_path: str)->pd.DataFrame:
        """abstract method for ingesting data"""
        pass

class ZipDataIngestion(DataIngestor):
    def ingest(self,file_path: str)->pd.DataFrame:
        """extracting the .zip file and returning the content as pandas Dataframe"""
        if not file_path.endswith('.zip'):
            raise ValueError('The provided file is not a .zip file')
        
        #extracting the zip file
        with zipfile.ZipFile(file_path,'r') as zip_file:
            zip_file.extractall("extracted_data")

        extracted_file = os.listdir('extracted_data')
        csv_files = [f for f in extracted_file if f.endswith('.csv')]
        print(csv_files)

        if len(csv_files)==0:
            raise FileNotFoundError("No csv file found in the extracted data.")
        if len(csv_files)>1:
            raise ValueError("More than one csv file found in the extracted data. Please specify which file to use. ")
        
        #reading the csv file
        # csv_file_path = os.path.join('extracted_data',csv_files[0])
        # df = pd.read_csv(csv_file_path)

        df = FileExtractor.csv_file_extractor(file_path=file_path)

        #returning the data
        return df



class FileExtractor(DataIngestor):
    """deals with various data file and extracting them accordingly"""
    def ingest(self, file_path: str) -> pd.DataFrame:
        return super().ingest(file_path)

    def csv_file_extractor(file_path):
        return pd.read_csv(file_path)
    
    def json_file_extractor(file_path):
        return pd.read_json(file_path)
    
    def excel_file_extractor(file_path):
        return pd.read_excel(file_path)

    def database_integration(SQL_ENGINE_URL = ''):
        engine = create_engine(SQL_ENGINE_URL)
        return pd.read_sql_table('table_name', engine)

    def api_integration(API_URL=''):
        #making an api request call
        response = requests.get(API_URL)
        
        # returning the file as a pandas dataframe 
        return pd.json_normalize(response.json())


    def web_scrapping(WEB_URL='',YOUR_CLASS=''):
        #making an http request and parse the html file
        response = requests.get(WEB_URL)
        soup = BeautifulSoup(response.text,'html.parser')

        #extracting the relevent data from the site
        data = soup.find('div',class_=YOUR_CLASS).text

        #returning the file a an pandas dataframe
        return pd.DataFrame(data.split('\n'),columns=['data'])
    

    def cloud_platforms(YOUR_BUCKET='',FILE_NAME='',local_file=''):
        #creating a client and accessing the bucket
        client = storage.Client()
        bucket = client.get_bucket(YOUR_BUCKET)

        #downloading the file from the cloud storage as a csv file
        blobs = bucket.list_blobs(FILE_NAME)
        blobs.download_to_file(f"${local_file}.csv")

        #returning file as a data frame
        return pd.read_csv(f"${local_file}.csv")

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str)->DataIngestor:
        """returns the appropriate data ingestor based on the file extension"""
        if file_extension == '.zip':
            return ZipDataIngestion()
        else:
            raise ValueError('Unsupported file extension')

if __name__=='__main__':
    #file path to be entered 
    file_path = "D:/coding/ml/mlframework/data/archive.zip"

    #spliting the extension from the file path 
    file_extension = os.path.splitext(file_path)[1]

    #checking the file extension
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension=file_extension)

    #calling the data ingestor to extract file and store it as pandas dataframe in df variable
    df = data_ingestor.ingest(file_path=file_path)

    # file_path = 'D:/coding/ml/mlframework/data/setcalories.csv'
    # df = FileExtractor.csv_file(file_path)

    #printing data head
    print(df.head())