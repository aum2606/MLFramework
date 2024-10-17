
from src.analysis.basic_data_inspection import DataInspector, DataTypesInspectionStrategy, SummaryStatisticsInspectionStrategy
from src.steps.data_ingestion_step import data_ingestion_step


def training_piplelin():
    file_path = 'D:/coding/ml/mlframework/data/archive.zip'

    # step 1 ->  Data ingestion
    raw_data = data_ingestion_step(file_path=file_path)
    # print(raw_data)

    #step 2 -> performing data analysis
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.executing_inspection(raw_data)
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.executing_inspection(raw_data)


if __name__=='__main__':
    training_piplelin()