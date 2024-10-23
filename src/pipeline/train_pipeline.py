
from src.analysis.basic_data_inspection import DataInspector, DataTypesInspectionStrategy, SummaryStatisticsInspectionStrategy
from src.steps.data_splitter_step import data_splitter_step
from src.steps.handle_missing_values_step import handle_missing_values_step
from src.steps.model_building_step import model_building_step
from src.steps.model_evaluator_step import model_evaluator_step
from src.steps.outlier_detection_step import outlier_detection_step
from src.steps.data_ingestion_step import data_ingestion_step
from src.analysis.univariate_analysis import UnivariateAnalyzer,NumericalUnivariateAnalysis,CategoricalUnivariateAnalysis
from src.steps.data_preprocessing_step import data_preprocessing_step
def training_piplelin():
    file_path = 'D:/coding/ml/mlframework/data/archive.zip'

    # step 1 ->  Data ingestion
    raw_data = data_ingestion_step(file_path=file_path)
    # print(raw_data)

    #step 2 -> missing data 
    filled_data = handle_missing_values_step(raw_data)
    print("handling missing values complete")
    print(f"filled data {filled_data.head()}")

    preprocessed_data = data_preprocessing_step(filled_data,strategy='log',featrues=['Gr Liv Area','SalePrice'])
    print("data preprocessing complete")
    print(f"preprocessed data {preprocessed_data.head()}")


    # detecting outlierrs
    clean_data = outlier_detection_step(preprocessed_data,column_name='SalePrice')
    print("cleaning data complete")
    print(f"clean data {clean_data.head()}")

    # data splitting
    X_train,X_test,y_train,y_test = data_splitter_step(df=clean_data,target_column='SalePrice')
    print("data splitting complete")
   
    # model building
    model = model_building_step(X_train= X_train,y_train=y_train)
    print("model building and trianing complete")
    

    evaluation_metrics,mse = model_evaluator_step(trained_model=model,X_test=X_test,y_test=y_test)
    print(evaluation_metrics,mse)
    return model
    


if __name__=='__main__':
    training_piplelin()



