from src.analysis.outlier_detection import OutlierDetector,ZScoreOutlierDetection
from src.logger import logging
from src.exception import CustomException
import pandas as pd


def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector."""
    logging.info(f"Starting outlier detection step with DataFrame of shape")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise CustomException("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise CustomException("Input df must be a pandas DataFrame.")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise CustomException(f"Column '{column_name}' does not exist in the DataFrame.")
        # Ensure only numeric columns are passed
    df_numeric = df.select_dtypes(include=[int, float])

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    return df_cleaned
