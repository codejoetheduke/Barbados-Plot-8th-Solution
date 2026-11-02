from dataset_creation_imports import *

class DatasetConfig:
    """Centralized configuration for the training pipeline."""

    def __init__(
        self,
        data_path="./data/survey_plans/",
        extract_path="./survey_plans/",
        drive_hf_data_path="./model_dataset/",
        drive_hf_data_path_test="./model_testdataset/",
    ):
        self.DATA_PATH = data_path
        self.EXTRACT_PATH = extract_path
        self.DRIVE_HF_DATA_PATH = drive_hf_data_path
        self.DRIVE_HF_DATA_PATH_Test = drive_hf_data_path_test

    def __repr__(self):
        return (
            f"DatasetConfig(\n"
            f"  DATA_PATH='{self.DATA_PATH}',\n"
            f"  EXTRACT_PATH='{self.EXTRACT_PATH}',\n"
            f"  DRIVE_HF_DATA_PATH='{self.DRIVE_HF_DATA_PATH}',\n"
            f"  DRIVE_HF_DATA_PATH_Test='{self.DRIVE_HF_DATA_PATH_Test}'\n"
            f")"
        )

