from src.bird_classification.entity.config_entity import DataIngestionConfig
from src.bird_classification.utils.common import create_directories
from src.bird_classification.utils import *
from src.bird_classification.components.data_ingestion import DataIngestion
from src.bird_classification import logger

STAGE_NAME = "Data Ingestion stage"

class ConfigurationManager: #Config Entity data
    def __init__(self):

        self.config = CONFIG_FILE
        self.params = PARAMS_FILE
    
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_URL=config["source_URL"],
            local_data_file=config["local_data_file"],
            unzip_dir=config["unzip_dir"] 
        )

        return data_ingestion_config



class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config) #data ingetion related function
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e