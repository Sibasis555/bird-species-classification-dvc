from bird_classification import logger
from bird_classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from bird_classification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from bird_classification.pipeline.stage_03_model_training import ModelTrainingPipeline
# from src.bird_classification.pipeline.predict import PredictionPipeline


# STAGE_NAME = "Data Ingestion stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Model Initialization stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = PrepareBaseModelTrainingPipeline()
#    data_ingestion.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
except Exception as e:
        logger.exception(e)
        raise e

# obj=PredictionPipeline("C:/Users/SIBASIS/Downloads/dummy_bird_species/Test/American_Redstart/American_Redstart_0009_103974.jpg")
# result=obj.predict()
# print(result)