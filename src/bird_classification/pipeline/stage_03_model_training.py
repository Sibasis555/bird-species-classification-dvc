from bird_classification import logger
from bird_classification.components.model_training import Data_preprocessing, Training
from bird_classification.utils import *
from bird_classification.utils.common import create_directories
from bird_classification.entity.config_entity import TrainingConfig
from pathlib import Path

STAGE_NAME = "Training"

class ConfigurationManager: #Config Entity data
    def __init__(self):
        get_config_and_param()
        self.config = CONFIG_FILE
        self.params = PARAMS_FILE
    
        create_directories([self.config["artifacts_root"]])

    def get_training_config(self) -> TrainingConfig:
        training = self.config["training"]
        prepare_base_model = self.config["prepare_base_model"]
        params = self.params
        # print(self.config["training"])
        training_data = self.config["training"]["train_data"]
        testing_data = self.config["training"]["test_data"]
        create_directories([
            Path(training["root_dir"])
        ])
        
        training_config = TrainingConfig(
            root_dir=Path(training["root_dir"]),
            trained_model_path=Path(training["trained_model_path"]),
            updated_base_model_path=Path(prepare_base_model["updated_base_model_path"]),
            training_data=Path(training_data),
            testing_data=Path(testing_data),
            params_epochs=params["EPOCHS"],
            params_batch_size=params["BATCH_SIZE"],
            params_is_augmentation=params["AUGMENTATION"],
            params_image_size=params["IMAGE_SIZE"],
            params_lr_rate=params["LEARNING_RATE"]
        )

        return training_config

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()

        data_preprocessing = Data_preprocessing(config=training_config)
        train_set, test_set = data_preprocessing.data_normalization()
        train_loader, test_loader = data_preprocessing.data_lode()

        training = Training(train_loader, test_loader, train_set, config=training_config)
        training.get_base_model()
        # training.train_valid_generator()
        training.train() 

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e
        