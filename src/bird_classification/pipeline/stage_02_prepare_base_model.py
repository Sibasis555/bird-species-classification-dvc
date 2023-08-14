from bird_classification.utils import *
from bird_classification.utils.common import create_directories
from bird_classification.entity.config_entity import PrepareBaseModelConfig
from bird_classification import logger
from bird_classification.components.prepare_base_model import PrepareBaseModel

STAGE_NAME = "Prepare base model"

class ConfigurationManager: #Config Entity data initialization
    def __init__(self):

        self.config = CONFIG_FILE
        self.params = PARAMS_FILE
    
        create_directories([self.config["artifacts_root"]])


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config["prepare_base_model"]
        
        create_directories([config["root_dir"]])
        print(self.params)
        print(config["base_model_path"])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config["root_dir"]),
            base_model_path=Path(config["base_model_path"]),
            updated_base_model_path=Path(config["updated_base_model_path"]),
            params_image_size=self.params["IMAGE_SIZE"],
            params_learning_rate=self.params['LEARNING_RATE'],
            params_include_top=self.params["INCLUDE_TOP"],
            params_momentum=self.params["MOMENTUM"],
            params_classes=self.params["CLASSES"]
        )

        return prepare_base_model_config

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_model()
        prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e