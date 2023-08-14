import torch
import torch.nn as nn
from torchvision import models
from bird_classification.entity.config_entity import PrepareBaseModelConfig
from collections import OrderedDict

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_model(self):
        self.base_model=models.resnet152(weights=None)
        for param in self.base_model.parameters():
            param.required_grad=False
        print(self.config.base_model_path)
        self.save_model(self.config.base_model_path, self.base_model)

    # @staticmethod
    def model_fc_layer(self, classes):
        self.base_model.fc = nn.Sequential(OrderedDict([
                ('dp1', nn.Dropout(0.2)),
                ('fc1', nn.Linear(2048, classes)),
                ('out', nn.Softmax(dim=1))
        ]))
        return self.base_model

    def get_updated_BASE_MODEL(self):
        full_model = self.model_fc_layer(classes=self.config.params_classes)
        return full_model

    def update_base_model(self):
        self.full_model = self.model_fc_layer(classes=self.config.params_classes)
        # optimizer=torch.optim.SGD(self.full_model.parameters(),lr=self.config.params_learning_rate,momentum=self.config.params_momentum)
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    @staticmethod
    def save_model(path, model):
        torch.save(model,path)