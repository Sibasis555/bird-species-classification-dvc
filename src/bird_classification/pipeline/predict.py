import numpy as np
import torch
from torchvision import transforms,datasets
from PIL import Image
import os
import cv2
from src.bird_classification.utils.common import create_directories
from src.bird_classification.utils import *

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename
        self.config = CONFIG_FILE
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_model(trained_model_path, path_of_base_model):
        # model = PrepareBaseModel.get_updated_BASE_MODEL()
        # model = torch.load(path_of_base_model)
        # print(model)
        return torch.load(trained_model_path)
    
    def data_noramalization(self):
        self.data_transforms = transforms.Compose([
        transforms.Resize(size=(224,224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        return self.data_transforms

    def get_classname(self):
        # self.data_transforms = self.data_noramalization()
        train_set=datasets.ImageFolder(self.config["training"]["train_data"])
        class_name=train_set.classes
        return class_name
        
    def predict(self):
        self.model = self.load_model(self.config["prepare_base_model"]["updated_base_model_path"], self.config["training"]["trained_model_path"])
        self.model.eval()
        class_name = self.get_classname()

        data_transforms=transforms.Compose([
                              transforms.ToPILImage(),
                              transforms.Resize(224),
                              # transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        
        
        imagename = cv2.imread(self.filename)
        imagename=cv2.cvtColor(imagename,cv2.COLOR_BGR2RGB)
        # print(imagename)
        image_data = data_transforms(imagename)
        image_data = image_data.to(self.device)
        image_data = image_data.unsqueeze(0)
        
        with torch.no_grad():
            preds=self.model(image_data)
            pred_retinopathy=int(torch.argmax(preds))
            # print(class_name[pred_retinopathy].split('/')[-1])
            return (class_name[pred_retinopathy].split('/')[-1])
        
