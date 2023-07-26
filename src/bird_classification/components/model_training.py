import os
# import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from src.bird_classification.entity.config_entity import TrainingConfig

class ImageDataset(Dataset):
    def __init__(self, data_folder, transform):
        self.data_folder = data_folder
        self.transform = transform
        self.classes = os.listdir(data_folder)
        self.image_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_folder, class_name)
            images = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, img) for img in images])
            self.labels.extend([class_idx] * len(images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            # print('before',image.size)
            image = self.transform(image)
            # print('after',image.size)
        return image, label
    
class Data_preprocessing:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_path = self.config.training_data
        self.test_path = self.config.testing_data
        # print("path--",self.config.training_data)
        
    def data_normalization(self):
        data_transforms = transforms.Compose([
        transforms.Resize(size=self.config.params_image_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        #Getting the data
        # print(self.train_path)
        self.train_set = ImageDataset(self.config.training_data, data_transforms)
        self.test_set = ImageDataset(self.config.testing_data, data_transforms)

        self.class_name=self.train_set.classes

        return self.train_set, self.test_set
    
    def data_lode(self):
        train_loader = DataLoader(self.train_set, batch_size=self.config.params_batch_size, shuffle=True)
        test_loader = DataLoader(self.test_set, batch_size=self.config.params_batch_size, shuffle=False)
        return train_loader, test_loader
    
class Training:
    def __init__(self, train_loader, test_loader, train_set, config: TrainingConfig):
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_set = train_set
        # print(self.config)
    def get_base_model(self):
        self.model = torch.load(
            self.config.updated_base_model_path
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.config.params_lr_rate)

    @staticmethod 
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def train(self):
        tb = SummaryWriter()
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)
        for epoch in range(self.config.params_epochs):

            total_loss = 0
            total_correct = 0

            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)

                loss = self.criterion(preds, labels)
                total_loss+= loss.item()
                total_correct+= self.get_num_correct(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tb.add_scalar("Loss", total_loss, epoch)
            tb.add_scalar("Correct", total_correct, epoch)
            tb.add_scalar("Accuracy", total_correct/ len(self.train_set), epoch)
            # print("parameters",model.named_parameters())
            for name, weight in model.named_parameters():
                # print("name&name",name)
                # print("weight&weight",weight)
                tb.add_histogram(name, weight, epoch)
                tb.add_histogram(f'{name}.grad',weight, epoch)
 
            print("epoch:", epoch, "total_correct:", total_correct, "loss:",total_loss, "Accuracy", total_correct/ len(self.train_set))

        tb.close()

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
    @staticmethod
    def save_model(path, model):
        print("saving model ...")
        torch.save(model,path)