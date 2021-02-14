from operator import itemgetter
import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
#from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import lr_scheduler

from trainer import Trainer, hooks, configuration
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

class KenyanFood13Dataset(torch.utils.data.Dataset):
    """
    This custom dataset class takes root directory, flag, and transform 
    and returns training dataset if flag is 0, validation dataset if flag is 1 
    else it returns test dataset.
    """
    def __init__(self, data_root, flag, split, transform, random_state):
        labels_to_idx = {'githeri' : 0, 'ugali' : 1, 'kachumbari' : 2, 'matoke' : 3, 
                         'sukumawiki' : 4, 'bhaji' : 5, 'mandazi' : 6, 
                         'kukuchoma' : 7, 'nyamachoma' : 8, 'pilau' : 9, 
                         'chapati' : 10, 'masalachips' : 11, 'mukimo' : 12}
        self.idx_to_labels = {0 : 'githeri', 1 : 'ugali', 2 : 'kachumbari', 3 : 'matoke',
                              4 : 'sukumawiki', 5 : 'bhaji', 6 : 'mandazi',
                              7 : 'kukuchoma', 8 : 'nyamachoma', 9 : 'pilau',
                              10 : 'chapati', 11 : 'masalachips', 12 : 'mukimo'}
        self.data_root = data_root
        self.transform = transform
        self.imgs = []
        self.labels = []
        self.weights = None
        
        if flag == 0 or flag == 1:
            
            data_csv = pd.read_csv(os.path.join(data_root, 'train.csv'))
            
            food_category = {}
            
            for idx, row in data_csv.iterrows():
                key = row['class']
                if key not in food_category.keys():
                    food_category[key] = []
                food_category[key].append(row['id'])
            
            num_classes = len(food_category.keys())
            assert num_classes == 13
            
            for category in food_category.keys():
                #random.seed(random_state)
                #random.shuffle(food_category[category])
                
                count = len(food_category[category])
                split_idx = math.floor(count*split)
                if flag == 0:
                    self.imgs.extend(food_category[category][:split_idx])
                    count = len(food_category[category][:split_idx])
                    self.labels.extend([labels_to_idx[category]]*count)
                else:
                    self.imgs.extend(food_category[category][split_idx:])
                    count = len(food_category[category][split_idx:])
                    self.labels.extend([labels_to_idx[category]]*count)
                
                self.weights = compute_class_weight('balanced', 
                                                    classes=np.unique(self.labels), 
                                                    y=self.labels)
        else: # flag == 2
            data_csv = pd.read_csv(os.path.join(data_root, 'test.csv'))
            self.imgs.extend(data_csv['id'])
            self.labels = None
    
    def __len__(self):
        """
        return length of the dataset
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        For given index, return images with resize and preprocessing.
        """
        img_name = str(self.imgs[idx]) + '.jpg'
        img_path = os.path.join(self.data_root, 'images', 'images', img_name)
        image = Image.open(img_path).convert("RGB")
        target = None
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            target = self.labels[idx]
        else:
            target = str(self.imgs[idx])
        return image, target

    def get_class_weight(self):
        return self.weights

#
# Get the pretrained model:
def pretrained_resnext50(pretrained=True, fine_tune_start=1, num_class=13):
    resnet = models.resnext50_32x4d(pretrained=pretrained)
    
    if pretrained:
        for param in resnet.parameters():
            param.requires_grad = False

    if pretrained:
        if fine_tune_start <= 1:
            for param in resnet.layer1.parameters():
                param.requires_grad = True
        if fine_tune_start <= 2:
            for param in resnet.layer2.parameters():
                param.requires_grad = True
        if fine_tune_start <= 3:
            for param in resnet.layer3.parameters():
                param.requires_grad = True
        if fine_tune_start <= 4:
            for param in resnet.layer4.parameters():
                param.requires_grad = True
                
    last_layer_in = resnet.fc.in_features
    resnet.fc = nn.Sequential(nn.Linear(last_layer_in, 256),
                              nn.ReLU(inplace=True),
                              nn.Dropout(0.5),
                              nn.Linear(256, 32),
                              nn.ReLU(inplace=True),
                              nn.Dropout(0.2),
                              nn.Linear(32, num_class))
    
    return resnet

#
# Define the experiment with the given model and given data. It's the same idea again: 
# we keep the less-likely-to-change things inside the object and configure it with the things 
# that are more likely to change.
#
# You may wonder, why do we put the specific metric and optimizer into the experiment code and 
# not specify them as parameters. 
# But the experiment class is just a handy way to store all the parts of your experiment in one place. 
# If you change the loss function, or the optimizer, or the model - it seems like another experiment.
# So it deserves to be a separate class.
#
class Experiment:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataset_config: configuration.DatasetConfig = configuration.DatasetConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
        optimizer_config: configuration.OptimizerConfig = configuration.OptimizerConfig()
    ):

        # train dataloader
        train_dataset = KenyanFood13Dataset(dataset_config.root_dir, flag=0, 
                                            split=dataset_config.split,
                                            transform=dataset_config.train_transforms, 
                                            random_state=system_config.seed)
        class_weight = train_dataset.get_class_weight()
        self.loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=True,
            num_workers=dataloader_config.num_workers
        )
        
        # validation dataloader
        val_dataset = KenyanFood13Dataset(dataset_config.root_dir, flag=1, 
                                          split=dataset_config.split,
                                          transform=dataset_config.test_transforms, 
                                          random_state=system_config.seed)
        self.loader_test = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            num_workers=dataloader_config.num_workers
        )


        setup_system(system_config)
        self.model = pretrained_resnext50(pretrained=True, fine_tune_start=4)

        self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight))
        self.metric_fn = AccuracyEstimator(topk=(1, ))
        #self.optimizer = optim.SGD(
        #    self.model.parameters(),
        #    lr=optimizer_config.learning_rate,
        #    weight_decay=optimizer_config.weight_decay,
        #    momentum=optimizer_config.momentum
        #)
        #self.lr_scheduler = MultiStepLR(
        #    self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        #)
        self.optimizer = optim.Adam(self.model.parameters())
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.visualizer = TensorBoardVisualizer()

    def run(self, trainer_config: configuration.TrainerConfig) -> dict:

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),
            target_getter=itemgetter(1),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_classification)
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics

class ExperimentF:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataset_config: configuration.DatasetConfig = configuration.DatasetConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
        optimizer_config: configuration.OptimizerConfig = configuration.OptimizerConfig()
    ):

        # train dataloader
        train_dataset = KenyanFood13Dataset(dataset_config.root_dir, flag=0, 
                                            split=1.0,
                                            transform=dataset_config.train_transforms, 
                                            random_state=system_config.seed)
        class_weight = train_dataset.get_class_weight()
        self.loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=True,
            num_workers=dataloader_config.num_workers
        )
        

        setup_system(system_config)
        self.model = pretrained_resnext50(pretrained=True, fine_tune_start=4)
        self.model.load_state_dict(torch.load('test/model_39_0.917'))
        
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight))
        self.metric_fn = AccuracyEstimator(topk=(1, ))
        #self.optimizer = optim.SGD(
        #    self.model.parameters(),
        #    lr=optimizer_config.learning_rate,
        #    weight_decay=optimizer_config.weight_decay,
        #    momentum=optimizer_config.momentum
        #)
        #self.lr_scheduler = MultiStepLR(
        #    self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        #)
        self.optimizer = optim.AdamW(self.model.parameters())
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001,
                                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        self.visualizer = TensorBoardVisualizer()

    def run(self, trainer_config: configuration.TrainerConfig) -> dict:

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=None,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),
            target_getter=itemgetter(1),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        #model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_classification)
        self.metrics = model_trainer.fit0(20)
        return self.metrics
    
from tqdm import tqdm
class Test:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataset_config: configuration.DatasetConfig = configuration.DatasetConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
    ):
        # test dataloader
        test_dataset = KenyanFood13Dataset(dataset_config.root_dir, flag=2,
                                           split=1.0,
                                           transform=dataset_config.test_transforms, 
                                           random_state=system_config.seed)
        self.loader_test = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            num_workers=dataloader_config.num_workers
        )
        self.idx_to_labels = test_dataset.idx_to_labels
        self.model = pretrained_resnext50(pretrained=False)
        self.model.load_state_dict(torch.load('test/model_17'))
        
    def run(self, trainer_config: configuration.TrainerConfig):
        device = torch.device(trainer_config.device)
        
        # set model to eval
        self.model.eval()
        self.model = self.model.to(device)
        data_getter=itemgetter(0)
        imgname_getter=itemgetter(1)
        iterator = tqdm(self.loader_test, disable=not trainer_config.progress_bar, 
                        dynamic_ncols=True)
        preds = []
        imgs = []
        for i, sample in enumerate(iterator):
            inputs = data_getter(sample).to(device)
            names = imgname_getter(sample)
            with torch.no_grad():
                predict = self.model(inputs)
            _, predict = torch.max(predict.cpu(), dim=1)
            #predict = [self.idx_to_labels[i] for i in predict.tolist()]
            preds.extend(predict.tolist())
            imgs.extend(names)
        
        preds = [self.idx_to_labels[i] for i in preds]
        preds = np.c_[preds]
        imgs = np.c_[imgs]
        df = pd.DataFrame({'id': imgs[:,0], 'class': preds[:,0]})
        df.to_csv('submission.csv', index=False)                

def main():
    '''Run the experiment
    '''
    # patch configs depending on cuda availability
    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=15)
    dataset_config = configuration.DatasetConfig(root_dir=".")
    #experiment = ExperimentF(dataset_config=dataset_config, dataloader_config=dataloader_config)
    #results = experiment.run(trainer_config)
    test = Test(dataset_config=dataset_config, dataloader_config=dataloader_config)
    test.run(trainer_config)

    #return results

if __name__ == '__main__':
    main()
