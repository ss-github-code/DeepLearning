from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision import transforms

@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = False  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


@dataclass
class DatasetConfig:
    root_dir: str = "."  # dataset directory root
    split: float = 0.8
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resiz = 256
    inpsz = 224

    train_transforms: Iterable[Callable] = transforms.Compose([
        #transforms.Resize(resiz),
        #transforms.RandomAffine(degrees=15, translate=(0.25,0.25), scale=(0.5,1.5)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(inpsz),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3,0.3,0.3),
        transforms.RandomAffine(degrees=15, translate=(0.25,0.25), scale=(0.5,1.5)),
        #transforms.CenterCrop(inpsz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])  # data transformation to use during training data preparation
    test_transforms: Iterable[Callable] = transforms.Compose([
        transforms.Resize(resiz),
        transforms.CenterCrop(inpsz),
        transforms.ToTensor(), # this re-scales image tensor values between 0-1. image_tensor /= 255
        # subtract mean (0.485, 0.456, 0.406) and divide by variance (0.229, 0.224, 0.225)
        transforms.Normalize(mean, std),
    ])  # data transformation to use during test data preparation

@dataclass
class DataloaderConfig:
    batch_size: int = 32  # amount of data to pass through the network at each forward-backward iteration
    num_workers: int = 8  # number of concurrent processes using to prepare data


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.0001  # determines the speed of network's weights update
    momentum: float = 0.9  # used to improve vanilla SGD algorithm and provide better handling of local minimas
    weight_decay: float = 0.0001  # amount of additional regularization on the weights values
    lr_step_milestones: Iterable = (
        30, 40
    )  # at which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
    lr_gamma: float = 0.1  # multiplier applied to current learning rate at each of lr_step_milestones


@dataclass
class TrainerConfig:
    model_dir: str = "checkpoints"  # directory to save model states
    model_saving_frequency: int = 1  # frequency of model state savings per epochs
    device: str = "cpu"  # device to use for training.
    epoch_num: int = 50  # number of times the whole dataset will be passed through the network
    progress_bar: bool = True  # enable progress bar visualization during train process
