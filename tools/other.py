# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import SubsetRandomSampler
from visdom import Visdom
from unet.unet_model import UNet
from tools.data_loading import *
from skimage import morphology
import cv2

def fill_holes2(mask):
    """
    Fill holes from mask matrix that less than 900 pixels in terms of area
      with ones in mask matrix and with mean color of image
      with applied mask in origin image
    Parameters
    ----------
    img: numpy array
      image which holes will be filled with mean color
    mask: numpy array
      image which holes will be filled with ones
    Returns
    -------
      pair of modified image and mask
    """
    mask = np.squeeze(mask)
    mod_mask = morphology.remove_small_holes(mask, 6000, connectivity=1)

    return mod_mask


def remove_little_blobs2(img):
    """
    Return image without blobs
    that less than 150 in terms of area (amount of pixels)
    Parameters
    ----------
    img: numpy array

    Returns
    -------

    """
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids =\
        cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component
    # with information on each of them, such as size
    # the following part is just taking out the background
    # which is also considered a component,
    # but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want,
    # eg the mean of the sizes or whatever
    # min_size = 500

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    # for i in range(0, nb_components):
    #     if sizes[i] >= min_size:
    #         img2[output == i + 1] = 255
    if sizes.shape[0] > 0:
        img2[output == np.argmax(sizes) + 1] = 1
    return img2


def remove_small_object_and_close_holes(sample):
    mask = remove_little_blobs2(sample["mask"].astype("uint8"))
    # mask = fill_holes2(mask.astype("bool"))
    return {'color': mask[:,:,np.newaxis] * sample["color"],
            'mask': mask.astype("uint8")}


def get_default_visdom_env():
    """
    Create and return default environment from visdom
    Returns
    -------
    Visdom
           Visdom class object
    """
    default_port = 8097
    default_hostname = "http://localhost"
    parser = argparse.ArgumentParser(description='Demo arguments')
    parser.add_argument('-port', metavar='port', type=int, default=default_port,
                        help='port the visdom server is running on.')
    parser.add_argument('-server', metavar='server', type=str,
                        default=default_hostname,
                        help='Server address of the target to run the demo on.')
    flags = parser.parse_args()
    viz = Visdom(port=flags.port, server=flags.server)

    assert viz.check_connection(timeout_seconds=3), \
        'No connection could be formed quickly'
    return viz


def pair_of_dataloaders(split_ratio, path_to_train_folder,path_to_val_folder,
                        transform,
                        batch_size=128, num_workers=4,
                        names=("train", "val")):
    """
    Take path to folder with data and percentage that defines,
    how much of them should be used as training data.
    Shuffle and split data. Returns random sampling
    DataLoader without replacement for training and validation sets
    and sizes of this sets.
    Parameters
    ----------
    transform:
        trasforms that applied to data during loading
    names: (string,string)
        Pair of names for DataLoaders
    split_ratio: float
        How much of data should be used as training data
    path_to_train_folder: string
        Valid path to training data.
    batch_size: int
        Size of batch for DataLoaders
    num_workers
        Number of threads for data prepossessing
    Returns
    -------
    Pair of dictionaries.
    First of them contains DataLoaders for validation and training sets.
    Second contains sizes of validation and training sets.
    """
    train_data_set = NewDataset(root_dir=path_to_train_folder,
                                      transforms=transform)
    shuffle_data_set = True
    random_seed = 42
    train_dataset_size = len(train_data_set)
    train_indices = list(range(train_dataset_size))
    if shuffle_data_set:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)

    val_data_set = NewDataset(root_dir=path_to_val_folder,
                                transforms=transform)
    shuffle_data_set = True
    random_seed = 42
    val_dataset_size = len(val_data_set)
    val_indices = list(range(val_dataset_size))
    if shuffle_data_set:
        np.random.seed(random_seed)
        np.random.shuffle(val_indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    dataset_sizes = {names[0]: train_dataset_size,
                     names[1]: val_dataset_size}

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val_data_set,
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler,
                                                    num_workers=num_workers)

    dataloaders = {names[0]: train_loader, names[1]: validation_loader}
    return dataloaders, dataset_sizes


def rgb_classif_model_and_criterion(device, weight, is_pretrained):
    """
    Create resnet18 and changes fully connected layer.
    Parameters
    ----------
    weight: torch vector
        weights for every class for loss function
    is_pretrained: bool
        Define is network`s weights are predefined or not
    device: torch.device
        Define CPU or GPU will be used for training
    Returns
    -------
    model architecture and criterion in tuple
    """
    model = torchvision.models.resnet18(pretrained=is_pretrained)
    num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(num_ftrs, 110), nn.BatchNorm1d(110),
    #                          nn.ReLU(), nn.Linear(110, 24))
    model.fc = nn.Linear(num_ftrs, len(get_name_of_classes()))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    return model, criterion


def gray_classif_model_and_criterion(device, weight,
                                     is_pretrained):
    """
    Create resnet18 and changes fully connected layer.
    Parameters
    ----------
    weight: torch vector
        weights for every class for loss function
    is_pretrained: bool
        Define is network`s weights are predefined or not
    device: torch.device
        Define CPU or GPU will be used for training
    Returns
    -------
    model architecture and criterion in tuple
    """
    model = torchvision.models.resnet34(pretrained=is_pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(num_ftrs, 110), nn.BatchNorm1d(110),
    #                          nn.ReLU(), nn.Linear(110, 24))
    model.fc = nn.Linear(num_ftrs, len(get_name_of_classes()))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    return model, criterion


def dice(pred, target):
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(iflat * iflat)
    result = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    iflat = -(1 - iflat)
    tflat = -(1 - tflat)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(iflat * iflat)
    return result + 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def get_segment_model_and_criterion(device):
    """
    Create U-NET and changes fully connected layer.
    Parameters
    ----------
    device: torch.device
        Define CPU or GPU will be used for training
    Returns
    -------
    model architecture and criterion in tuple
    """
    model = UNet(n_channels=3, n_classes=1)
    model = model.to(device)
    criterion = dice
    return model, criterion


def get_optimizer_and_scheduler(parameters, lr=0.3, momentum=0.9,
                                weight_decay=0.005, step_size=1, gamma=0.8):
    """
    Create and return SGD optimizer with arguments
    Parameters
    ----------
    parameters: weight of model that should be modified
    lr: float
        learning rate
    momentum: float
        momentum for SGD optimization
    weight_decay: float
        L2 regularization coefficient
    step_size: int
        define number steps after that learning rate will be multiplied by gamma
    gamma: float
        coefficient for learning rate decreasing
    Returns
    -------
    optimizer and scheduler in tuple
    """
    optimizer = optim.SGD(parameters, lr, momentum, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler


def get_segmented_image(image, prediction):
    """
    Apply segmentation on image using prediction map`s value
    Trashold 0.5
    Parameters
    ----------
    image: numpy array
        image to which segmentation are applied
    prediction
        map that shows confidence for corresponding pixel of image
    Returns
    -------
        image with applied segmentation
    """
    idx = np.argmax(prediction, 2)
    pred = np.max(prediction, 2)
    # pred = (pred - pred.min()) / (pred.max() - pred.min())
    confidence_mask = pred > 0.5
    # confidence_mask = remove_small_object_and_close_holes({'color': image, "mask": confidence_mask})
    return image * confidence_mask[:, :, np.newaxis]
    return image * (confidence_mask["mask"])[:, :, np.newaxis]


def remove_little_blobs(img):
    """
    Return image without blobs
    that less than 150 in terms of area (amount of pixels)
    Parameters
    ----------
    img: numpy array

    Returns
    -------

    """
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids =\
        cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component
    # with information on each of them, such as size
    # the following part is just taking out the background
    # which is also considered a component,
    # but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want,
    # eg the mean of the sizes or whatever
    # min_size = 500

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    # for i in range(0, nb_components):
    #     if sizes[i] >= min_size:
    #         img2[output == i + 1] = 255
    if sizes.shape[0] > 0:
        img2[output == np.argmax(sizes) + 1] = 1
    return img2