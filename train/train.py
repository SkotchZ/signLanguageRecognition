# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
import copy
import time
from tqdm import tqdm
from visdom import Visdom
from tools.transformers import *
from config import Config
import tools.other
import sklearn.metrics


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def train_classifier(model, criterion, optimizer, scheduler, dataloaders,
                     device, viz, dataset_sizes, num_epochs=25):
    """
    Perform num_epochs steps of SGD
    Parameters
    ----------
    model: model of network
    criterion: loss function
    optimizer: algorithm of optimization
    scheduler: scheduler for learning rate decrease
    dataloaders: (torch.utils.data.DataLoader,torch.utils.data.DataLoader)
        pair of DataLoader for test and validation sets
    device: CPU or GPU
    viz: Visdom
        object created by Visdom constructor
    dataset_sizes: (int,int)
        pair of test set size and validation set size
    num_epochs
        amount of epochs
    Returns
    -------
        trained model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_history = {'train': np.array([]), 'val': np.array([])}
    win = viz.line(
        X=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
        Y=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
    )
    loss_history = {'train': np.array([]), 'val': np.array([])}
    win2 = viz.line(
        X=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
        Y=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
    )
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for data in tqdm(dataloaders[phase]):
                inputs = data['color'].float().to(device)
                labels = data['label'].long().to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                x1 = labels.cpu().numpy().tolist()
                x2 = preds.cpu().numpy().tolist()
                running_corrects += sklearn.metrics.f1_score(x1, x2, average="micro")
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / len(dataloaders[phase])
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # temp = epoch_acc.cpu().item()
            acc_history[phase] = np.append(acc_history[phase], epoch_acc)
            loss_history[phase] = np.append(loss_history[phase], epoch_loss)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        viz.line(
            X=np.column_stack(
                (np.arange(0, acc_history['train'].shape[0]),
                 np.arange(0, acc_history['val'].shape[0]))),
            Y=np.column_stack((acc_history['train'],
                               acc_history['val'])),
            win=win,
            update='insert',
            opts=dict(title='{}'.format(optimizer.defaults))
        )
        viz.line(
            X=np.column_stack(
                (np.arange(0, loss_history['train'].shape[0]),
                 np.arange(0, loss_history['val'].shape[0]))),
            Y=np.column_stack((loss_history['train'],
                               loss_history['val'])),
            win=win2,
            update='insert',
            opts=dict(title='{}'.format(optimizer.defaults))
        )
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.
          format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_segmentator(model, criterion, optimizer, scheduler, dataloaders,
                      device, viz, dataset_sizes, num_epochs=25):
    """
    Perform num_epochs steps of SGD
    Parameters
    ----------
    model: model of network
    criterion: loss function
    optimizer: algorithm of optimization
    scheduler: scheduler for learning rate decrease
    dataloaders: (torch.utils.data.DataLoader,torch.utils.data.DataLoader)
        pair of DataLoader for test and validation sets
    device: CPU or GPU
    viz: Visdom
        object created by Visdom constructor
    dataset_sizes: (int,int)
        pair of test set size and validation set size
    num_epochs
        amount of epochs
    Returns
    -------
        trained model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_history = {'train': np.array([]), 'val': np.array([])}
    win = viz.line(
        X=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
        Y=np.column_stack((np.arange(0, 1), np.arange(0, 1))),
    )
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for data in tqdm(dataloaders[phase]):
                inputs = data['color'].float().to(device)
                labels = data['mask'].to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    pred, idx = torch.max(outputs, 1)
                    print(pred.min(), pred.max())
                    w1 = torch.sum(labels).float() / torch.prod(torch.tensor(labels.shape)).float()
                    w0 = 1 - w1
                    loss = weighted_binary_cross_entropy(outputs, labels.float(), torch.tensor([1/w0, 1/w1]))
                    confidence_mask = pred > 0.5
                    idx += 1
                    idx = idx * confidence_mask.long()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                corect_map, _ = torch.max(labels.data, 1)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                x1 = np.squeeze(data['mask'].cpu().numpy()).flatten().tolist()
                x2 = confidence_mask.cpu().numpy().flatten().tolist()
                tmp = sklearn.metrics.f1_score(x1, x2)
                print(tmp)
                running_corrects += tmp
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / len(dataloaders[phase]) #/ (dataset_sizes[phase] // data['mask'].shape[0])


            print('{} Loss: {:.4f} Acc: {:.8f}'.format(
                phase, epoch_loss, epoch_acc))
            # temp = epoch_acc.cpu().item()
            acc_history[phase] = np.append(acc_history[phase], epoch_acc)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        viz.line(
            X=np.column_stack(
                (np.arange(0, acc_history['train'].shape[0]),
                 np.arange(0, acc_history['val'].shape[0]))),
            Y=np.column_stack((acc_history['train'],
                               acc_history['val'])),
            win=win,
            update='insert',
            opts=dict(title='{}'.format(optimizer.defaults))
        )
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.
          format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def resnet18(is_gray):
    """
    Create, train and save classification model.
    Returns
    -------

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('../config.cfg') as f:
        cfg = Config(f)
    train_data_path = cfg.training_data_path
    val_data_path = cfg.validation_data_path
    if is_gray:
        cur_transforms = gray_classif_training_transforms()
        cur_weights_path = cfg.classif_weights_path_gray
    else:
        cur_transforms = rgb_classif_training_transforms()
        cur_weights_path = cfg.classif_weights_path_rgb
    batch_size = 128
    train_split_ratio = 0.8
    is_pretrained = True
    # treshold_ratio_dsd = 0.25
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.01
    step_size = 2
    gamma = 0.5
    num_epochs = 10
    viz = get_default_visdom_env()
    (dataloaders, dataset_sizes) = pair_of_dataloaders(train_split_ratio,
                                                       train_data_path,
                                                       val_data_path,
                                                       cur_transforms,
                                                       batch_size, 12)

    weights_per_class = list(dataloaders['train'].dataset.samples_per_class.values())
    min_weight = np.max(weights_per_class)
    weights_per_class = torch.tensor(weights_per_class)
    weights_per_class = (1 / weights_per_class * min_weight).to(device)
    if is_gray:
        (model, criterion) = gray_classif_model_and_criterion(device,
                                                              weights_per_class,
                                                              is_pretrained)
    else:
        (model, criterion) = rgb_classif_model_and_criterion(device,
                                                             weights_per_class,
                                                             is_pretrained)

    model = model.to(device)
    # model = prepare_model(model,cfg.classif_weights_path_rgb, device)
    (optimizer, scheduler) = get_optimizer_and_scheduler(model.parameters(),
                                                         lr, momentum,
                                                         weight_decay,
                                                         step_size, gamma)

    train_classifier(model, criterion, optimizer, scheduler, dataloaders,
                     device, viz, dataset_sizes, num_epochs)
    torch.save(model.state_dict(), cur_weights_path)


def unet():
    """
    Create, train and save segmentation model.
    Returns
    -------

    """
    with open('../config.cfg') as f:
        cfg = Config(f)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    train_split_ratio = 0.8
    # treshold_ratio_dsd = 0.25
    lr = 0.001
    momentum = 0.95
    weight_decay = 0.24
    step_size = 1
    gamma = 0.1
    num_epochs = 5
    viz = get_default_visdom_env()

    (dataloaders, dataset_sizes) =\
        pair_of_dataloaders(train_split_ratio, cfg.training_data_path, cfg.validation_data_path,
                            segment_training_transforms(),
                            batch_size=batch_size, num_workers=12)

    (model, criterion) = get_segment_model_and_criterion(device)

    model = model.to(device)

    (optimizer, scheduler) = get_optimizer_and_scheduler(model.parameters(),
                                                         lr, momentum,
                                                         weight_decay,
                                                         step_size, gamma)
    # segment_model = prepare_model_for_test(segment_model,
    #                                        cfg.segment_weights_path,
    #                                        device)

    train_segmentator(model, criterion, optimizer, scheduler, dataloaders,
                      device, viz, dataset_sizes, num_epochs)
    torch.save(model.state_dict(), cfg.segment_weights_path)


# def prepare_model(model, path, device):
#     """
#     Set model on test regime
#     Parameters
#     ----------
#     model:
#
#     path: string
#         path to model`s weights
#     device: CPU or GPU
#
#     Returns
#     -------
#     Prepared model
#     """
#     model.load_state_dict(torch.load(path))
#     model = model.to(device)
#     return model

if __name__ == '__main__':
    print(torch.cuda.is_available())
    # unet()
    resnet18(is_gray=False)
    #

