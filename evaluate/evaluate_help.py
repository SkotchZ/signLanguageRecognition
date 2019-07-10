# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
import skimage.transform
from tools.transformers import *


def gray_classif_pipeline(segment_model, classif_model,
                          sample, device, names_of_classes):
    """
    Process segmentation followed by classification
    to classify grayscale image
    Parameters
    ----------
    segment_model:
        model for segmentation
    classific_model
        model for classification that takes grayscale image
    sample
        image to classify
    device
        CPU or GPU
    names_of_classes
        list on names of classes
    Returns
    -------
    Pair of predicted class and segmented image
    """
    sample = sample['color'][np.newaxis, :, :, :].float().to(device)
    outputs = segment_model(sample)
    outputs_cpu = outputs.cpu().numpy()
    outputs_cpu = np.squeeze(outputs_cpu, 0)
    outputs_cpu = outputs_cpu.transpose((1, 2, 0))
    croped_on_cpu = np.squeeze(sample.cpu().numpy()).transpose((1, 2, 0))
    segmented_image = get_segmented_image(croped_on_cpu, outputs_cpu).copy()
    segmented_image = img_as_ubyte(color.rgb2gray(segmented_image))
    mask = (segmented_image > 8).astype("uint8")
    mask = remove_little_blobs(mask).astype("uint8") // 255
    segmented_image = segmented_image * mask
    tmp = img_as_ubyte(skimage.transform.resize(segmented_image,
                                                (224, 224)))
    segmented_image_torch = torch.from_numpy(tmp[:, :, np.newaxis].transpose((2, 0, 1)))
    segmented_image_torch = segmented_image_torch.unsqueeze(0).float()
    segmented_image_torch = segmented_image_torch.to(device)

    outputs = classif_model(segmented_image_torch)

    # print(torch.nn.Sigmoid()(outputs))
    _, preds = torch.max(outputs, 1)
    answer = names_of_classes[preds]
    return answer, segmented_image


def rgb_classif_pipeline(segment_model, classific_model,
                         sample, device, names_of_classes):
    """
    Process segmentation followed by classification to classify rgb image
    Parameters
    ----------
    segment_model:
        model for segmentation
    classific_model
        model for classification that takes rgb image
    sample
        image to classify
    device
        CPU or GPU
    names_of_classes
        list on names of classes
    Returns
    -------
    Pair of predicted class and segmented image
    """
    croped = sample['color'][np.newaxis, :, :, :].float().to(device)
    outputs = segment_model(croped)

    outputs_on_cpu = outputs.cpu().numpy()
    outputs_on_cpu = np.squeeze(outputs_on_cpu, 0)
    outputs_on_cpu = outputs_on_cpu.transpose((1, 2, 0))
    croped_on_cpu = np.squeeze(croped.cpu().numpy()).transpose((1, 2, 0))
    segmented_image = get_segmented_image(croped_on_cpu, outputs_on_cpu).copy()
    segmented_image_torch = torch.from_numpy(img_as_ubyte(
        skimage.transform.resize(segmented_image, (224, 224))).transpose(
        (2, 0, 1)))
    segmented_image_torch = segmented_image_torch.unsqueeze(0).float().to(
        device)
    outputs = classific_model(segmented_image_torch)

    # print(torch.nn.Sigmoid()(outputs))
    _, preds = torch.max(outputs, 1)
    answer = names_of_classes[preds]
    return answer, segmented_image


def prepare_model_for_test(model, path, device):
    """
    Set model on test regime
    Parameters
    ----------
    model:

    path: string
        path to model`s weights
    device: CPU or GPU

    Returns
    -------
    Prepared model
    """
    model.load_state_dict(torch.load(path))
    model.train(False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model = model.to(device)
    return model
