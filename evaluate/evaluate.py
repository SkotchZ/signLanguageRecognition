# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
from visdom import Visdom
import matplotlib.pyplot as plt
from evaluate_help import *
import numpy as np
import torch
from tqdm import tqdm
from config import Config
from PIL import ImageFont, ImageDraw, Image
from sklearn.metrics import f1_score

import cv2


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy()
    if len(inp.shape) == 3:
        inp = inp.transpose((1, 2, 0))
    if inp.shape[2] == 1:
        inp = np.squeeze(inp)
    plt.imshow(inp, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)



def measure_segmentation_quality(model, dataloader, device):
    was_training = model.training
    model.eval()
    running_corrects = 0
    for data in tqdm(dataloader):
        inputs = data['color'].float().to(device)
        labels = data['mask'].to(device)
        outputs = model(inputs)
        pred, idx = torch.max(outputs, 1)
        confidence_mask = pred > 0.5
        idx += 1
        idx = idx * confidence_mask.long()
        corect_map, _ = torch.max(labels.data, 1)
        # statistics
        x1 = np.squeeze(data['mask'].cpu().numpy()).flatten().tolist()
        x2 = confidence_mask.cpu().numpy().flatten().tolist()

        running_corrects += f1_score(x1, x2)
    epoch_acc = running_corrects / len(dataloader)
    print('Acc: {:.8f}'.format(epoch_acc))
    model.train(mode=was_training)


def plot_heatmap(model, dataloader, device):
    """
    Create confusion matrix and measure accuracy on test set
    Parameters
    ----------
    model: model for testing
    dataloader: DataLoader for test set
    device: CPU of GPU
    viz: Visdom
        object created by Visdom constructor

    Returns
    -------

    """
    viz = get_default_visdom_env()
    was_training = model.training
    model.eval()
    name_of_classes = get_name_of_classes()
    nb_classes = len(name_of_classes)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data in tqdm(enumerate(dataloader)):
            inputs = data[1]['color'].float().to(device)
            classes = data[1]['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    confusion_matrix = confusion_matrix / torch.sum(confusion_matrix, dim=1)
    tmp = (confusion_matrix * torch.from_numpy(np.eye(nb_classes))
           .float()).sum() / nb_classes
    print(tmp)
    print(confusion_matrix)
    print(type(confusion_matrix))
    viz.heatmap(X=confusion_matrix,
                opts=dict(columnnames=name_of_classes,
                          rownames=name_of_classes,
                          colormap='Viridis',
                          layout=dict(title="Test acc: {}".format(tmp),
                                      xaxis={'Ground truth class': 'x1'},
                                      yaxis={'Predicted class': 'x2'})
                          )
                )
    model.train(mode=was_training)


def visualize_results(model, dataloader, device,
                      with_prediction, num_images=12):
    """
    Shows prediction for some images
    Parameters
    ----------
    model: model for testing
    dataloader:  DataLoader for visualisation set
    device: CPU or GPU
    num_images: maximum amount of images will be shown
    Returns
    -------

    """
    # plt.figure()
    was_training = model.training
    model.eval()
    images_so_far = 0
    names_of_classes = get_name_of_classes()
    with torch.no_grad():
        for data in enumerate(dataloader):
            inputs = data[1]['color'].float().to(device)

            # outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    break
                images_so_far += 1
                ax = plt.subplot(num_images // 6, 6, images_so_far)
                ax.axis('off')
                # if with_prediction:
                #     ax.set_title('predicted: {}'.
                #                  format(names_of_classes[preds[j]]))
                imshow(inputs.cpu().data[j])
        model.train(mode=was_training)
    plt.ioff()
    plt.show()


def test_and_visualize(classif_model, device, transforms):
    """
    Prepare splitting for testing and visualization and run them
    Parameters
    ----------
    classif_model: model for testing
    device: CPU or GPU

    Returns
    -------

    """

    with open('../config.cfg') as f:
        cfg = Config(f)
    root_dir = cfg.testing_data_path
    batch_size = 12
    test_split_ratio = 0.95
    dataloaders, _ = pair_of_dataloaders(split_ratio=test_split_ratio,
                                         transform=transforms,
                                         path_to_train_folder=root_dir,
                                         path_to_val_folder=root_dir,
                                         batch_size=batch_size,
                                         num_workers=4,
                                         names=("test", "visualize"))
    measure_segmentation_quality(classif_model, dataloaders["test"], device)
    # plot_heatmap(classif_model, dataloaders["test"], device)
    # visualize_results(classif_model, dataloaders["visualize"],
    #                   device, True)


def run_webcam_cycle(segment_model, classif_model, device, is_gray):
    """
    Grab frame from webcam,
    crop top left piece of it and predict class of gesture,
    Parameters
    ----------
    segment_model: model for testing
    device: CPU or GPU

    Returns
    -------

    """
    names_of_classes = get_name_of_classes()
    cap = cv2.VideoCapture(0)

    text = ""
    last_answer = " "
    i=0
    # plt.ion()
    while True:

        ret, frame = cap.read()
        # rgb =
        rgb = frame
        croped = apply_transforms({'color': img_as_float(img_as_float(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)))},
                                  prepoc_segmnet_webcam_transforms())

        if cv2.waitKey(1) & 0xFF == ord('p'):
            gr = skimage.color.rgb2gray(img_as_ubyte(croped['color']))
            plt.hist(img_as_ubyte(gr.copy()).ravel(), bins=256, range=(0, 256), fc='k', ec='k')
            plt.pause(0.01)  # I ain't needed!!!
            plt.show()


        # show the plotting graph of an image

        if is_gray:
            answer, segmented_image = gray_classif_pipeline(segment_model,
                                                            classif_model,
                                                            croped,
                                                            device,
                                                            names_of_classes)
        else:
            answer, segmented_image = rgb_classif_pipeline(segment_model,
                                                           classif_model,
                                                           croped,
                                                           device,
                                                           names_of_classes)
        if len(segmented_image.shape) == 3 and segmented_image.shape[2] == 3:
            segmented_image = img_as_ubyte(segmented_image)
        if segmented_image.sum() == 0:
            answer = "space"

        if answer!=last_answer:
            i += 1
            if i>30:
                text+=answer
                print(text)
                last_answer = answer
        else:
            i = 0
        font_size = 36
        font_color = (255, 0, 0)
        unicode_text = answer

        im = Image.fromarray(segmented_image)
        draw = ImageDraw.Draw(im)
        unicode_font = ImageFont.truetype("../DejaVuSans.ttf", font_size)
        draw.text((10, 10), unicode_text, font=unicode_font, fill=font_color)

        cv2.imshow("cropped", cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB))
        cv2.imshow('frame',cv2.resize(frame, (640, 400)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Measure quality of loaded model (confusion matrix and accuracy)
     or perform classification of gesture from webcam stream
    Returns
    -------

    """
    is_gray = False
    with open('../config.cfg') as f:
        cfg = Config(f)
    # plt.ion()
    is_webcam = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_gray:
        classif_model, _ = gray_classif_model_and_criterion(device,
                                                            weight=None,
                                                            is_pretrained=False)
        classif_model = prepare_model_for_test(classif_model,
                                               cfg.classif_weights_path_gray,
                                               device)
        transforms = gray_classif_training_transforms()
    else:
        classif_model, _ = rgb_classif_model_and_criterion(device,
                                                           weight=None,
                                                           is_pretrained=False)
        classif_model = prepare_model_for_test(classif_model,
                                               cfg.classif_weights_path_rgb,
                                               device)
        transforms = rgb_classif_training_transforms()

    segment_model, _ = get_segment_model_and_criterion(device)
    segment_model = prepare_model_for_test(segment_model,
                                           cfg.segment_weights_path,
                                           device)

    if is_webcam:
        run_webcam_cycle(segment_model, classif_model, device, is_gray)
    else:
        test_and_visualize(segment_model, device, segmnet_test_transforms())


if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
