import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
from evaluate.evaluate import *
import cv2
import argparse


def sequence_from_webcam(model_conv, device):
    """
    Grab frame from webcam,
    crop top left piece of it and predict class of gesture,
    Parameters
    ----------
    model_conv: model for testing
    device: CPU or GPU

    Returns
    -------

    """
    cap = cv2.VideoCapture(0)
    plt.ion()
    classes = os.listdir('../../prepared_data/test/A')
    classes.sort()
    while (True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        croped = rgb #crop_and_prepare(rgb, device)
        outputs = model_conv(croped)
        _, preds = torch.max(outputs, 1)

        #show_gesture_and_class(croped, classes[preds])
        cv2.imshow('frame', cv2.cvtColor(cv2.resize(rgb, (640, 400)),
                                         cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Recognize hand gesture for sign language")
    parser.add_argument("path to segmentation model", type=str)
    parser.add_argument("path to classification model", type=str)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (model_conv, criterion) = rgb_classif_model_and_criterion(device=device,
                                                              is_pretrained=False)
    model_conv.load_state_dict(torch.load("../saved_classification_model/simple_rgb.pt"))
    model_conv.eval()
    model_conv = model_conv.to(device)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
