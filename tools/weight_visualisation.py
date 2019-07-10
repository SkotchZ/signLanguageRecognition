import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
import scipy.misc
import matplotlib.pyplot as plt
from tools.other import *
from config import Config


def main():
    """

    -------

    """
    plt.ion()
    with open('../config.cfg') as f:
        cfg = Config(f)
    device = torch.device("cpu")
    (model_conv, criterion) = \
        rgb_classif_model_and_criterion(device=device,
                                        is_pretrained=False)
    model_conv.load_state_dict(torch.load(cfg.classif_model_path))
    model_conv.eval()
    model_conv = model_conv.to(device)
    viz = get_default_visdom_env()
    for m in model_conv.modules():
        if isinstance(m, nn.Conv2d):
            kernels = m.weight.data.numpy()
            for kernel_idx0 in range(0, kernels.shape[0] - 1):
                for kernel_idx1 in range(0, kernels.shape[1] - 1):
                    if kernels.shape[1] == 3 and kernels.shape[2] != 1:
                        kernel = kernels[kernel_idx0]
                        kernel = (kernel * (255 / kernel.max())).astype('uint8')
                        kernel = kernel.transpose(1, 2, 0)
                        kernel = scipy.misc.imresize(kernel,
                                                     (50, 50),
                                                     interp="nearest")
                        kernel = kernel.transpose(2, 0, 1)
                        viz.image(kernel)
                        break
                    elif kernel_idx1 % 50 == 0\
                            and kernel_idx0 % 50 == 0\
                            and kernels.shape[2] != 1:
                        kernel = kernels[kernel_idx0, kernel_idx1]
                        kernel = (kernel * (255 / kernel.max())).astype('uint8')
                        kernel = scipy.misc.imresize(kernel, (50, 50),
                                                     interp="nearest")
                        viz.image(kernel)


if __name__ == '__main__':
    main()
