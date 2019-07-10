# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(__file__, "../../research"))
from skimage import transform, color, exposure,\
    morphology, img_as_ubyte, img_as_float, restoration
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import random
import cv2
from tools.other import *
from config import Config
from skimage import filters


def rgb_classif_training_transforms():
    """
    Returns
    Composition of transformation that applied during training of classifier
    """
    return torchvision.transforms.Compose([MaskCreatorTransform(),
                                           RotateTranform(),
                                           RandomHReflectTransformer(),
                                           ResizeTransformer((224, 224)),
                                           PaddingTransformer('constant', extra=True),
                                           ResizeTransformer((224, 224)),
                                           RandomGammaTransformer(),
                                           GaussianNoiseAdderTransform(0.01, 0.1, using_mask=True),
                                           BlurTransformer(),
                                           ToTensorTransformer()])


def gray_classif_training_transforms():
    """
    Returns
    Composition of transformation that applied during training of classifier
    """
    return torchvision.transforms.Compose([MaskCreatorTransform(),
                                           RotateTranform(),
                                           RandomHReflectTransformer(),
                                           ResizeTransformer((224, 224)),
                                           PaddingTransformer('constant', extra=True),
                                           ResizeTransformer((224, 224)),
                                           RandomGammaTransformer(),
                                           GaussianNoiseAdderTransform(0.01, 0.1, using_mask=True),
                                           BlurTransformer(),
                                           ToGrayscaleTransform(),
                                           ToTensorTransformer()])


def segment_training_transforms():
    """
    Returns
    Composition of transformation that applied during training of segmentation
    """
    with open('../config.cfg') as f:
        cfg = Config(f)
    return torchvision.transforms.Compose([MaskCreatorTransform(),
                                           RandomGammaTransformer(),
                                           RandomHReflectTransformer(),
                                           RotateTranform(),
                                           PaddingTransformer('constant'),
                                           ShiftAndPaddransformer('constant'),
                                           ResizeTransformer((200, 200)),
                                           BackgroundAddTransformer
                                           (cfg.backgrounds_path),
                                           # NegativeSkinTransformer(cfg.negative_images_path),
                                           GaussianNoiseAdderTransform(0.01, 0.03, using_mask=False),
                                           BlurTransformer(),
                                           ToTensorTransformer()])

def segmnet_test_transforms():
    """
    Returns
    Composition of transformation that applied during testing of segmentation
    """
    with open('../config.cfg') as f:
        cfg = Config(f)
    return torchvision.transforms.Compose([MaskCreatorTransform(),
                                           RandomGammaTransformer(),
                                           RandomHReflectTransformer(),
                                           RotateTranform(),
                                           # PaddingTransformer('constant'),
                                           ShiftAndPaddransformer('constant'),
                                           ResizeTransformer((200, 200)),
                                           BackgroundAddTransformer
                                           (cfg.backgrounds_path),
                                           # NegativeSkinTransformer(cfg.negative_images_path),
                                           GaussianNoiseAdderTransform(0.01, 0.03, using_mask=False),
                                           BlurTransformer(),
                                           ToTensorTransformer()])


def classif_testing_transforms():
    """
    Returns
    Composition of transformation that applied during testing of classifier
    """
    return torchvision.transforms.Compose([MaskCreatorTransform(),
                                           BlurTransformer(),
                                           PaddingTransformer('constant', extra=True),
                                           GaussianNoiseAdderTransform(0.02,
                                                                       0.021,
                                                                       using_mask=True),
                                           BlurTransformer(0.05),
                                           ResizeTransformer((224, 224)),
                                           ToTensorTransformer()])


def prepoc_segmnet_webcam_transforms():
    """
    Returns
    Composition of transformation that applied during testing of segmentation
    """
    return torchvision.transforms.Compose([CropTransform((240, 50), (640, 450)),
                                           ResizeTransformer((200, 200)),
                                           GaussianNoiseAdderTransform(0.01, 0.03, using_mask=False),
                                           BlurTransformer(),
                                           ToTensorTransformer()])


def apply_transforms(img, transforms):
    """
    Apply composition of transforms to image
    Parameters
    ----------
    img: numpy array
        image to which the transforms are applied
    transforms: torch.transforms.Compose
        Composition of transforms
    Returns
    -------
        image with applied transforms
    """
    return transforms(img)


def max_sub_square(m):
    """
    Find coordinates of biggest square of ones in matrix
    Parameters
    ----------
    m: numpy array
        matrix with ones inside

    Returns
    -------
        pair of coordinates representing left-top
        and bottom-right bound of square
    """
    r = len(m)
    c = len(m[0])

    s = [[0 for k in range(c)] for l in range(r)]
    # here we have set the first row and column of s[][]

    # Construct other entries
    for i in range(1, r):
        for j in range(1, c):
            if m[i][j] == 1:
                s[i][j] = min(s[i][j - 1], s[i - 1][j],
                              s[i - 1][j - 1]) + 1
            else:
                s[i][j] = 0

    # Find the maximum entry and indices of maximum entry in s[][]
    max_of_s = s[0][0]
    max_i = 0
    max_j = 0
    for i in range(r):
        for j in range(c):
            if max_of_s < s[i][j]:
                max_of_s = s[i][j]
                max_i = i
                max_j = j

    return (max_i - max_of_s, max_j - max_of_s), (max_i, max_j)


def cut_borders(img):
    """
    Cut 3 pixels from each side of image
    Parameters
    ----------
    img: numpy array
        image to which the cut are applied

    Returns
    -------
        Cutted image
    """
    return img[3:img.shape[0] - 3, 3:img.shape[1] - 3, :]


def fill_holes_with_median_color(img, mask):
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
    usefull_pixels = img[mask]
    pixels_value = np.sum(usefull_pixels, axis=1)
    pixels_order = np.argsort(pixels_value)
    usefull_pixels = usefull_pixels[pixels_order]
    median_color = usefull_pixels[usefull_pixels.shape[0] // 2]
    mod_mask = morphology.remove_small_holes(mask, 1100, connectivity=1)
    temp = np.broadcast_to(np.logical_xor(mask, mod_mask)[:, :, np.newaxis],
                           (mask.shape[0], mask.shape[1], 3)).astype('uint8')
    img = img + (median_color * temp)
    return img, mod_mask


def padding(padding_type, extra, *args):
    """
    Apply padding to rectangle image to make it square image
    Also apply some extra padding to make object on image smaller
    Parameters
    ----------
    img: numpy array

    padding_type:
        'constant' for zero padding
        'edge' for edge padding
    Returns
    -------
        image with applied transforms
    """
    if extra is None:
        extra_pad = 0
    else:
        extra_pad = random.randint(0, (args[0].shape[0] + args[0].shape[1]) // 4)
    extra_pad_top = random.randint(0, extra_pad)
    extra_pad_left = random.randint(0, extra_pad)
    extra_pad_bottom = extra_pad - extra_pad_top
    extra_pad_right = extra_pad - extra_pad_left
    result = []
    for img in args:
        if img.shape[0] < img.shape[1]:
            diff = img.shape[1] - img.shape[0]
            half_of_diff = diff // 2
            other_diff = diff - half_of_diff
            img = np.pad(img,
                         ((extra_pad_top + half_of_diff, extra_pad_bottom + other_diff),
                          (extra_pad_left, extra_pad_right),
                          (0, 0)),
                         padding_type)
        if img.shape[0] >= img.shape[1]:
            diff = img.shape[0] - img.shape[1]
            half_of_diff = diff // 2
            other_diff = diff - half_of_diff
            img = np.pad(img,
                         ((extra_pad_top, extra_pad_bottom),
                          (extra_pad_left + half_of_diff, extra_pad_right + other_diff),
                          (0, 0)),
                         padding_type)
        result.append(img)
    return result


def shift_and_padd(padding_type, *args):
    """
    Apply padding to rectangle image to make it square image
    Also apply some extra padding to make object on image smaller
    Parameters
    ----------
    img: numpy array

    padding_type:
        'constant' for zero padding
        'edge' for edge padding
    Returns
    -------
        image with applied transforms
    """

    shift_h = random.randint(-25, 25)
    shift_v = random.randint(-25, 25)
    base_pad = int(np.clip(abs(random.gauss(0, 100)), 0, 200))

    if shift_v >= 0:
        pad_v = (base_pad, 0)
    else:
        pad_v = (0, base_pad)

    if shift_h >= 0:
        pad_h = (base_pad, 0)
    else:
        pad_h = (0, base_pad)

    result = []
    for img in args:
        img = np.clip(shift(img, (shift_v, shift_h, 0), order=0), 0, 1)

        img = np.pad(img, (pad_v, pad_h, (0, 0)), padding_type)
        result.append(img)
    return result


def random_gamma_transform(img):
    """
    Apply gamma-transform with random gamma value
     from some uniform distribution
    Parameters
    ----------
    img: numpy array

    Returns
    -------
     image with applied gamma-transform
    """
    c = random.uniform(1, 1.3)
    b = np.clip(np.random.normal(-0.13, 0.06, 1), -0.2, 0.06)
    # g1 = random.uniform(0.33, 3)
    g2 = random.uniform(0.8, 1.8)
    img = np.clip(img_as_float(img) + b, 0, 1)
    img = np.clip(img * c, 0, 1)
    return img_as_ubyte(np.clip(img ** g2, 0, 1))


def random_horizontal_flip(*args):
    """
    Apply horizontal flip to all passed images with 50% chance
    Parameters
    ----------
    args: list of numpy images

    Returns
    -------
        list of images
    """
    rand_numer = random.randint(0, 1)
    result = []
    for img in args:
        if rand_numer == 0:
            result.append(np.flip(img, 1))
        else:
            result.append(img)
    return result


def rescale(img, shape, order=1):
    """
    Rescale image to fit for input of rensent18
    Parameters
    ----------
    shape : shape of target image
    img: numpy array
    order: int
        type of interpolation that applied during rescale
        look skimage.transform.rescale for list of different orders

    Returns
    -------
    rescaled image
    """
    return transform.resize(img, shape, order=order,
                            mode="constant")


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
    min_size = 500

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2


def some_local_blur(img, sigma=None):
    """
    Apply some type of blurring to iamge
    Parameters
    ----------
    img: numpy array

    Returns
    -------
    blurred image
    """
    # return restoration.denoise_tv_chambolle(img, weight=0.1, multichannel=True)
    if sigma is None:
        sigma = random.uniform(0.1, 0.3)
        sigma_spatial = random.uniform(0.05, 0.1)
    else:
        sigma = sigma
        sigma_spatial = 1
    return filters.gaussian(img, sigma, multichannel=False)


class BackgroundAddTransformer(object):
    """
    Replace black (0,0,0) pixels of image with some backgrounds
     to create create distribution that looks closer to natural
    """
    def __init__(self, root_dir):
        sub_dirs = os.listdir(root_dir)
        path_func = np.vectorize(os.path.join)
        background_images = path_func(root_dir, sub_dirs)
        self.paths_to_files = []
        for cur_dir in background_images:
            files_from_dir = path_func(cur_dir, os.listdir(cur_dir)).tolist()
            self.paths_to_files = self.paths_to_files + files_from_dir

    def __call__(self, sample):
        if np.min(sample['mask'] == np.zeros(sample['mask'].shape)):
            return sample
        width = sample['color'].shape[0]
        height = sample['color'].shape[1]
        min_foreground_length = min(width, height)
        while True:
            file_idx = random.randint(0, len(self.paths_to_files) - 1)
            background = io.imread(self.paths_to_files[file_idx])
            if background.shape[0] > width and\
               background.shape[1] > height and \
               len(background.shape) == 3 and\
               background.shape[2] == 3:
                break
        min_length = min(background.shape[0], background.shape[1])
        scale_factor = random.uniform(1, 4 * min_foreground_length / min_length)
        background = transform.rescale(background,
                                              scale_factor,
                                              mode="constant")

        start_x = random.randint(0, background.shape[0] - width)
        start_y = random.randint(0, background.shape[1] - height)
        cropped = background[start_x:start_x + width, start_y:start_y + height]
        inv_mask = np.abs(sample['mask'].astype('int') - 1).astype('uint8')
        sample['color'] = sample['color'] * sample['mask']
        temp = inv_mask * cropped + sample['color']

        return {'color': temp,
                'mask': sample['mask'],
                'label': sample['label']}


class NegativeSkinTransformer(object):
    """
        Create negative examples of skin, to prevent segmentation from
        finding face and other human parts except hands
    """
    def __init__(self, path):
        path_func = np.vectorize(os.path.join)
        temp = os.listdir(path)
        self.files = path_func(path, temp)

    def __call__(self, sample):
        rand_numer = random.randint(0,3)
        if rand_numer == 0:
            width = sample['color'].shape[0]
            height = sample['color'].shape[1]
            while True:
                file_number = random.randint(0, len(self.files) - 1)
                color_image = io.imread(self.files[file_number])
                if color_image.shape[0] > width and \
                   color_image.shape[1] > height and \
                   len(color_image.shape) == 3 and \
                   color_image.shape[2] == 3:
                    break
            start_x = random.randint(0, color_image.shape[0] - width)
            start_y = random.randint(0, color_image.shape[1] - height)
            cropped = img_as_float(color_image[start_x:start_x + width, start_y:start_y + height])
            sample = {'color': cropped,
                      'mask': np.zeros((cropped.shape[0], cropped.shape[1], 1), dtype=np.uint8),
                      'label': sample['label']}
            return sample
        else:
            return sample


class BlurTransformer(object):
    def __init__(self, sigma=None):
        self.sigma = sigma
    """
    Apply blur to color image
    """
    def __call__(self, sample):
        result = {'color': some_local_blur(sample['color'], sigma=self.sigma)}
        if 'mask' in sample.keys():
            result['mask'] = sample['mask']
        if 'label' in sample.keys():
            result['label'] = sample['label']
        return result


class CropTransform(object):
    """
    Crop colored image
    """
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def __call__(self, sample):
        result = {'color': (sample['color'])[self.point1[1]:self.point2[1],
                                             self.point1[0]:self.point2[0]]}
        if 'mask' in sample.keys():
            result['mask'] = sample['mask']
        if 'label' in sample.keys():
            result['label'] = sample['label']
        return result


class GaussianNoiseAdderTransform(object):
    """
    Apply gaussian noise with some sigma to image and multiply image on mask
    """
    def __init__(self, sigma_h, sigma_l, using_mask=True):
        self.using_mask = using_mask
        self.sigma_h = sigma_h
        self.sigma_l = sigma_l

    def __call__(self, sample):
        sigma = random.uniform(self.sigma_l, self.sigma_h)
        sample['color'] = img_as_float(sample['color']) + \
                          np.random.normal(0, sigma,
                                           sample['color'].shape)
        sample['color'] = np.clip(sample['color'], 0, 1)
        if self.using_mask:
            result = {'color': sample['color'] * sample['mask']}
        else:
            result = {'color': sample['color']}
        if 'mask' in sample.keys():
            result['mask'] = sample['mask']
        if 'label' in sample.keys():
            result['label'] = sample['label']
        return result


class MaskCreatorTransform(object):
    """
    Create binary mask from color picture
     to separate meaningful parts from trash
    """
    def __call__(self, sample):
        # struct_elem = morphology.disk(2)
        color_image = sample['color']
        mask = (color_image.sum(axis=2) > 0).astype("uint8")
        # mask = morphology.erosion(mask, struct_elem)
        # mask = morphology.dilation(mask, struct_elem)
        # mask = morphology.erosion(mask, struct_elem)
        # mask = remove_little_blobs(mask).astype("uint8") // 255
        # color_image = color_image * mask[:, :, np.newaxis]
        # (color_image, mask) = fill_holes_with_median_color(color_image,
        #                                                    mask.astype('bool'))

        mask = mask[:, :, np.newaxis].astype("uint8")
        # plt.imshow(mask)
        # plt.show()
        color_image = mask * color_image

        sample = {'color': color_image,
                  'mask': mask,
                  'label': sample['label']}
        return sample


class PaddingTransformer(object):
    """
    Apply padding to image and mask
    """
    def __init__(self, padding_type, extra=None):
        self.padding_type = padding_type
        self.extra = extra

    def __call__(self, sample):
        if 'mask' in sample.keys():
            result = padding(self.padding_type, self.extra, sample['color'], sample['mask'])
            return {'color': result[0].copy(),
                    'mask': img_as_ubyte(result[1].copy()),
                    'label': sample['label']}
        else:
            result = padding(self.padding_type, self.extra, sample['color'])
            return {'color': result[0], 'label': sample['label']}


class ShiftAndPaddransformer(object):
    """
    Apply padding to image and mask
    """
    def __init__(self, padding_type):
        self.padding_type = padding_type

    def __call__(self, sample):
        if 'mask' in sample.keys():
            result = shift_and_padd(self.padding_type, sample['color'], sample['mask'])
            return {'color': result[0].copy(),
                    'mask': img_as_ubyte(result[1].copy()),
                    'label': sample['label']}
        else:
            result = shift_and_padd(self.padding_type, sample['color'])
            return {'color': result[0], 'label': sample['label']}


class RandomGammaTransformer(object):
    """
    Apply Gamma-transform to color image
    """
    def __call__(self, sample):
        result = {'color': random_gamma_transform(sample['color'])}
        if 'mask' in sample.keys():
            result['mask'] = sample['mask']
        if 'label' in sample.keys():
            result['label'] = sample['label']
        return result


class RandomHReflectTransformer(object):
    """Apply horizontal reflection to mask and color image"""
    def __call__(self, sample):
        if 'mask' in sample.keys():
            result = random_horizontal_flip(sample['color'], sample['mask'])
            return {'color': result[0].copy(),
                    'mask': img_as_ubyte(result[1].copy()),
                    'label': sample['label']}
        else:
            result = random_horizontal_flip(sample['color'])
            return {'color': result[0], 'label': sample['label']}


class ResizeTransformer(object):
    """Apply resize to image and mask to fit for resnet18 input"""
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        result = {'color': transform.resize(sample['color'],
                                                         self.shape,
                                                         mode="constant")}
        if 'mask' in sample.keys():
            result['mask'] = img_as_ubyte(transform.resize(sample['mask'],
                                                           self.shape,
                                                           order=0,
                                                           mode="constant"))
        if 'label' in sample.keys():
            result['label'] = sample['label']
        return result


class RotateTranform(object):
    """Rotate color image and mask for some angle"""
    def __call__(self, sample):
        angle = random.uniform(-5, 5)
        half_h = sample['color'].shape[0] // 4
        half_w = sample['color'].shape[1] // 4
        sample['color'] = np.pad(sample['color'], ((half_h, half_h), (half_w, half_w), (0, 0)), mode='constant')
        rotated = transform.rotate(sample['color'], angle)
        result = {'color': transform.resize(rotated,
                                            sample['color'].shape,
                                            mode="constant"
                                            )}
        if 'mask' in sample.keys():
            sample['mask'] = np.pad(sample['mask'], ((half_h, half_h), (half_w, half_w), (0, 0)), mode='constant')
            rotated = transform.rotate(sample['mask'], angle)
            result['mask'] = img_as_ubyte(transform.resize(rotated,
                                              sample['mask'].shape,
                                              mode="constant"
                                              ))
            points = np.argwhere(result['mask'] == 1)
            top = -1
            left = 100000
            right = -1
            bottom = 100000
            for p in points:
                if p[0] > top:
                    top = p[0]
                if p[0] < bottom:
                    bottom = p[0]
                if p[1] < left:
                    left = p[1]
                if p[1] > right:
                    right = p[1]
            result['mask'] = result['mask'][bottom:top, left:right]
            result['color'] = result['color'][bottom:top, left:right]
        if 'label' in sample.keys():
            result['label'] = sample['label']
        return result


class ToTensorTransformer(object):
    """Convert color image and mask to torch.tensor and transpose"""
    def __call__(self, sample):
        # plt.imshow(sample['color'])
        # plt.imshow(np.squeeze(sample['mask']), cmap='gray')
        # plt.show()
        if len(sample['color'].shape) == 2:
            sample['color'] = sample['color'][:, :, np.newaxis]
        transposed = sample['color'].transpose((2, 0, 1))
        result = {'color': torch.from_numpy(transposed)}
        if 'mask' in sample.keys():
            transposed = sample['mask'].transpose((2, 0, 1))
            result['mask'] = torch.from_numpy(transposed)
        if 'label' in sample.keys():
            result['label'] = torch.tensor(sample['label'])
        return result


class ToGrayscaleTransform(object):
    """Convert color image to grayscale"""
    def __call__(self, sample):
        result = {"color": color.rgb2gray(sample['color'])[:, :, np.newaxis]}
        if 'mask' in sample.keys():
            result['mask'] = sample['mask']
        if 'label' in sample.keys():
            result['label'] = sample['label']
        return result
