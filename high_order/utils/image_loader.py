"""
This is the image loading class used to load target images
"""
import os
import torch
import numpy as np
from imageio import imread
from torchvision.transforms.functional import resize

import utils.utils as utils

class ImageLoader:
    """Basic iterator to load amplitude from sRGB images

    Class initialization parameters
    -------------------------------
    data_path: folder containing images named by indices
        (ie. img_index.png for index in [0, N])
    channel: color channel to load (0, 1, 2 for R, G, B, None for all 3),
        default None
    batch_size: number of images to pass each iteration, default 1
    image_res: Desired dimensions of output images
    idx_subset: for the iterator, skip all but these images.
        Forces batch_size=1. Defaults to None to use full set.
    device: torch.device

    Iterator Output
    -----
    >>> image_loader = ImageLoader(...)
    >>> for ims, input_resolutions, filenames in image_loader:
    >>>     ...

    ims: images in the batch after transformation and conversion to linear
        amplitude, with dimensions [batch, channel, *image_res]
    idxs: list of batch indices

    Alternatively, can be used to manually load a single image:

    >>> ims, input_resolutions, filenames = image_loader.load_image(idx)

    idxs: the index for the image to load
    """
    def __init__(self, data_path, channel=None, batch_size=1,
        image_res=(1080, 1920), idx_subset=None, device=torch.device('cuda:2')):

        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')
        self.data_path = data_path
        self.channel = channel
        self.batch_size = batch_size
        self.image_res = image_res
        self.subset = idx_subset
        self.dev = device

        self.im_files, self.im_idxs = identify_images(data_path) 

        # Sort im_files based on im_idxs
        zipped = zip(self.im_idxs, self.im_files)
        zipped = sorted(zipped)
        self.im_files = [im_file for _, im_file in zipped]

        # if subsetting indices, force batch size 1
        if self.subset is not None:
            self.batch_size = 1

    def __iter__(self):
        """
        Output
        ------
        :return resetted iterator
        """
        self.ind = 0
        return self

    def __next__(self):
        """Loads amplitude of next batch of images
        Output
        ------
        :return batch: (amplitude of specified images, 
            idxs of specified images)
        """
        if self.subset is not None:
            while self.ind not in self.subset and self.ind<len(self.im_files):
                self.ind += 1

        if self.ind < len(self.im_files):
            batch_idxs = np.arange(self.ind, self.ind + self.batch_size)
            self.ind += self.batch_size
            batch = self.load_batch(batch_idxs)
            return batch
        else:
            raise StopIteration

    def __len__(self):
        """
        Output
        ------
        :return length of iterator
        """
        if self.subset is None:
            return len(self.im_files)
        else:
            return len(self.subset)

    def load_batch(self, idxs):
        """Loads amplitude of images with specified idxs
        Input
        -----
        :param idxs: idx in image filename of desired image
        Output
        ------
        :return ims: amplitude of specified images
        :return idxs: idxs of specified images
        """
        ims = [self.load_image(idx) for idx in idxs]
        ims = torch.stack(ims, 0)
        return (ims, idxs)

    def load_image(self, idx):
        """Loads amplitude of image with specified idx
        Input
        -----
        :param idx: idx in image filename of desired image
        Output
        ------
        :return im: amplitude of specified image
        """
        im = imread(self.im_files[idx])
        im = utils.im2float(im, dtype=np.float32)  # convert to double, max 1
        im = torch.from_numpy(im).to(self.dev)

        if len(im.shape) < 3:
            # augment channels for gray images
            im = im.unsqueeze(2).repeat(1, 1, 3)

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        # linearize intensity from sRGB input and convert to amplitude
        low_val = im <= 0.04045
        im[low_val] = 25 / 323 * im[low_val]
        im[torch.logical_not(low_val)] = ((200*im[torch.logical_not(low_val)]
             +11)/211)**(12/5)
        im = torch.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = im.permute(2, 0, 1)

        # normalize resolution
        im = resize_keep_aspect(im, self.image_res)

        return im

def identify_images(dir):
    """Identifies all files in the input directory dir that are images
    Input
    -----
    :param dir: directory with image files
    Output
    ------
    :return images: paths to image files
    :return idxs: idxs in image filenames
    """
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif')
    files = os.listdir(dir)
    exts = (os.path.splitext(f)[1] for f in files)
    images = [os.path.join(dir, f)
        for e, f in zip(exts, files)
        if e[1:] in image_types]
    # idxs = [int((os.path.basename(im).split('_')[1]).split('.')[0]) for im in images]
    idxs = [str(im[:-4]) for im in images]

    return images, idxs


def resize_keep_aspect(image, target_res):
    """Resizes image to the target_res while keeping aspect ratio by cropping
    Input
    -----
    :param image: image with dims [channel, height, width]
    :param target_res: desired output spatial dimensions
    Output
    ------
    :return resized_image: image with dims [channel, *target_res]
    """
    # finds the resolution needed for either dimension to have the target
    # aspect ratio, when the other is kept constant. If the image doesn't have
    # the target ratio, then one of these two will be larger, and the other
    # smaller, than the input image dimensions.
    im_res = image.shape[-2:]
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
        int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # Crops input image to produce the desired target resolution after 
    # resizing. This only affects the dimension which is smaller in 
    # resized_res than in the input image dimensions.
    image = utils.crop_image(image, resized_res)

    # Resizes tensor with a scale factor that is the same for both dimensions
    resized_image = resize(image, target_res)
    return resized_image

