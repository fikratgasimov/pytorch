from __future__ import print_function, division # future statement ensure the exact validity of operations for updates versions
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21 # Exact number of classes

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 ): # Initilization of VOCSegmentation of Dataset
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__() # start
        self._base_dir = base_dir # define base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages') # and use it for self._image_dir
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str): # if instance is seen as instance and str
            self.split = [split] # and then define split as self.split
        else:
            split.sort() # if it is not the case, order split
            self.split = split

        self.args = args # we have also args we defined for as self.args

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation') # see new split directory  as self_base_dir and plus 'ImageSets', 'Segmentation'

        self.im_ids = [] # create self array for im_ids
        self.images = [] # create self array for images
        self.categories = [] # create self array for categories

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f: # take advantage of csv file to write _splits_dir and splt which is in loop of self.split
                lines = f.read().splitlines() # then read and split each rows

            for ii, line in enumerate(lines):  # accordingly, ii and line looping through enumuration of lines [0,1,2,3,4]
                _image = os.path.join(self._image_dir, line + ".jpg") # then find _image by concetanating image_dir and line + 'jpg' 
                _cat = os.path.join(self._cat_dir, line + ".png") # find _cat by concatenating cat_dir and line + 'png'
                assert os.path.isfile(_image) # assert function is just boolean expression that checks whether it is true or not
                assert os.path.isfile(_cat)  # if condition is true, programs moves to next line, otherwise throws an error
                self.im_ids.append(line) # append line to the im_ids
                self.images.append(_image) # append _image to the self.images
                self.categories.append(_cat) # append _categories to the self.categories

        assert (len(self.images) == len(self.categories)) # then check len of self.images is equal to the len of self.categories

        # Display stats: if above assert 'true', throw an error
        print('Number of images in {}: {:d}'.format(split, len(self.images))) # put the split inside the barcket and put self.images into barcket

    def __len__(self):
        return len(self.images) # return len of the self.images array


    def __getitem__(self, index): # once we have accurate data based on arrays, we could initialize '__getitem_' to access the data 
        _img, _target = self._make_img_gt_point_pair(index) # as we see, we initialize the '_img' and '_target'
        sample = {'image': _img, 'label': _target} # then sample string 'image' is seen as _img; and string 'label' is seen as _target

        for split in self.split: # again as we know, split is ' train', and lets loop over the self.split
            if split == "train": # of split is equal to train 
                return self.transform_tr(sample) # return self.transform_tr
            elif split == 'val': # or split is found to be equal to 'value'
                return self.transform_val(sample) # then return transform_val


    def _make_img_gt_point_pair(self, index): # once we have seen getitem provides self._make_img_gt_point_pair(index), then we transform it from self->def
        _img = Image.open(self.images[index]).convert('RGB') # in  this respect, _img could be opened as accessing the index of self_image
        _target = Image.open(self.categories[index]) # as in the case of _img, _target could be acquired via accessing index of the self.categories
  
        return _img, _target # certainly, we returm  _img and _target, respectively

    def transform_tr(self, sample): # eventually, according to the condition of split in self.split, then split == 'train'
        composed_transforms = transforms.Compose([     # define transform_tr
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size), # random scale crop, we have to calcualte base_size and crop_size based on argparse
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)  # return composed_transforms

    def transform_val(self, sample): # as appearing in tranform_tr, transform_val also holds the same principle

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size), # fixscalecrop, we have to calcualte base_size and crop_size based on argparse
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample) # return composed_transform

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


