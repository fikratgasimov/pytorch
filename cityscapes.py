import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class CityscapesSegmentation(data.Dataset): #Initializr CityScape
    NUM_CLASSES = 19  # 19 classes of CityScape

    def __init__(self, args, root=Path.db_root_dir('cityscapes'), split="train"):

        self.root = root
        self.split = split #{Initilization of Parameters}#
        self.args = args
        self.files = {}
        
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split) # define image_base
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split) # define annotation_base
        
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png') # put the split into the bracket of files

    
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]   #  define void_classes
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33] # define valid_classes
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle'] # define class_names

        self.ignore_index = 255 # if index = 255, ignore that pixel
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES))) # in this operation, aim was at mapping via zip and concatenate valid classes and number of classes 
        # consecutively, then introduce as a dictionary!

        if not self.files[split]: # self.files[split] is not the case, pass to the further
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base)) # pass the self.images_base 
        
        print("Found %d %s images" % (len(self.files[split]), split)) # print len of self.files[split]

    def __len__(self): # then again create __init__ to form constructor
        return len(self.files[self.split]) # and then initialize not only self.files but (self.files[self.split])

    def __getitem__(self, index): # immediately apply to the getitem()

        img_path = self.files[self.split][index].rstrip() # define img_path by adding to rstrip()  to the self.files[self.split][index]
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png') # Use also annotation_base

        _img = Image.open(img_path).convert('RGB') # apply to PIL to open image based on img_path indexing operating
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8) # define _temporary parameter, open image and convert it to array
        _tmp = self.encode_segmap(_tmp) # then encode temporary parameter
        _target = Image.fromarray(_tmp) # define _our target parameter to read encoded _tmp array

        sample = {'image': _img, 'label': _target} # once we have _img and _target, we certainly define sample 

        if self.split == 'train': # then again the same consideration: if self.split == 'train' -> TRAIN SET
            return self.transform_tr(sample) # return self.transform_tr(sample) 

        elif self.split == 'val': # if self.split == 'value' -> VALIDATION SET
            return self.transform_val(sample) # return self.transform(sample) 

        elif self.split == 'test': # if self.split = 'test' - > TEST SET
            return self.transform_ts(sample) # return self.transform_ts(sample)

    def encode_segmap(self, mask):  # once we initialized encode_segmap for _tmp parameter, we have to declare encode_segmap by function to analyze mask
        # Put all void classes to zero
        for _voidc in self.void_classes: # then, try to loop void classes
            mask[mask == _voidc] = self.ignore_index # define mask as being equaled to _validc, so all these == self.ignore_index
        for _validc in self.valid_classes: # then loop through  valid_classes 
            mask[mask == _validc] = self.class_map[_validc] # and define mask being equal to _validc, then == class_map[_validc]
        return mask # return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample): # define transform_train set
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255), #define base_size and crop_size below
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample): # define transform_validation set

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample): # define transform_test set

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
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

