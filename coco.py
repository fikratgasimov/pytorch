import numpy as np # numpy array construction
import torch  # provides multidimensional tensor and mathematical operations on them
from torch.utils.data import Dataset # All datasets are subclasses of torch.utils.data.
from mypath import Path
from tqdm import trange # instantly make our loop show progress meter
import os
from pycocotools.coco import COCO # import COCO dataset
from pycocotools import mask # coco mask 
from torchvision import transforms # transform can easily be chained with Compose
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(Dataset):  # Large COCO dataset INITILIAZED AT THE BOTTOM
    NUM_CLASSES = 21 # initialization of Number of Classes
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]  # cat lists

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('coco'),
                 split='train',
                 year='2017'): # initialization of COCOSegmentation Dataset
        super().__init__() 
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year)) # here we 'split' and 'year' are appended into .json instance file for ann_file
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))  # here we again 'split' and 'year' are appended into ids .pth for ids_file
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))  # initialize self.img_dir by appending split and year into images
        self.split = split   # 'train'
        self.coco = COCO(ann_file) # then COCO dataset grap ann_file
        self.coco_mask = mask      # accordingly, COCO GRAP mask due to implementation principle
        if os.path.exists(ids_file): # then os.path.exist function checks wether ids_file existed or not
            self.ids = torch.load(ids_file) # if yes, then define self.ids by loading ids_file
        else:
            ids = list(self.coco.imgs.keys()) # if not, define ids as list parameter of the coco image keys parameters
            self.ids = self._preprocess(ids, ids_file) # sequentially, self.id is determined in a way: 1.preprocess of ids and 2. ids_file defined(which is defined above)
            self.args = args # eventually, we took advantage of args

    def __getitem__(self, index): # getitem provides loop via index operation of each file
        _img, _target = self._make_img_gt_point_pair(index) # _img, _target are defined for index operation
        sample = {'image': _img, 'label': _target}  # and in turn, _img and _target are regarded as a strings later on: 'image', 'label'

        if self.split == "train": # by calling function of self.splits train (statement '==') 
            return self.transform_tr(sample) #recover self and return as already predefined sample : transform_tr(sample)
        elif self.split == 'val': # in another case as in the case of elif statement, return transform_val(sample)
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index): # take advantage of getitem function, we reuse _img,_target
        coco = self.coco #define coco by recalling 'self.coco = COCO(ann_file)'
        img_id = self.ids[index] # img_id is defined by recalling self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0] # img_metadata captures data we loaded with through img_id
        path = img_metadata['file_name'] # then define path via meta_data
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB') # once we have path, self.img_dir, we use PIL open images
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id)) # use img_id to identify cocotarget which will in turn be used in cocotarget
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width'])) #_target generated new array of image by  cocotarget and index operation of img_metadata

        return _img, _target # return _img, _target

    # THIS PREPROCESS IS ALMOST BACK-UP VERSIO N OF THE FUNCTION OF _make_img_gt_point_pair
    def _preprocess(self, ids, ids_file): # preprocessing mask which will take a while
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids)) # tbar new value defined via trange based on len(ids)
        new_ids = [] # new array of new_ids defined
        for i in tbar: # loop through the (tbar)
            img_id = ids[i] # img_id is defined as ids which is turn,defines above list parameter of the coco image keys parameters
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))  # use img_id to identify cocotarget which will in turn be used in cocotarget by SELF	
            img_metadata = self.coco.loadImgs(img_id)[0]  # img_metadata captures data we loaded with through img_id
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000: # if statement represents amount of mask around objects
                new_ids.append(img_id) # then put the img_id into new_ids
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask (self, target, h, w): # declare _gen_seg_mask as a function
        mask = np.zeros((h, w), dtype=np.uint8) # mask is defined as np.zeros array
        coco_mask = self.coco_mask # finally coco_mask found self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle) # decode 'encode' version of rle
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(), # given PIL image randomly with a given probability
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.ids)



if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser() # first step of using argparse is to create ArgumentParser object
    args = parser.parse_args() # program defines what arguments it requires
    args.base_size = 513 # args.base_size defined above, but generated value here
    args.crop_size = 513 # args.crop_size defined above but generated value here

    coco_val = COCOSegmentation(args, split='val', year='2017') # COCOSegmentation parameters are defined

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
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
