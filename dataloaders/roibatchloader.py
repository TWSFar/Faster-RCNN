import os
import pickle
import os.path as osp
import numpy as np
from PIL import Image
from datasets.pascal_voc import pascal_voc

from torch.utils.data import Dataset
from torchvision import transforms
try:
    from dataloaders import custom_transforms as ctf
    from mypath import Path
except:
    print('test...')
    import sys
    sys.path.extend(['G:\\CV\\Reading\\Faster-RCNN',])
    # from dataloaders import custom_transforms as ctf
    from mypath import Path  

class roibatchloader(Dataset):
    
    def __init__(self,
                 args,
                 dataset='pascal',
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 mode='train'):
        super().__init__()
        self.args = args
        self.mode = mode
        if dataset == 'pascal':
            imdb = pascal_voc(base_dir=base_dir,
                                   split=split,
                                   mode=mode)
        else:
            raise NotImplementedError
        self._num_classes = imdb.num_classes
        

        self.roidb = imdb.roidb
        self.ratio_list, self.ration_index = self.rank_roidb_ratio()
    def rank_roidb_ratio(self):
        # rank roidb based on the ratio between width and height.
        ratio_large = 2 # largest ratio to preserve.
        ratio_small = 0.5 # smallest ratio to preserve.    
        
        ratio_list = []
        for i in range(len(self.roidb)):
            width = self.roidb[i]['width']
            height = self.roidb[i]['height']
            ratio = width / float(height)

            if ratio > ratio_large:
                self.roidb[i]['need_crop'] = 1
                ratio = ratio_large
            elif ratio < ratio_small:
                self.roidb[i]['need_crop'] = 1
                ratio = ratio_small        
            else:
                self.roidb[i]['need_crop'] = 0

            ratio_list.append(ratio)

        ratio_list = np.array(ratio_list)
        ratio_index = np.argsort(ratio_list)
        return ratio_list[ratio_index], ratio_index

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            return self._transforms(sample, split)
    
    def _transforms(self, sample, split):
        if self.mode == 'train':
            composed_transforms = transforms.Compose([
                ctf.RandomHorizontalFlip(),
                ctf.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                ctf.RandomGaussianBlur(),
                ctf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ctf.ToTensor()])
            return composed_transforms(sample)

        elif self.mode == 'val':
            composed_transforms = transforms.Compose([
                ctf.FixScaleCrop(crop_size=self.args.crop_size), # default = 513
                ctf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ctf.ToTensor()])
            return composed_transforms(sample)
    
    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target


if __name__ == "__main__":
    temp = roibatchloader(args=None)
    pass