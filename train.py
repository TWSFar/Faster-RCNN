import os
from tqdm import tqdm

from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils.torch_utils import select_device
from dataloaders.datasets import Datasets

import torch
from torch.utils.data import Dataloader
import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self, **kwargs):
        opt._parse(kwargs)
        self.opt = opt
        self.device, self.device_id = select_device(is_head=True)

        # Define Dataloader
        train_dataset = Datasets(opt.dataset, mode='train')
        self.train_loader = Dataloader(train_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)
        self.num_batch = len(self.train_loader)
        print("load data")

        # Define Network
        # initilize the network here.
        faster_rcnn = FasterRCNNVGG16()
        self.trainer = FasterRCNNTrainer(faster_rcnn)
        print("define network")

        # Resuming Checkpoint
        self.start_epoch = 0
        self.best_map = 0
        self.lr = opt.lr
        if opt.load_path:
            self.trainer.load(opt.load_path)
            self.start_epoch = self.trainer.start_epoch
            self.best_map = self.trainer.best_map
            print('load pretrained model from %s' % opt.load_path)

        # Use multiple GPU
        if opt.use_mgpu and len(self.device_id) > 1:
            self.trainer = torch.nn.DataParallel(self.trainer,
                                                 device_ids=self.device_id)
            print("Using multiple gpu")
        else:
            self.model = self.model.to(self.device)

        # Visdom
        self.trainer.vis.text(train_dataset.classes, win='labels')

    def train(self):
        self.trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(self.train_loader)):
            img = img.to(self.device)
            bbox = bbox_.to(self.device)
            label = label_.tp(self.device)
            trainer.train_step(img, bbox, label, scale)
            


def main():
    train_class = Trainer()
    for epoch in range(train_class.start_epoch, train_class.opt.epoch):
        train_class.train()


if __name__ == "__main__":
    main()
