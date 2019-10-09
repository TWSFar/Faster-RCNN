import os
from tqdm import tqdm

from model import FasterRCNNVGG16
from dataloaders.datasets import Datasets
from trainer import FasterRCNNTrainer
from utils.config import opt
from utils import array_tool as at
from utils.torch_utils import select_device

import torch
from torch.utils.data import DataLoader

import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Trainer(object):
    def __init__(self, **kwargs):
        opt._parse(kwargs)
        self.opt = opt
        self.test_num = self.opt.test_num
        self.device, self.device_id = select_device(is_head=True)
        # Define Dataloader
        print("load data")
        self.train_dataset = Datasets(opt, mode='train')
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)
        self.val_dataset = Datasets(opt, mode='val')
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=opt.test_num_workers)
        self.num_batch = len(self.train_loader)

        # Define Network
        # initilize the network here.
        print("define network")
        faster_rcnn = FasterRCNNVGG16()
        self.trainer = FasterRCNNTrainer(faster_rcnn)

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
            self.trainer = self.trainer.to(self.device)

        # Visdom
        self.trainer.vis.text(self.train_dataset.classes, win='labels')

    def train(self):
        self.trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(self.train_loader)):
            scale[0] = at.scalar(scale[0])
            scale[1] = at.scalar(scale[1])
            img = img.to(self.device)
            bbox = bbox_.to(self.device)
            label = label_.to(self.device)
            self.trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    os.makedirs(opt.debug_file)

                self.trainer.vis.plot_many(self.trainer.get_meter_data())

                # plot groud truth bboxes

                # plot predicti bboxes

                # rpn confusion matrix(meter)
                self.trainer.vis.text(str(self.trainer.rpn_cm.value().tolist()), win='rpn_cm')
                self.trainer.vis.img('roi_cm', at.totensor(self.trainer.roi_cm.conf, False).float())

    def eval(self):
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        


def main():
    train_class = Trainer()
    for epoch in range(train_class.start_epoch, train_class.opt.epoch):
        train_class.train()


if __name__ == "__main__":
    main()
