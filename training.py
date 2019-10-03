from utils.config import opt
from dataloaders.datasets import Datasets

from torch.utils.data import Dataloader


class Trainer(object):
    def __init__(self, **kwargs):
        opt._parse(kwargs)
        self.opt = opt

        # Define Visdom

        # Define Dataloader
        train_dataset = Datasets(opt.dataset, mode='train')
        self.train_loader = Dataloader(train_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)
        self.num_batch = len(self.train_loader)

        # Define Network
        # initilize the network here.
 
        # Define Optimizer

        # Define Criterion
        # Whether to use class balanced weights

        # Define Evaluater

        # Define lr scherduler

        # Resuming Checkpoint

        # Using cuda

        # Clear start epoch if fine-tuning


def main():
    trainer = Trainer()
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     trainer.training(epoch)
    #     if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval -  1):
    #         trainer.validation(epoch)

    # trainer.writer.close()


if __name__ == "__main__":
    main()