import os.path as osp
from pprint import pprint
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from mypath import Path


class Config:
    # data
    min_size = 600
    max_size = 1000
    input_size = (1000, 800)
    num_workers = 8
    test_num_workes = 8
    batch_size = 1

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    # visualization
    env = "faster-rcnn"  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    dataset = "voc"
    data_dir = Path.db_root_dir(dataset)
    pretrained_model = "vgg16"

    # training
    epoch = 14

    use_mgpu = True
    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chairs
    use_drop = False  # use dropout in RoIHead

    # debug
    debug_file = "tmp/debug"

    test_num = 10000

    # model
    load_path = None

    caffe_pretrain = False
    caffe_pretrain_path = ""

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError("UnKnown Option: --{}".format(k))
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
