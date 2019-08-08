import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.config import cfg

class RPN(nn.Modules):

    def __init__(self, din):
        super(RPN, self).__init__()
        
        self.din = din
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # Define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # Define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # Define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_ratios) * len(self.anchor_scales) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # Define proposal layer
        self.RPN_proposal = 