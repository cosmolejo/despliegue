import sys
#sys.path.insert(1, '/gdata/liyh/pylib')

import os
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_transform_sketch, get_transform_race
from util.Dictx import DictX
from collections import OrderedDict
from torch.autograd import Variable
#from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import torch
import time

class FaceGenerator:
    def __init__(self):
        self.opt = DictX({})
        self.opt['isTrain'] = False
        self.opt.label_nc = 0
        self.opt['no_instance'] = True
        self.opt['ngf'] = 48
        self.opt['resize_or_crop'] = 'scale_width'
        self.opt['loadSize'] = 256
        self.opt['fineSize'] = 256
        self.opt['gpu_ids'] = [0]
        self.opt['batchSize'] = 1
        self.opt['which_epoch'] = 'latest'
        self.opt['gfm_layers'] = []
        self.opt['gfm_layers'].append(0) 
        self.opt['gfm_layers'].append(1) 
        self.opt['gfm_layers'].append(2) 
        self.opt['gfm_layers'].append(3)
        self.opt['sap_branches'] = []
        self.opt['sap_branches'].append(1)
        self.opt['sap_branches'].append(5)
        self.opt['sap_branches'].append(9)
        self.opt['sap_branches'].append(13)
        self.opt['nThreads'] = 1   # test code only supports nThreads = 1
        self.opt['batchSize'] = 1  # test code only supports batchSize = 1
        self.opt['serial_batches'] = True  # no shuffle
        self.opt['no_flip'] = True  # no flip
        self.opt.model = 'pix2pixHD'
        self.opt.checkpoints_dir = './checkpoints'
        self.opt.name = 'latest'
        self.opt.input_nc = 3
        self.opt.netG = 'global'
        self.opt.norm = 'instance'
        self.opt.output_nc = 3
        self.opt.n_downsample_global = 4
        self.opt.n_blocks_global = 9
        self.opt.race = False
        self.opt.continue_train = False
        self.opt.race_continue_training = False
        self.opt.verbose = True
        self.opt.offset_x = 0
        self.opt.offset_y = 0
        self.opt.degree = 0
        self.opt.deform = False

        self.model = create_model(self.opt)
        
    def generate_face(self,sketch):
        params = get_params(self.opt, sketch.size)
        transform_A = get_transform_sketch(self.opt, params)
        A_tensor = transform_A(sketch.convert('RGB')) # A_tensor: [-1, 1]
        A_tensor = A_tensor.unsqueeze(0)
        model_output = self.model.inference(A_tensor)
        return util.tensor2im(model_output['fake_image'][0])
    
    def get_model(self):
        return self.model