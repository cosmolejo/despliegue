### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import sys
sys.path.insert(1, '/gdata/liyh/pylib')

import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import time

start = time.time()
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
    opt.dataroot = './datasets/debug'
    opt.name = 'debug'
    print('debug mode sets the name of model to *debug*, and disable continue_train')
    opt.continue_train = False

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
        
            
    if opt.verbose:
        print(model)
    
else:
    from run_engine import run_trt_engine, run_onnx
    
for i, data in enumerate(dataset):
    if opt.how_many > 0 and i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['sketch'] = data['sketch'].half()
        data['sketch_deform'] = data['sketch_deform'].half()
        data['photo']  = data['photo'].half()
    elif opt.data_type == 8:
        data['sketch'] = data['sketch'].uint8()
        data['sketch_deform'] = data['sketch_deform'].uint8()
        data['photo']  = data['photo'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['sketch'], data['sketch_deform'], data['photo']],
                          opt.export_onnx, verbose=True)
        exit(0)
    
    minibatch = 1 
    if opt.engine:
        
        generated = run_trt_engine(opt.engine, minibatch, [data['sketch'], data['photo']])
    elif opt.onnx:
        
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['sketch'], data['photo']])
    else:        
        model_output = model.inference(data['sketch'], data['photo'])
        
    visuals = OrderedDict([
                            ('sketch', util.tensor2label(data['sketch'][0], opt.label_nc)),
                            ('synthesized', util.tensor2im(model_output['fake_image'].data[0])),
                            ])
        
    img_path = data['path']
    print('process image... %s' % img_path)
    race_str = data_loader.dataset.getLabelEncoder().inverse_transform(Variable(data['race']))[0]
    img_num = str(data['img_num'][0].item())
    visualizer.save_images(webpage, visuals, img_path, race_str, img_num)

webpage.save()
print('Time used: ', time.time() - start)