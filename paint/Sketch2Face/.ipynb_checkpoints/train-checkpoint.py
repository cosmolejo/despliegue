import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
print(opt.gpu_ids)
if opt.continue_train:
    
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('start_epoch',start_epoch, 'epoch_iter', epoch_iter)
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    if(opt.niter_fix_global > 0):
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G_sap, model.optimizer_D], opt_level='O1')   
    else:
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')           
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    if(opt.niter_fix_global > 0):
        optimizer_G, optimizer_D = model.module.optimizer_G_sap, model.module.optimizer_D
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    if(opt.race):
        optimizer_R = model.module.optimizer_R

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq 
skiped = 0
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        ############## Forward Pass ######################
            
        if(opt.race):
            result = model(Variable(data['sketch']), Variable(data['photo']), Variable(data['sketch_deform']),        Variable(data['race']))
        else:
            result = model(Variable(data['sketch']), Variable(data['photo']), Variable(data['sketch_deform']))
            
       
        # sum per device losses
        losses = result['losses']
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))
         # calculate final loss scalar
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)
        if(not opt.race):
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            total = loss_D+loss_G
        else:
            if(opt.race_loss_opt1):
                loss_R = loss_dict['cross_entropy_race']
                total = loss_R
            else:
                loss_R = loss_dict['cross_entropy_race'] + loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)
                total = loss_R
        
        
        ############### Backward Pass ####################
        # update generator weights
        if(opt.race):
            
            optimizer_R.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_R, optimizer_R) as scaled_loss: scaled_loss.backward()                
            else:
                loss_R.backward()
            optimizer_R.step()
        else:
            optimizer_G.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
            else:
                loss_G.backward()          
            optimizer_G.step()
            # update discriminator weights
            
            optimizer_D.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
            else:
                loss_D.backward()        
            optimizer_D.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t,total)
            visualizer.plot_current_errors(errors, total_steps,total)
            call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        ### display output images
        
        race_str = data_loader.dataset.getLabelEncoder().inverse_transform(Variable(data['race']))[0]
        img_num = str(data['img_num'][0].item())

        if save_fake:
            if(opt.deform):
                visuals = OrderedDict([('input_label', util.tensor2im(data['sketch'][0])),
                                       ('synthesized_image', util.tensor2im(result['fake_image'][0])),
                                       ('deformed_image', util.tensor2im(result['fake_image_deform'][0])),
                                       ('real_image', util.tensor2im(data['photo'][0]))])
            else:
                visuals = OrderedDict([('input_label', util.tensor2im(data['sketch'][0])),
                                       ('synthesized_image', util.tensor2im(result['fake_image'][0])),                         
                                       ('real_image', util.tensor2im(data['photo'][0]))])
    
            visualizer.display_current_results(visuals, epoch, total_steps, race_str, img_num)
        
        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')    
        if epoch_iter >= dataset_size:
            break
  
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()  
