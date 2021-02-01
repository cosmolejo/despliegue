import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from PIL import Image
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
import scipy.misc
import pandas as pd
import tensorflow as tf
import datetime
import cv2

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.create_file_writer(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def put_text(self,img, text):
 
        # You may need to adjust text size and position and size.
        # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = cv2.putText(img, str(text.numpy()), (0, 13), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
        result = result[:,:,::-1]

        return result

    def tf_put_text(self,imgs, texts):
        return tf.py_function(self.put_text, [imgs, texts], Tout=imgs.dtype)
    def display_current_results(self, visuals, epoch, step, race, idx):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                Image.fromarray(image_numpy).save(s, format="jpeg")
                

                # Create a Summary value
                image_numpy = self.tf_put_text(image_numpy, race+' '+idx)
                image_numpy = np.expand_dims(image_numpy, axis=0)
                with self.writer.as_default():
                    tf.summary.image(name=label,step=epoch, data=image_numpy)
                    self.writer.flush()
        races = {}
        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.png' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                        races['img_path'] = img_path
                        races['epoch'] = epoch
                        races['label'] = label
                        races['img'] = i
                        races['race'] = race
                        races['idx'] = idx
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)
                    races['img_path'] = img_path
                    races['epoch'] = epoch
                    races['label'] = label
                    races['race'] = race
                    races['idx'] = idx
           
            try:
                races_df = pd.read_csv(os.path.join(self.web_dir,'races_train.csv'))
                races_df = pd.concat([races_df,pd.DataFrame(races,index=[1])])
            except:
                races_df = pd.DataFrame(races,index=[1])
            races_df.to_csv(os.path.join(self.web_dir,'races_train.csv'),index=False)
            
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
 
                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.png' % (n, label, i)
                            #imagemeta = races_df[races_df['img_path'] == self.img_dir+'/'+img_path]
                            #race = imagemeta['race']
                            #idx = imagemeta['idx']
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
       
                    else:
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        #imagemeta = races_df[races_df['img_path'] == self.img_dir+'/'+img_path]
                        #race = imagemeta['race']
                        #idx = imagemeta['idx']
                        #print(imagemeta['img_path'])
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
 
                 
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step,total):
        if self.tf_log:
            with self.writer.as_default():
                for tag, value in errors.items():
                    tf.summary.scalar(tag, value, step=step)
                    self.writer.flush()
                tf.summary.scalar('TOTAL_ERROR', total.cpu().detach().numpy(), step=step)
                self.writer.flush()

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, total):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        print('ERROR TOTAL: ', total)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, race,idx):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, race +'_'+ image_name)
            img = cv2.cvtColor(np.array(image_numpy), cv2.COLOR_RGB2BGR)
            text = race+' '+idx
            result = cv2.putText(img, text, (0, 13), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)
            result = result[:,:,::-1]
            util.save_image(result, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
        
    def save_feature_as_image(self, feature, dir, basename):
        '''
            The i-th channel of feature is saved as 'dir/basename_i.png'
   
        '''
        feature = feature.cpu().detach().numpy()

        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(dir, basename)
        num_ch = feature.shape[1]
        for i in range(num_ch):
            tmp = feature[0,i,:,:]
            scipy.misc.imsave(dir + '_'+str(i)+'.png', tmp)

