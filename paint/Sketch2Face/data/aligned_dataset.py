import os.path
import glob
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_transform_sketch, get_transform_race
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot  
        if(opt.race):
            metadata = pd.read_csv(opt.dataroot+'meta_full.csv')
        else:
            metadata = None
        ### input A (sketch)
        images = metadata[metadata['selected']==True]
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A' )
        
        if(not opt.test_csv):
            self.A_paths,_,_ = make_dataset(self.dir_A, metadata)
            self.A_paths = list(filter(lambda x: not '.ipynb_checkpoints' in x, self.A_paths))
        else:
            self.A_paths = self.dir_A +'/'+ images['id'].values
            self.race = images['race'].values
            images[['a','b']] = images['id'].str.split('.', n=2)[0]
            self.image_num = images['a'].values
            
     
        
        
        ### input B (photo)
        if self.opt.isTrain or self.opt.use_encoded_image:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
            print('OPT TEST',opt.test_csv)
            if(not opt.test_csv):
                self.B_paths,self.race, self.image_num = make_dataset(self.dir_B, metadata)
            else:
                print('si entra')
                self.B_paths = self.dir_B + images['id'].values
                

        ### input C (deform sketch)
        
       
        self.C_list_paths = []
        self.dir_C = os.path.join(opt.dataroot, opt.phase + '_C')
        for A_path in self.A_paths:
            #if(".ipynb_checkpoints" in A_path):
            #    continue
            basename = os.path.splitext(os.path.basename(A_path))[0]
            
            C_list_path = glob.glob(os.path.join(self.dir_C, '*', basename + '.png'))
            self.C_list_paths.append(C_list_path)
       
        self.dataset_size = len(self.A_paths) 
        
        ### OHE RACE
        if(opt.race):
            race_OHE = np.array(self.race)
            # integer encode
            self.label_encoder = LabelEncoder()
            self.integer_encoded = self.label_encoder.fit_transform(race_OHE)
            ## binary encode
            #onehot_encoder = OneHotEncoder(sparse=False)
            #integer_encoded = self.integer_encoded.reshape(len(self.integer_encoded), 1)
            #self.race_OHE = onehot_encoder.fit_transform(self.integer_encoded)

        
      
    def __getitem__(self, index):
        ### race
        if(self.race is not None):
            race = self.race[index]
            #race_OHE = self.race_OHE[index]
            race_OHE = torch.tensor(self.integer_encoded[index], dtype=torch.long)
        
        ### input A 
        
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        transform_A = get_transform_sketch(self.opt, params)
        A_tensor = transform_A(A.convert('RGB')) # A_tensor: [-1, 1]
        if(self.race is not None):
            A_tensor = get_transform_race(A_tensor, race)

        ### input B (real images)
        B_tensor = inst_tensor = feat_tensor = C_tensor = 0
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
            
        ### input C 
        #print(self.C_list_paths)
        if self.opt.deform:
            rand_index = np.random.randint(len(self.C_list_paths[index]))
            C_path = self.C_list_paths[index][rand_index] 
            C = Image.open(C_path).convert('RGB')
            transform_C = get_transform_sketch(self.opt, params)      
            C_tensor = transform_C(C)
        ## image number
        
        img_num = torch.tensor(int(self.image_num[index]), dtype=torch.long)
        

        if self.opt.mix_sketch:
            rand_mix = np.random.randint(2)
        input_dict = {}
        input_dict['sketch'] =  C_tensor if self.opt.mix_sketch and rand_mix else A_tensor
        input_dict['photo'] =  B_tensor
        if self.opt.deform:
            input_dict['sketch_deform'] = A_tensor if self.opt.mix_sketch and rand_mix else C_tensor 
        input_dict['path'] =  A_path
        input_dict['img_num'] = img_num
        if(self.race is not None):
            input_dict['race'] = race_OHE


        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset' 
    def getLabelEncoder(self):
        return self.label_encoder