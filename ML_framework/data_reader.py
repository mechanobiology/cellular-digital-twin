import numpy as np
import h5py
import os


class data_reader:
    def __init__(self, flag):
        self.conf = flag
        file_name = self.conf.datadir
        self.data = h5py.File(file_name, 'r')
        self.images = self.data['X']
        self.label = self.data['Y']
        self.z_r = self.data['R']
        self.train_range = int(self.conf.train_range)
        self.test_range = int(self.conf.test_range)
        self.train_idx = 0
        self.test_idx = int(self.conf.train_range)
        self.gen_index()
        print("DATA successfully loaded!!!!!!!!!========================")
        self.image_index= 0
    
    def gen_index(self):
        self.indexes = np.random.permutation(range(self.train_range))
        self.train_idx = 0
    
    def next_batch(self, batch_size):
        next_index = self.train_idx + batch_size
        cur_indexes = list(self.indexes[self.train_idx:next_index])
        self.train_idx = next_index
        if len(cur_indexes) < batch_size:
            self.gen_index()
            return self.next_batch(batch_size)
        cur_indexes.sort()
        return self.images[cur_indexes], self.label[cur_indexes], self.z_r[cur_indexes]
    #    return self.images[cur_indexes], self.label[cur_indexes]

    def next_test_batch(self, batch_size):
        if self.test_idx>=(self.train_range+self.test_range):
            self.test_idx=self.train_range
        prev_idx = self.test_idx
        self.test_idx += batch_size
        return self.images[prev_idx:self.test_idx], self.label[prev_idx: self.test_idx], self.z_r[prev_idx: self.test_idx]
    #    return self.images[prev_idx:self.test_idx], self.label[prev_idx: self.test_idx]


    def reset(self):
        self.test_idx = self.conf.train_range
        
    def extract(self, imgs):
        return np.expand_dims(imgs[:,:,:,:,0], axis=4) #mod

    def extract_label(self, imgs):
        return imgs[:,:,:,:,int(self.conf.mode)] #mod

    def get_next_h5(self):
        prev_idx =self.image_index
        self.image_index = self.image_index +10
        if self.image_index >= 6077:
            self.image_index=6077
            prev_idx =6067
        return self.images[prev_idx:self.image_index], self.label[prev_idx:self.image_index]