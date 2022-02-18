# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:49:31 2021

@author: user
"""
from PIL import Image
import numpy as np

class Data_Gen():
    
    def __init__(self):
        
        self.map_img = Image.open('C:/Users/user/.spyder-py3/CNN_LSTM/CNU_first_floor_full_edit.pgm')
        
        self.train_data_num = 220
        self.train_data_left = 140
        
        self.test_data_num = 22
        self.test_data_left = 12
        
        self.train_file_name1 = 'batch_train_data1'
        self.train_label_file_name1 = 'batch_train_label_data1'
        self.test_file_name1 = 'batch_test_data1'
        self.test_label_file_name1 = 'batch_test_label_data1'
        
        self.data_height1 = 310
        self.data_width1 = 1000
        self.channel_n = 4

        self.train_file_name2 = 'batch_train_data2'
        self.train_label_file_name2 = 'batch_train_label_data2'
        self.test_file_name2 = 'batch_test_data2'
        self.test_label_file_name2 = 'batch_test_label_data2'
        
        self.data_height2 = 200
        self.data_width2 = 4
        
    def pose_path_gen(self, file_name, path):
        
        pose_path = np.load(path.format(file_name))
        pose_path[:, 0] = np.round(pose_path[:,0] * 20)  # 20 : map resolution. this converts position to map index. 
        pose_path[:, 1] = np.round(pose_path[:,1] * 20)
        pose_path[:, 2] = np.round(pose_path[:,2] * 20)
        pose_path = np.delete(pose_path, 2, axis = 1)    # delete thata column
        pose_path[:, [1, 0]] = pose_path[:, [0, 1]]      # swith x, y
        
        return pose_path
    
    def map_array_gen(self, ogm):
        
        map_imgArray = np.array(ogm)
        map_imgArray = 255 - map_imgArray
        
        plus_map = np.zeros((1,1000), dtype = np.uint8) + 50
        
        map_imgArray = np.concatenate([map_imgArray, plus_map], axis = 0)
        
        return map_imgArray
    
    def empty_batch1_gen(self, data_num, height, width, channel):
        
        empty_batch1 = np.zeros((data_num, height, width, channel))
        empty_label_batch1 = np.zeros((data_num, height, width))
        
        return empty_batch1, empty_label_batch1
    
    def empty_batch2_gen(self, data_num, height, width):
        
        empty_batch2 = np.zeros((data_num, height, width))
        empty_label_batch2 = np.zeros((data_num, height, width))
        
        return empty_batch2, empty_label_batch2
        
    def data_batch_gen(self):
        
        map_imgArray =  self.map_array_gen(self.map_img)
        
        map_imgArray1 = map_imgArray.copy()
        map_imgArray1 = np.reshape(map_imgArray1, (1,310, 1000))
        
        empty_train_batch1, empty_train_label_batch1 = self.empty_batch1_gen(self.train_data_num, self.data_height1, self.data_width1, self.channel_n)
        empty_train_batch1[:, :, :, 0] = map_imgArray1.copy()
        
        empty_test_batch1, empty_test_label_batch1 = self.empty_batch1_gen(self.test_data_num, self.data_height1, self.data_width1, self.channel_n)
        empty_test_batch1[:, :, :, 0] = map_imgArray1.copy()
        
        empty_train_batch2, empty_train_label_batch2 = self.empty_batch2_gen(self.train_data_num, self.data_height2, self.data_width2)
        empty_test_batch2, empty_test_label_batch2 = self.empty_batch2_gen(self.test_data_num, self.data_height2, self.data_width2)
        
        particle_filter_label = np.zeros((self.test_data_num, 17, 4))


        ### train batch generation ###
        for i in range(1, self.train_data_num + 1):
         
            file_name = 'pose%d'%i
            path ='C:/Users/user/.spyder-py3/CNN_LSTM/data/dataset/train/{}.npy'
            pose_path = self.pose_path_gen(file_name, path)
            
            
            for j in range(len(pose_path)): # Creating batch data
                
                empty_train_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])][1] = 255
                empty_train_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])][2] = pose_path[j][2]
                empty_train_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])][3] = pose_path[j][3]
                
                empty_train_label_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])] = 255
                
                if j !=  (len(pose_path) -1):
                    
                    empty_train_batch2[i - 1][j][0] = int(310 - pose_path[j][0])
                    empty_train_batch2[i - 1][j][1] = int(pose_path[j][1])
                    empty_train_batch2[i - 1][j][2] = pose_path[j][2]
                    empty_train_batch2[i - 1][j][3] = pose_path[j][3]
                    
                    empty_train_label_batch2[i - 1][j][0] = int(310 - pose_path[j + 1][0])
                    empty_train_label_batch2[i - 1][j][1] = int(pose_path[j + 1][1])
                    empty_train_label_batch2[i - 1][j][2] = pose_path[j + 1][2]
                    empty_train_label_batch2[i - 1][j][3] = pose_path[j + 1][3]
            
            map_imgArray_view = map_imgArray.copy()
            
            for k in range(len(pose_path)):  # creating path on map
        
                map_imgArray_view[int(310 - pose_path[k][0])][int(pose_path[k][1])] = 255
            
            
            pil_image_train = Image.fromarray(map_imgArray_view)
            pil_image_train.save('./CNN_LSTM/data/dataset/picture/train/{}.png'.format(file_name),'PNG')


        ### test batch generation ###            
        for i in range(1, self.test_data_num + 1):
            
            file_name = 'pose%d'%i
            path = 'C:/Users/user/.spyder-py3/CNN_LSTM/data/dataset/test/{}.npy'
            pose_path = self.pose_path_gen(file_name, path)
            
            for j in range(len(pose_path)): # Creating batch data
                
                empty_test_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])][1] = 255
                empty_test_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])][2] = pose_path[j][2]
                empty_test_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])][3] = pose_path[j][3]
                
                empty_test_label_batch1[i - 1][int(310 - pose_path[j][0])][int(pose_path[j][1])] = 255
                
                if j !=  (len(pose_path) -1):
                    
                    empty_test_batch2[i - 1][j][0] = int(310 - pose_path[j][0])
                    empty_test_batch2[i - 1][j][1] = int(pose_path[j][1])
                    empty_test_batch2[i - 1][j][2] = pose_path[j][2]
                    empty_test_batch2[i - 1][j][3] = pose_path[j][3]
                    
                    empty_test_label_batch2[i - 1][j][0] = int(310 - pose_path[j + 1][0])
                    empty_test_label_batch2[i - 1][j][1] = int(pose_path[j + 1][1])
                    empty_test_label_batch2[i - 1][j][2] = pose_path[j + 1][2]
                    empty_test_label_batch2[i - 1][j][3] = pose_path[j + 1][3]
                    
                
                
            if i < self.test_data_left + 1:
                turning_index = np.where(pose_path[:, 1] > 656)
                bb = np.asarray(turning_index)
                
                k = np.split(pose_path, [bb[0,0], len(pose_path)], axis=0)
            
            else:
                turning_index = np.where(pose_path[:, 1] > 634)
                bb = np.asarray(turning_index)
                
                k = np.split(pose_path, [bb[0,0], len(pose_path)], axis=0)
                
            for kk in range(17):
                cccc = k[1]
                particle_filter_label[i - 1, kk, :] = cccc[kk, :]

            
            
            map_imgArray_view = map_imgArray.copy()    
            
            for k in range(len(pose_path)):  # creating path on map
        
                map_imgArray_view[int(310 - pose_path[k][0])][int(pose_path[k][1])] = 255
                
                        
            pil_image_test = Image.fromarray(map_imgArray_view)
            pil_image_test.save('./CNN_LSTM/data/dataset/picture/test/{}.png'.format(file_name),'PNG')
            
            
        empty_train_batch1[0:self.train_data_left, :, 656:1001, 1:4] = 0                # upper direction
        empty_train_batch1[self.train_data_left:self.train_data_num, :, 634:1001, 1:4] = 0    # lower direction
        empty_train_label_batch1[0:self.train_data_left, :, 0:656] = 0                        # upper direction
        empty_train_label_batch1[self.train_data_left:self.train_data_num, :, 0:634] = 0            # lower direction
        
        empty_test_batch1[0:self.test_data_left, :, 656:1001, 1:4] = 0                # upper direction
        empty_test_batch1[self.test_data_left:self.test_data_num, :, 634:1001, 1:4] = 0    # lower direction
        empty_test_label_batch1[0:self.test_data_left, :, 0:656] = 0                        # upper direction
        empty_test_label_batch1[self.test_data_left:self.test_data_num, :, 0:634] = 0            # lower direction
        
        np.save('./CNN_LSTM/data/train/{}'.format(self.train_file_name1), empty_train_batch1)
        np.save('./CNN_LSTM/data/train/{}'.format(self.train_label_file_name1), empty_train_label_batch1)
        np.save('./CNN_LSTM/data/test/{}'.format(self.test_file_name1), empty_test_batch1)
        np.save('./CNN_LSTM/data/test/{}'.format(self.test_label_file_name1), empty_test_label_batch1)
        
        np.save('./CNN_LSTM/data/train/{}'.format(self.train_file_name2), empty_train_batch2)
        np.save('./CNN_LSTM/data/train/{}'.format(self.train_label_file_name2), empty_train_label_batch2)
        np.save('./CNN_LSTM/data/test/{}'.format(self.test_file_name2), empty_test_batch2)
        np.save('./CNN_LSTM/data/test/{}'.format(self.test_label_file_name2), empty_test_label_batch2)
        
        np.save('./CNN_LSTM/data/test/pf_label', particle_filter_label)
    
    
    def data_picture_gen(self):
        
        train_data_batch = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(self.train_file_name1))
        train_label_batch = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(self.train_label_file_name1))
        test_data_batch = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/test/{}.npy'.format(self.test_file_name1))
        test_label_batch = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/test/{}.npy'.format(self.test_label_file_name1))
        
        map_img = train_data_batch[:, :, :, 0].copy()
        
        train_input_batch = train_data_batch[:, :, :, 1].copy()
        test_input_batch = test_data_batch[:, :, :, 1].copy()
        
        for i in range(1, self.train_data_num + 1):
            
            file_name1 = 'input_pose%d'%i
            file_name2 = 'label_pose%d'%i
            file_name3 = 'input_pic%d'%i
            file_name4 = 'label_pic%d'%i
            
            input_view = train_input_batch[i - 1].copy()
            input_view = input_view.astype('uint8')
            
            label_view = train_label_batch[i - 1].copy()
            label_view = label_view.astype('uint8')
            
            pic1_view = map_img[i - 1].copy() + train_input_batch[i - 1].copy()
            pic1_view = np.where(pic1_view > 255, 255, pic1_view)
            pic1_view = pic1_view.astype('uint8')
            
            pic2_view = map_img[i - 1].copy() + train_label_batch[i - 1].copy()
            pic2_view = np.where(pic2_view > 255, 255, pic2_view)
            pic2_view = pic2_view.astype('uint8')
            
            pil_image1 = Image.fromarray(input_view)
            pil_image1.save('./CNN_LSTM/data/dataset/picture/train/input/{}.png'.format(file_name1),'PNG')
            
            pil_image2 = Image.fromarray(label_view)
            pil_image2.save('./CNN_LSTM/data/dataset/picture/train/label/{}.png'.format(file_name2),'PNG')
            
            pil_image3 = Image.fromarray(pic1_view)
            pil_image3.save('./CNN_LSTM/data/dataset/picture/train/input/map+input/{}.png'.format(file_name3),'PNG')

            pil_image4 = Image.fromarray(pic2_view)
            pil_image4.save('./CNN_LSTM/data/dataset/picture/train/label/map+label/{}.png'.format(file_name4),'PNG')
        
        for i in range(1, self.test_data_num + 1):
            
            file_name1 = 'input_pose%d'%i
            file_name2 = 'label_pose%d'%i
            file_name3 = 'input_pic%d'%i
            file_name4 = 'label_pic%d'%i
            
            input_view = test_input_batch[i - 1].copy()
            input_view = input_view.astype('uint8')
            
            label_view = test_label_batch[i - 1].copy()
            label_view = label_view.astype('uint8')
            
            pic1_view = map_img[i - 1].copy() + test_input_batch[i - 1].copy()
            pic1_view = np.where(pic1_view > 255, 255, pic1_view)
            pic1_view = pic1_view.astype('uint8')
            
            pic2_view = map_img[i - 1].copy() + test_label_batch[i - 1].copy()
            pic2_view = np.where(pic2_view > 255, 255, pic2_view)
            pic2_view = pic2_view.astype('uint8')
            
            pil_image1 = Image.fromarray(input_view)
            pil_image1.save('./CNN_LSTM/data/dataset/picture/test/input/{}.png'.format(file_name1),'PNG')
            
            pil_image2 = Image.fromarray(label_view)
            pil_image2.save('./CNN_LSTM/data/dataset/picture/test/label/{}.png'.format(file_name2),'PNG')
            
            pil_image3 = Image.fromarray(pic1_view)
            pil_image3.save('./CNN_LSTM/data/dataset/picture/test/input/map+input/{}.png'.format(file_name3),'PNG')

            pil_image4 = Image.fromarray(pic2_view)
            pil_image4.save('./CNN_LSTM/data/dataset/picture/test/label/map+label/{}.png'.format(file_name4),'PNG')  
    
        
if __name__ == '__main__':
    
    dg = Data_Gen()
    dg.data_batch_gen()
    dg.data_picture_gen()
