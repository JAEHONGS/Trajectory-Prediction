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
        self.data_num = 100
        
        self.file_name1 = 'batch_train_data1'
        self.label_file_name1 = 'batch_label_data1'
        self.data_height1 = 310
        self.data_width1 = 1000
        self.channel_n = 4

        self.file_name2 = 'batch_train_data2'
        self.label_file_name2 = 'batch_label_data2'
        self.data_height2 = 300
        self.data_width2 = 4
        
        
    def data_batch_gen(self):
        
        map_imgArray = np.array(self.map_img)
        map_imgArray = 255 - map_imgArray
        
        plus_map = np.zeros((1,1000), dtype = np.uint8) + 50
        
        map_imgArray = np.concatenate([map_imgArray, plus_map], axis = 0)
        map_imgArray1 = map_imgArray.copy()
        map_imgArray1 = np.reshape(map_imgArray1, (1,310, 1000))
        
        empty_train_batch1 = np.zeros((self.data_num, self.data_height1, self.data_width1, self.channel_n))
        empty_label_batch1 = np.zeros((self.data_num, self.data_height1, self.data_width1))
        
        empty_train_batch2 = np.zeros((self.data_num, self.data_height2, self.data_width2))
        empty_label_batch2 = np.zeros((self.data_num, self.data_height2, self.data_width2))

        empty_train_batch1[:, :, :, 0] = map_imgArray1.copy()
        
        for i in range(1, self.data_num + 1):
         
            file_name = 'pose%d'%i
            pose_path = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/dataset/{}.npy'.format(file_name))
            pose_path[:, 0] = np.round(pose_path[:,0] * 20)  # 20 : map resolution. this converts position to map index. 
            pose_path[:, 1] = np.round(pose_path[:,1] * 20)
            
            for j in range(len(pose_path)): # Creating batch data
                
                empty_train_batch1[i - 1][int(310 - pose_path[j][1])][int(pose_path[j][0])][1] = 255
                empty_train_batch1[i - 1][int(310 - pose_path[j][1])][int(pose_path[j][0])][2] = pose_path[j][3]
                empty_train_batch1[i - 1][int(310 - pose_path[j][1])][int(pose_path[j][0])][3] = pose_path[j][4]
                
                empty_label_batch1[i - 1][int(310 - pose_path[j][1])][int(pose_path[j][0])] = 255
                
                if j !=  (len(pose_path) -1):
                    
                    empty_train_batch2[i - 1][j][0] = int(310 - pose_path[j][1])
                    empty_train_batch2[i - 1][j][1] = int(pose_path[j][0])
                    empty_train_batch2[i - 1][j][2] = pose_path[j][3]
                    empty_train_batch2[i - 1][j][3] = pose_path[j][4]
                    
                    empty_label_batch2[i - 1][j][0] = int(310 - pose_path[j + 1][1])
                    empty_label_batch2[i - 1][j][1] = int(pose_path[j + 1][0])
                    empty_label_batch2[i - 1][j][2] = pose_path[j + 1][3]
                    empty_label_batch2[i - 1][j][3] = pose_path[j + 1][4]
            
            map_imgArray_view = map_imgArray.copy()    
            
            for k in range(len(pose_path)):  # creating path on map
        
                map_imgArray_view[int(310 - pose_path[k][1])][int(pose_path[k][0])] = 255
                
                        
            pil_image = Image.fromarray(map_imgArray_view)
            #pil_image.show()
            pil_image.save('./CNN_LSTM/data/dataset/picture/{}.png'.format(file_name),'PNG')
            
        for cc in range(self.data_num): # some weird data exist, so eliminated.
            empty_train_batch2[cc][299][0] = 0
            empty_train_batch2[cc][299][1] = 0
            empty_label_batch2[cc][299][0] = 0
            empty_label_batch2[cc][299][1] = 0
            
        empty_train_batch1[:, :, 656:1001, 1:4] = 0
        empty_label_batch1[:, :, 0:656] = 0
        
        np.save('./CNN_LSTM/data/train/{}'.format(self.file_name1), empty_train_batch1)
        np.save('./CNN_LSTM/data/train/{}'.format(self.label_file_name1), empty_label_batch1)
        
        np.save('./CNN_LSTM/data/train/{}'.format(self.file_name2), empty_train_batch2)
        np.save('./CNN_LSTM/data/train/{}'.format(self.label_file_name2), empty_label_batch2)
        
        
    def data_picture_gen(self):
        
        data_batch = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(self.file_name1))
        label_batch = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(self.label_file_name1))
        
        map_img = data_batch[:, :, :, 0].copy()
        
        input_batch = data_batch[:, :, :, 1].copy()
        
        for i in range(1, self.data_num + 1):
            
            file_name1 = 'input_pose%d'%i
            file_name2 = 'label_pose%d'%i
            file_name3 = 'input_pic%d'%i
            file_name4 = 'label_pic%d'%i
            
            input_view = input_batch[i - 1].copy()
            input_view = input_view.astype('uint8')
            
            label_view = label_batch[i - 1].copy()
            label_view = label_view.astype('uint8')
            
            pic1_view = map_img[i - 1].copy() + input_batch[i - 1].copy()
            pic1_view = np.where(pic1_view > 255, 255, pic1_view)
            pic1_view = pic1_view.astype('uint8')
            
            pic2_view = map_img[i - 1].copy() + label_batch[i - 1].copy()
            pic2_view = np.where(pic2_view > 255, 255, pic2_view)
            pic2_view = pic2_view.astype('uint8')
            
            pil_image1 = Image.fromarray(input_view)
            pil_image1.save('./CNN_LSTM/data/dataset/picture/input/{}.png'.format(file_name1),'PNG')
            
            pil_image2 = Image.fromarray(label_view)
            pil_image2.save('./CNN_LSTM/data/dataset/picture/label/{}.png'.format(file_name2),'PNG')
            
            pil_image3 = Image.fromarray(pic1_view)
            pil_image3.save('./CNN_LSTM/data/dataset/picture/input/map+input/{}.png'.format(file_name3),'PNG')

            pil_image4 = Image.fromarray(pic2_view)
            pil_image4.save('./CNN_LSTM/data/dataset/picture/label/map+label/{}.png'.format(file_name4),'PNG')            
    
        
if __name__ == '__main__':
    
    dg = Data_Gen()
    dg.data_batch_gen()
    dg.data_picture_gen()