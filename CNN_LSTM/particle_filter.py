# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:35:41 2021

@author: user
"""

import numpy as np
from numpy.random import uniform
from scipy import stats

np.set_printoptions(threshold=5, suppress=True)

class Particle_filter():
    
    def __init__(self):
        
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_th = 0.01
        

    def calculate_heading(self,v, w, dt=0.2):
        
        global previous_x
        global previous_y
        global previous_th
        
        previous_x = 1.0
        previous_y = 1.0
        previous_th = 0.01
        
        dth = w * dt
        
        if previous_x > 0:
            heading=np.array([previous_th + dth])
     
            # if heading>0:
            #     heading=-(heading-np.pi)
            # else:
            #     heading=-(np.pi+heading)
                
            distance=np.array([v*dt])
            u=np.array([heading,distance])
            
            return heading, distance, u
    
    def create_uniform_particles(self,x_range, y_range, N,initial_pose_x,initial_pose_y):
        
        particles = np.empty((N, 2))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N) + initial_pose_x
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N) + initial_pose_y
        
        return particles
    
    def predict(self,particles, u, std, dt=0.2):
        N = len(particles)
        dist = (u[1]) + (np.random.randn(N) * std[1]) ## radius
        particles[:, 0] += np.cos(u[0]) * dist
        particles[:, 1] += np.sin(u[0]) * dist
        
        # print(u[0])
        
    def update(self,particles, weights, landmarks , landmark_prob):
        weights.fill(1.)
        
        for i, landmark in enumerate(landmarks):
    
            ##Calculate the distance from each particle to trajectory
            distance_particle=np.power((particles[:,0] - landmark[0])**2 +(particles[:,1] - landmark[1])**2,0.5)
            distance_particle = distance_particle.reshape(N,1)
            
            #zhat 
            distance_x_y = np.power((initial_pose_x - landmark[0])**2 +(initial_pose_y - landmark[1])**2,0.5)
        
            #z_max, distance_threshold = 200
            temp_weights = stats.norm(distance_x_y,100).pdf(distance_particle)
            temp_weights = temp_weights.reshape(N,1)
            temp_weights[distance_particle[:,0]<200.0,:] = 0
            temp_weights = temp_weights.reshape(N)
        
            distance_particle[distance_particle[:,0] < 5, :] = 0
            
            # z_hit
            distance_particle_prob = 1/(distance_particle + 1.e-300) * np.exp(landmark_prob[i])
            distance_particle_prob = distance_particle_prob.reshape(N)
        
    
            weights += temp_weights
            weights +=distance_particle_prob
            # weights *= scipy.stats.norm(distance, R).pdf(z[i])  ## change
     
     
        weights += 1.e-300 # avoid round-off to zero
        weights /= sum(weights)
        
        return weights
    
    
    def neff(self,weights):
        return 1. / np.sum(np.square(weights))    
    
    def systematic_resample(self,weights):
        N = len(weights)
        positions = (np.arange(N) + np.random.random()) / N
     
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        
        while i < N and j<N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                
        return indexes
    
    def estimate(self,particles, weights):
        pos = particles[:, 0:1]
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean)**2, weights=weights, axis=0)
        
        return mean, var
    
    def resample_from_index(self,particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights /= np.sum(weights)
        
        return particles, weights
    
    
if __name__ == "__main__" :
    
    pf = Particle_filter()


    final_output = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/final_output.npy')
    final_vw = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/final_vw.npy')
    
    final_vw[:, :, 0] = final_vw[:, :, 0] * 20
    
    test_label_file_name = 'batch_test_label_data2'
    vel_test_label = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/test/{}.npy'.format(test_label_file_name))
    num = 10
    
    b = final_output[num].copy()
    a = final_vw[num].copy()
    c = final_output[num].copy()

    
    b_shape = b.shape

    prob_position = np.zeros([1,2])
    prob_p = np.zeros([1,1])
    

    b_max = np.max(b)
    b_min = np.min(b)


    b = (b-b_min)/(b_max-b_min)

    count = 0

    for i in range(0,b_shape[0]):
  
        for j in range(0,b_shape[1]):

    

            if b[i, j] > 0.3 :

                if count == 0 :
        
                    prob_position = prob_position + np.array([i,j])
                    prob_p = prob_p + b[i, j]

                else:
                    prob_position = np.vstack((prob_position,np.array([i,j])))
                    prob_p = np.vstack((prob_p,b[i, j]))

            count = count + 1
    
    x_range=np.array([-75,75])
    y_range=np.array([-75,75])
    
    std=np.array([2,4])
    
    # ###
    example_velocity = a[0, 0]
    example_angle_velocity = a[0, 1]
    initial_pose_x = 141
    initial_pose_y = 656
    landmarks=prob_position
    landmark_prob = prob_p
    # ######
    
    
    predict_trajectory = np.array([[initial_pose_x,initial_pose_y]])
    
    N=300

    NL = len(landmarks)
    
    particles= pf.create_uniform_particles(x_range, y_range, N,initial_pose_x,initial_pose_y)
    
    
    
    weights = np.array([1.0]*N)
    
    # # # while(1):
    for i in range(16):
        heading, distance, u = pf.calculate_heading(a[i, 0],a[i, 1],dt=0.2)
        
        u=np.array([heading,distance])
        pf.predict(particles, u, std, dt=0.2)
        
        weights = pf.update(particles, weights, landmarks=landmarks, landmark_prob =landmark_prob )
        indexes = pf.systematic_resample(weights)
        particles, weights = pf.resample_from_index(particles, weights, indexes)
        
        max_weights = np.argmax(weights)
        previous_x = particles[max_weights][0]
        previous_y = particles[max_weights][1]
        
        predict_trajectory = np.vstack((predict_trajectory,np.array([previous_x,previous_y])))
    
    # for num in range(22):
        # b = final_output[num].copy()
        # a = final_vw[num].copy()
        # c = final_output[num].copy()
    
        
        # b_shape = b.shape
    
        # prob_position = np.zeros([1,2])
        # prob_p = np.zeros([1,1])
        
    
        # b_max = np.max(b)
        # b_min = np.min(b)
    
    
        # b = (b-b_min)/(b_max-b_min)
    
        # count = 0
    
        # for i in range(0,b_shape[0]):
      
        #     for j in range(0,b_shape[1]):
    
        
    
        #         if b[i][j] > 0.3 :
    
        #             if count == 0 :
            
        #                 prob_position = prob_position + np.array([i,j])
        #                 prob_p = prob_p + b[i][j]
    
        #             else:
        #                 prob_position = np.vstack((prob_position,np.array([i,j])))
        #                 prob_p = np.vstack((prob_p,b[i][j]))
    
        #         count = count + 1
        
        # x_range=np.array([-25,25])
        # y_range=np.array([-25,25])
        
        # std=np.array([2,4])
        
        # # ###
        # example_velocity = a[0][0]
        # example_angle_velocity = a[0][1]
        # initial_pose_x = 155
        # initial_pose_y = 656
        # landmarks=prob_position
        # landmark_prob = prob_p
        # # ######
        
        
        # predict_trajectory = np.array([[initial_pose_x,initial_pose_y]])
        
        # N=300
    
        # NL = len(landmarks)
        
        # particles= pf.create_uniform_particles(x_range, y_range, N,initial_pose_x,initial_pose_y)
        
        
        
        # weights = np.array([1.0]*N)
        
        # # # # while(1):
        # for i in range(16):
        #     heading, distance, u = pf.calculate_heading(a[i, 0],a[i, 1],dt=0.2)
            
        #     u=np.array([heading,distance])
        #     pf.predict(particles, u, std, dt=1.)
            
        #     weights = pf.update(particles, weights, landmarks=landmarks, landmark_prob =landmark_prob )
        #     indexes = pf.systematic_resample(weights)
        #     particles, weights = pf.resample_from_index(particles, weights, indexes)
            
        #     max_weights = np.argmax(weights)
        #     previous_x = particles[max_weights][0]
        #     previous_y = particles[max_weights][1]
            
        #     predict_trajectory = np.vstack((predict_trajectory,np.array([previous_x,previous_y])))
            
            # previous_x=initial_pose_x
            # previous_y=initial_pose_y
            
        # np.save('./CNN_LSTM/data/test/pf_predict_trajectory%d'%num, predict_trajectory)
        
        

