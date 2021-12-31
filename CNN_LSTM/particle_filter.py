# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:35:41 2021

@author: user
"""

import numpy as np
from numpy.random import uniform 
from scipy import stats

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)

def calculate_heading(v,w,dt=1.):
    
    global previous_x
    global previous_y
    global previous_th
    
    previous_x = 1.0
    previous_y = 1.0
    previous_th = 0.01
    
    dth = w * dt
    
    if previous_x >0:
        heading=np.array([previous_th + dth])
 
        # if heading>0:
        #     heading=-(heading-np.pi)
        # else:
        #     heading=-(np.pi+heading)
            
        distance=np.array([v*dt])
        u=np.array([heading,distance])
        return heading, distance, u

def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles

def predict(particles, u, std, dt=1.):
    N = len(particles)
    dist = (u[1]) + (np.random.randn(N) * std[1]) ## radius
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist
    
def update(particles, weights,landmarks , landmark_prob):
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
    
        distance_particle[distance_particle[:,0]<5,:] = 0
        
        # z_hit
        distance_particle_prob = 1/(distance_particle + 1.e-300) * np.exp(landmark_prob[i])
        distance_particle_prob = distance_particle_prob.reshape(N)
    
    
    
        weights += temp_weights
        weights +=distance_particle_prob
        # weights *= scipy.stats.norm(distance, R).pdf(z[i])  ## change
 
 
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights) 
    return weights


def neff(weights):
    return 1. / np.sum(np.square(weights))    

def systematic_resample(weights):
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

def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)
    
    
if __name__ == "__main__" :
       
    
    x_range=np.array([0,800])
    y_range=np.array([0,600])
    
    std=np.array([2,4])
    
    ###
    example_velocity = 0.2
    example_angle_velocity = 0.01
    initial_pose_x = 2
    initial_pose_y = 3
    landmarks=np.array([ [144,73], [410,13], [336,175], [718,159], [178,484], [665,464]  ])
    landmark_prob = np.array([ [0.2], [0.2], [0.2], [0.2], [0.2], [0.2]])
    ######
    
    
    N=400

    NL = len(landmarks)
    
    particles=create_uniform_particles(x_range, y_range, N)
    
    
    
    weights = np.array([1.0]*N)
    
    # while(1):
    for i in range(1):  
        
        heading, distance, u = calculate_heading(example_velocity,example_angle_velocity,dt=1.)
        
        u=np.array([heading,distance])
        predict(particles, u, std, dt=1.)
        
        weights = update(particles, weights, landmarks=landmarks, landmark_prob =landmark_prob )
        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)
        
        previous_x=initial_pose_x
        previous_y=initial_pose_y
