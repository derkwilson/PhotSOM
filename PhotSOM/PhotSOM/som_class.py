import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import sys


class SOM():
    def __init__(self, somsize):
        self.xpix = somsize[0]
        self.ypix = somsize[1]
        self.weights = None
        self.weight_labels = None
        self.tracker = [0., 0., 0.]


    def a_func(self, t, N_iters):
        return np.exp(-t/N_iters) # 0.5 ** (t / N_iters) #


    def HBK(self, sig_of_t, bmu_coords):
        return np.exp(-(self.map_distance(bmu_coords, 'toroidal')) / ((2. * sig_of_t) ** 2))[:, :, np.newaxis]


    def sig_t(self, t, N_iter):
        start = min([self.xpix, self.ypix]) / 10.
        return start * (1. / start) ** (t / N_iter)
    
    def map_distance(self, bmu_coords, topology):
        xinds = np.transpose(np.array([i * np.ones(3*self.xpix) for i in range(-self.xpix, 2*self.xpix)]))
        yinds = np.array([-i * np.ones(3*self.ypix) for i in range(-2*self.ypix+1, self.ypix+1)][::-1])
        d2 = (yinds - bmu_coords[0]) ** 2 + (xinds - bmu_coords[1]) ** 2
        
        if (topology == 'planar'):
            return d2[self.xpix:2*self.xpix, self.ypix:2*self.ypix]
            
        elif (topology == 'toroidal'):
            d2new, k = np.zeros((9, self.xpix, self.ypix)), 0
            for i in range(0, 3*self.ypix, self.ypix):
                for j in range(0, 3*self.xpix, self.xpix):
                    d2new[k] = ( d2[i:i+self.ypix, j:j+self.xpix] )
                    k += 1
            return np.amin(d2new, axis=0)


    def compute_BMU(self, sample_weights, sample_sigs):
        dist_array = (1. / len(sample_weights)) * np.nansum(((self.weights - sample_weights) / sample_sigs) ** 2, axis=2)
        location = np.where(dist_array == np.amin(dist_array)) #bmu
        return (location[0][0], location[1][0])
    
    
    def initialize_weights(self, length_of_weight_array, weight_labels, rseed=123):
        self.weight_labels = weight_labels
        np.random.seed(rseed)
        self.weights = np.random.uniform(size=(self.xpix, self.ypix, length_of_weight_array)) 

                                               
        
    def update_weights(self, a, hbk, training_weights):
        dif = training_weights - self.weights
        np.place(dif, np.isnan(dif), 0)
        self.weights = self.weights + a * hbk * dif
        
    
    
    def train(self, N_iters, df, verbose=True, seed=0):
        if verbose:
            prog = ProgressBar(N_iters)

        wnames = [elem for elem in list(df) if 'ERR' not in elem]
        snames = [elem+'_ERR' for elem in list(df) if 'ERR' not in elem]

        t = 1
        while t <= N_iters:
            if verbose:
                prog.update(t)

            #draw a random training sample, with replacement
            np.random.seed(seed+t)
            training_sample = df.iloc[ int(np.floor(np.random.uniform(0, len(df)))) ]
            sample_weights = training_sample[wnames].values
            sample_sigs    = training_sample[snames].values
                
            #get the map coordinates of the BMU
            bmu       = self.compute_BMU(sample_weights, sample_sigs)
            a         = self.a_func(float(t), N_iters)
            sig_param = self.sig_t(float(t), N_iters)
            hbk       = self.HBK(sig_param, bmu)                
            self.update_weights(a, hbk, sample_weights)
            
            t += 1
            


            
    def save_map(self, filename):
        np.save(filename + '.npy', self.weights)
        np.savetxt(filename + '.txt', self.weight_labels, delimiter=",", fmt="%s") 


    def test(self, dataframe, verbose=True):
        N_iters = len(dataframe.values)
        bmu_z = np.zeros(N_iters)
        if verbose:
            prog = ProgressBar(N_iters)
        wnames = [elem for elem in list(dataframe) if 'ERR' not in elem]
        snames = [elem+'_ERR' for elem in list(dataframe) if 'ERR' not in elem]
        t = 0
        pdf_dataframe = pd.DataFrame()
        while t < N_iters:
            if verbose:
                prog.update(t)
            test_sample = dataframe.iloc[t]
            sample_weights = test_sample[wnames].values
            sample_sigs    = test_sample[snames].values
     
            #get the map coordinates of the BMU
            bmu =  self.compute_BMU(sample_weights, sample_sigs)
            bmu_z[t] = self.weights[bmu][int(self.weight_labels.index('zspec')/2.)]
            t += 1

        return bmu_z





    

    def quantization_error(self, dataframe):
        N_its = len(dataframe.values)
        bmu_dist = np.zeros(N_its)
        t = 0
        while t < N_its:
            test_samp = dataframe.iloc[t]
            samp_weights = test_samp.values[range(0, len(test_samp.values), 2)]
            dist_arr = np.sqrt(np.nansum((self.weights - samp_weights) ** 2, axis=2))
            bmu_dist[t] = np.nanmin(dist_arr)
            t += 1
        return (1. / N_its) * np.nansum(bmu_dist)









class ProgressBar():
    def __init__(self, total_iterations):
        self.total_iters = total_iterations
        self.percs = list(range(100, 0, -5))
        print('0 %')
        
    def update(self, current_iteration):
        check = np.floor(100 * (float(current_iteration) / float(self.total_iters)))
        if check in self.percs:
            sys.stdout.flush()
            print('\r' + str(check) + '%')
            self.percs.pop()
    



def loadSOM(filename):
    weights = np.load(filename + '.npy')
    som = SOM([weights.shape[0], weights.shape[1]])
    som.weights = weights
    som.weight_labels = list(np.loadtxt(filename + '.txt', dtype='str',delimiter=","))
    return som




