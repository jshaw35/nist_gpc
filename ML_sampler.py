# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:17:42 2019

@author: jks7
Class for creating samplings for PID optimization and running a gaussian process ML program
"""
#            (kp, fi, fii, fd, fdf, fmin, fmax, gain_min, gain_max, bLock) = self.qloop_filters.getSettings()



from __future__ import print_function
import time
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

class ML_sampler:
    
    def __init__(self, mainwindow = None, qloop_filters = None):
        
        self.mainwindow = mainwindow # Main window handles most calls
        self.qloop_filters = qloop_filters
        self.ISPD_sum = [0,0]   # for ML application
        
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel()   # This seems to work and is what Robbie suggested
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9) # None is passed, the kernel “1.0 * RBF(1.0)” is used as default.
        
    def create_sparse_sampling(self):
        # get PID ranges
        (_, _, _, _, _, fmin, fmax, gain_min, gain_max, _) = self.qloop_filters.getSettings()
        

        gain_bounds = np.array([-3e2, gain_max])          # Added my own lower bounds from experience
        offset = gain_bounds[0]
        gain_bounds -= offset
        coeff_bounds = np.array([0, fmax])
        gain_bounds += 1
        coeff_bounds += 1
        gain_bounds = np.log10(gain_bounds)         # Convert to log scale for reasonable sampling
        coeff_bounds = np.log10(coeff_bounds)
#        gain_bounds_slider_units = 100*np.log10(gain_bounds)
#        coeff_bounds_slider_units = 100*np.log10(coeff_bounds)
# Need to convert back to qedit values to avoid slider limits

        print('fmin: %s' % str(fmin))
        print('fmax: %s' % str(fmax))
        print('gain_min: %s' % str(gain_min))
        print('gain_max: %s' % str(gain_max))
#        print('gain bounds: %s' % str(gain_bounds_slider_units))
#        print('coeff bounds: %s' % str(coeff_bounds_slider_units))
        
        spd = 3   # int(round(np.log(samples)/np.log(5),0)) # number of steps in each dimension to sparsely cover the space

        gain_step = (gain_bounds[1] - gain_bounds[0]) / (spd + 1)       # Steps in which to somewhat evenly sample the space
        coeff_step = (coeff_bounds[1] - coeff_bounds[0]) / (spd + 1)
        
        sampling = []
        # Create samplings to fill 5-D space
        for i in range(1,spd + 1):
            for j in range(1,spd + 1):
                for k in range(1,spd + 1):
                    for l in range(1,spd + 1):
                        for m in range(1,spd + 1):
                            gain_val = np.power(gain_bounds[0] + gain_step * i, 10) + offset
                            fi_val = np.power(coeff_bounds[0] + coeff_step * j, 10)
                            fii_val = np.power(coeff_bounds[0] + coeff_step * k, 10)
                            fd_val = np.power(coeff_bounds[0] + coeff_step * l, 10)
                            fdf_val = np.power(coeff_bounds[0] + coeff_step * m, 10)
                            sampling.append([gain_val, fi_val, fii_val, fd_val, fdf_val, 0]) # Last empty value is for appending the outcome

        sampling = np.array(sampling) # put into numpy array format for easy indexing!
        kps = sampling[:,0]
        fis = sampling[:,1]
        fiis = sampling[:,2]
        fds = sampling[:,3]
        fdfs = sampling[:,4]
        
        # check functionality of sampling   
        plt.subplot(2, 1, 1)
        plt.loglog(kps, fis, 'o-')
        plt.title('A tale of 2 subplots')
        plt.xlabel('kp')
        plt.ylabel('fi')
        
        plt.subplot(2, 1, 2)
        plt.loglog(fiis, fds, '.-')
        plt.xlabel('fii')
        plt.ylabel('fd')
        
#        plt.show()
        
        return sampling
        
    def create_localized_sampling(self):
        # get current PID values
        if self.qloop_filters != None:
            print('Using real PID values')
            (kp, fi, fii, fd, fdf, fmin, fmax, gain_min, gain_max, bLock) = self.qloop_filters.getSettings()
        else: 
            kp = 1; fi = 1; fii = 1; fd = 1; fdf = 1
        # get range to sample
        # First only feedback on kp and fi. Range is 2 orders of magnitude centered around current value
        kp_range = [0.1*kp, 10*kp]
        fi_range = [0.1*fi, 10*fi]
        rands = np.random.rand(25,2) # Creates random state vectors. Will need to be updated for higher dimensions
        lognormal_rands = np.random.lognormal(0, 1.5, 50)       # Use a lognormal distribution to evenly cover vals above and below setpoint
        lognormal_rands = lognormal_rands.reshape((25,2))
        print(lognormal_rands)
        sampling  = [[kp, fi, fii, fd, fdf, 0]] # include current value 
        for i in lognormal_rands:
            kp_rand  = i[0] * kp #(kp_range[1] - kp_range[0]) + kp_range[0]
            fi_rand  = i[1] * fi #(fi_range[1] - fi_range[0]) + fi_range[0]
            sampling.append([kp_rand, fi_rand, fii, fd, fdf, 0])
        sampling = np.array(sampling)
        print(np.shape(sampling))
#        print(sampling[:,:2]) # this will make it easy to choose which variables to train on
#        print(sampling[:,-1]) # These will be the output values after sampling
        return sampling # or self.sampling = sampling ? 
        
    def collect_training_data(self, sampling):
        # communicate with main window1
        self.mainwindow.collecting_training_data = True
            
    def update_training_data(self, completed_sampling):
        training_data = completed_sampling[:,:2] # Just the first two variables for now
        outputs = completed_sampling[:,-1]        # Median Integrated PSD
        
        self.gp.fit(training_data, outputs)
        
    def optimize_lock(self): # Use current gp to minimize the integrated PSD and set PIDs to that value
        (kp, fi, fii, fd, fdf, fmin, fmax, gain_min, gain_max, bLock) = self.qloop_filters.getSettings()  # Get settings and use to set bounds
        bounds = [(0.1*kp, 10*kp), (0.1*fi, 10*fi)]
        sampling = self.create_localized_sampling()  # Creates 26 random points in parameter space
        best_setting = [sampling[0], 100]
        for params in sampling:
            guess = params[:2] # First two values for now
            xopt = optimize.minimize(self.gp_prediction, guess, method='L-BFGS-B', tol = 1e-5, bounds = bounds) #, maxiter = 1e3) # This is a dictionary object
            if xopt['fun'] < best_setting[1]: # If the gp predict a better lock, update the recommended settings
                best_settings = xopt['x']
        best_settings = np.append(best_settings, [fii, fd, fdf])
        print("Best settings: " + str(xopt['x']))    
#        print(" Please... " + str(sampling[0]))   # Thank god!         
        return best_settings

    def gp_prediction(self, ins):             # Returns the value predicted by the GP
        ins = ins.reshape((1,2))
        pred = self.gp.predict(ins, return_std = False)
        outs = pred[0]

        return outs
            
def main():
    ml = ML_sampler()
#    ml.create_sparse_sampling()
    ml.create_localized_sampling()
            
            
if __name__ == "__main__": # testing function when called from the command line
    main()