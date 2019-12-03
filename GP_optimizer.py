# -*- coding: utf-8 -*-
"""
Online Optimizer 2D
Created on Thu Mar 28 13:47:20 2019

@author: jks7
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

#np.random.seed(1) # this messes up the random number generator when uncommented

class GP_optimizer:
    
    def __init__(self):
        self.name = "sampler1"
    
    def f(self, x, y):          # This will be replaced with an average taken by the RP
        """The function to predict."""
        return np.sin(x) * np.cos(y) / (1 + np.power(x,2) + np.power(y,2))

    def main(self):
        self.x = [-4, 4, 5]
        self.y = [-4, 4, 5]
        self.xy, shape1 = self.create_parameter_space([self.x,self.y])
        xy0 = self.xy                                            # Save initial grid for sampling
        z = []
        for i in range(np.shape(self.xy)[0]):
            z.append(self.f(self.xy[i][0], self.xy[i][1]))
        self.z = np.array(z)
#        print(self.parameter_space_shape)
#        print(xy[:,0])
#        print(xy[0,:])
#        print(np.shape(xy))
#        print(np.shape(z))
    
        # Add noise:
        dz = 0.01 + 0.02 * np.random.random(self.z.shape)
        noise = np.random.normal(0, dz)
        self.z += noise
    
        # Instantiate a Gaussian Process model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel()   # This seems to work and is what Robbie suggested
#        kernel = DotProduct() + WhiteKernel()
#        kernel = 1.0 * RBF(1.0) + 1.0 * WhiteKernel() # this doesn't give an equivalent output to None. Not sure why.
#        kernel = None
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9) # None is passed, the kernel “1.0 * RBF(1.0)” is used as default.
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        print("xy: ", np.shape(self.xy))
        print("z: ", np.shape(self.z))
        self.gp.fit(self.xy, self.z)
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        z_preds, sigma = self.gp.predict(self.xy, return_std=True)    
#        self.plot(self.z, z_preds, self.xy, shape1, plot_residuals = True)    
        self.calculate_avg_residual(self.z, z_preds)
        print("Initial min: %f" % min(z_preds))
        
        xx = [-4, 4, 15]
        yy = [-4, 4, 15]
        xy_dense, shape2 = self.create_parameter_space([xx,yy])
        zdense = []
        for i in range(np.shape(xy_dense)[0]):
            zdense.append(self.f(xy_dense[i][0], xy_dense[i][1]))
        zdense = np.array(zdense)
        zdense_preds, sigma = self.gp.predict(xy_dense, return_std=True) 

#        self.plot(zdense, zdense_preds, xy_dense, shape2, plot_residuals = True) 
        self.calculate_avg_residual(zdense, zdense_preds)
        print("Initial min on dense mesh: %f" % min(zdense_preds))
        self.min_search()
        
        zdense_preds, sigma = self.gp.predict(xy_dense, return_std=True)            # Replot things after the minsearch
#        self.plot(zdense, zdense_preds, xy_dense, shape2, plot_residuals = True) 
        self.calculate_avg_residual(zdense, zdense_preds)
        print("Min on dense mesh after training: %f" % min(zdense_preds))
        
        self.min_search()
        
        zdense_preds, sigma = self.gp.predict(xy_dense, return_std=True)            # Replot things after the minsearch
#        self.plot(zdense, zdense_preds, xy_dense, shape2, plot_residuals = True) 
        self.calculate_avg_residual(zdense, zdense_preds)
        print("Min on dense mesh after training: %f" % min(zdense_preds))
        
        self.min_search()
        
        zdense_preds, sigma = self.gp.predict(xy_dense, return_std=True)            # Replot things after the minsearch
#        self.plot(zdense, zdense_preds, xy_dense, shape2, plot_residuals = True) 
        self.calculate_avg_residual(zdense, zdense_preds)
        print("Min on dense mesh after training: %f" % min(zdense_preds))
    
    def min_search(self):
        bounds = [(self.x[0], self.x[1]), (self.y[0], self.y[1])]
        for i in range(20):                                                     # will be replaced with a while loop
            rands = np.random.rand(1,2)[0] # will need to be updated for higher dimensions
            x_rand = rands[0] * (self.x[1] - self.x[0]) - self.x[1]
            y_rand = rands[1] * (self.y[1] - self.y[0]) - self.y[1]
            guess = np.array([x_rand, y_rand])          # Random Guess in Parameter Space
    #        guess = guess.reshape((1,2))
    #        print(guess)
    #        self.fun(guess)
            xopt = optimize.minimize(self.fun, guess, method='L-BFGS-B', tol = 1e-5, bounds = bounds) #, maxiter = 1e3) # This is a dictionary object
#            print('fun: ', xopt['fun'], 'x: ', xopt['x'])
            self.xy = np.append(self.xy, [xopt['x']], axis = 0)                               # Add new point to the training inputs array
            self.z = np.append(self.z, self.f(xopt['x'][0], xopt['x'][1]))                    # Add new point to the training outputs array
            self.gp.fit(self.xy, self.z)
    
    def fun(self, ins):             # Returns the value predicted by the GP
        ins = ins.reshape((1,2))
        pred = self.gp.predict(ins, return_std = False)
        outs = pred[0]

        return outs

    def create_parameter_space(self, ranges): # Ranges should be a list of tuples of min, max, and num_points values for each dimension
        linspaces = []
        for dimensions in ranges:
            pts = np.linspace(dimensions[0], dimensions[1], dimensions[2])
            linspaces.append(pts)
        all_dims = np.meshgrid(*linspaces)
        ravelled = []
        for i in all_dims:                      # Ravel each value
            ravelled.append(np.ravel(i))
        ravelled = np.array(ravelled)
        state_vectors = np.transpose(ravelled)

        return state_vectors, np.shape(all_dims[0])
        
    def plot(self, z, z_preds, xy, space_shape, plot_residuals = False):        # Called to plot
        Z = z.reshape(space_shape)                 # Sampled data from functions
        Z_preds = z_preds.reshape(space_shape)      # Predictions
        residuals = z - z_preds
        R = residuals.reshape(space_shape)          # Residuals
        
        # Plotting stuff:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d') #fc = 'red'
        x = xy[:,0].reshape(space_shape)
        y = xy[:,1].reshape(space_shape)
        ax.plot_surface(x, y, Z)
#        ax.plot_surface(x, y, Z_preds)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('Sampled Data')
#        ax.legend(['Function', 'Fit'])
        
        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, Z_preds)    
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('Modelled Data')
        
        if plot_residuals == True:
            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, R)  
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_title('Residuals')
        
        plt.show()

    def calculate_avg_residual(self, z, z_preds):
        residuals = z - z_preds
        avg_squared_residual = np.sum(np.square(residuals)) / len(residuals)
        print('Integrated Residuals: %f' % avg_squared_residual)

def main():
    ml = GP_optimizer()
    ml.main()

if __name__ == "__main__":
    main()