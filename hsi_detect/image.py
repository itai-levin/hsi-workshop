from spectral import *
from hsi_detect import utils
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

class HyperspectralImage:
    def __init__(self, image_path, smoothing_window=11):
        self.image_path = image_path
        self.lib = envi.open(self.image_path)
        self.image = self.lib.load()
        self.image = utils.smooth_img_spectrum(self.image, window_size=11)
        self.centers = self.lib.bands.centers
        self.rgb = None

    def make_rgb(self, coefficients=(1,1,1)):
        """
        Creates a RGB image from the HSI image
        
        coefficients (3-tuple): multiples for the Red, Green, and Blue channel,
            respectively. Can be used to tune the color balance in the image.
        """
        self.rgb = utils.bandpass_rgb_function (self.image, 
                                      self.centers, 
                                      coeffs=coefficients)
        # add transparency layer
        alpha_layer = ((1-(np.sum(self.image, axis=2)==0))*255).astype(int)
        alpha_layer = alpha_layer[:,:,np.newaxis]
        self.rgb  = np.concatenate([self.rgb , alpha_layer], axis=2).astype(int)
    
    def flatten(self, normalize=True):
        """
        Return 2D-matrix of image (n_pixels,n_wavelengths)
        """
        image = self.image
        if normalize:
            image = image / np.nanmax(image, axis=2, keepdims=True)
        return np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
    
    def show(self, image=None, savepath=None, **fig_params):
        """
        Visualize RGB image of HSI image.
        If RGB image was not prevously created, it is created.
        """
        if self.rgb is None:
            self.make_rgb()
        
        plt.figure(**fig_params)
        if image is None:
            plt.imshow(self.rgb)
        else:
            plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        if savepath is not None:
            plt.savefig(savepath, transparent=True)
        
        plt.show()