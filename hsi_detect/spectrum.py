import numpy as np
import matplotlib.pyplot as plt

class Spectrum:
  def __init__(self, file_path=None, wavelengths=None, intensities=None):
    self.file_path = file_path
    if self.file_path is not None and wavelengths is None and intensities is None:
      self.wavelengths, self.intensities = self.load_spectrum()
    elif self.file_path is None and wavelengths is not None and intensities is not None:
      self.wavelengths = wavelengths
      self.intensities = intensities
    else:
      raise ValueError("Provide file_path OR spectrum data, not both.")
    
  def load_spectrum(self):
    if self.file_path.endswith('.npy'):
      data = np.load(self.file_path)
      if data.shape[0] != 2:
        raise ValueError("The npy file must have exactly 2 rows")
      wavelengths = data[0, :]
      intensities = data[1, :]
    elif self.file_path.endswith('.csv'):
      data = np.loadtxt(self.file_path, delimiter=',', skiprows=1)
      if data.shape[1] != 2:
        raise ValueError("The csv file must have exactly 2 columns")
      wavelengths = data[:, 0]
      intensities = data[:, 1]
    else:
      raise ValueError("Unsupported file format. Only .npy and .csv are supported")
    return wavelengths, intensities

  def interpolate_spectrum(self, new_wavelengths):
    self.intensities = np.interp(new_wavelengths, self.wavelengths, self.intensities)
    self.wavelengths = new_wavelengths

  def show(self, savedir=None, **figure_params):
    plt.figure(**figure_params)
    plt.plot(self.wavelengths, self.intensities)
    if savedir is not None:
      plt.savefig(savedir)
    plt.show()