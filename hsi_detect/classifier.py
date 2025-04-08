from hsi_detect.utils import *
from abc import ABC, abstractmethod
from hsi_detect import image
from hsi_detect import spectrum
import matplotlib.pyplot as plt

class HSIClassifier(ABC):
  @property
  def classified(self):
    pass

  @abstractmethod
  def fit(self, data):
    pass

  @abstractmethod
  def classify(self, hsi_image:image.HyperspectralImage):
    pass


class HierarchicalKMeansUnmixer(HSIClassifier):
  def __init__(self,
               #Initial clustering parameters
               clustering_method: str = None, # Defaults to MiniBatchKMeans
               reduced_dims: int = 3,
               n_init_clusters:int = 1000,
               # filtering params
               filter_threshold: float = 0.9,
               #agglomerative clustering parameters
               metric:str = 'cosine',
               linkage:str = 'average', 
               distance_threshold:float = 0.005,
               normalize = True
               ):
    
    super().__init__()
    self.clustering_method=clustering_method
    self.reduced_dims = reduced_dims
    self.n_init_clusters = n_init_clusters
    self.filter_threshold = filter_threshold
    self.metric = metric
    self.linkage = linkage
    self.distance_threshold = distance_threshold
    self.normalize = normalize    
    self.em_ls = self.clust_ls = None
  
  def fit(self, image: image.HyperspectralImage, reference_spec: spectrum.Spectrum):
    self.image = image.image
    self.reference_spec = reference_spec
    self.flattened_image = image.flatten()
    if self.normalize:
      self.flattened_image = self.flattened_image / np.nanmax(self.flattened_image, axis=1, keepdims=True)
    self.em_ls, self.clust_ls = kmeans_hierarchical_extract_endmembers(self.flattened_image, reference_spec=reference_spec.intensities,
                                                             reduced_dims=self.reduced_dims, n_clusters=self.n_init_clusters, norm=self.normalize,
                                                             filter_threshold=self.filter_threshold, metric=self.metric,
                                                             linkage=self.linkage, distance_threshold=self.distance_threshold,
                                                             return_cluster_idxs=True
                                                             )

  def classify(self, reference_spec: spectrum.Spectrum, threshold=0, image: image.HyperspectralImage = None, return_all_coeffs: bool = False):
    if self.em_ls is None or self.clust_ls is None or self.flattened_image is None:
      raise ValueError("Endmembers are not defined. Run `fit` first.")
    mat = np.vstack([-reference_spec.intensities, self.em_ls[0]])
    if image is None:
      unmixed = UCLS(self.flattened_image, mat)
      scored_img = unmixed[:,0] 
      scored_img = np.reshape(scored_img, self.image.shape[:2])
    else: # score a different image than fit
      unmixed = UCLS(image.flatten(), mat)
      scored_img = unmixed[:,0]
      scored_img = np.reshape(scored_img, image.image.shape[:2])

    if threshold is not None:
      scored_img[scored_img<threshold] = 0
    
    if return_all_coeffs:
      return scored_img, unmixed
    
    return scored_img

  def visualize_clusters(self):
    if self.clust_ls is None :
      raise ValueError("Endmembers are not defined. Run `fit` first.")
    
    plt.figure()
    plt.imshow(np.reshape(self.clust_ls[0], self.image.shape[:2]), cmap='tab20')
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.show()

  def visualize_endmembers(self):
    if self.em_ls is None :
      raise ValueError("Endmembers are not defined. Run `fit` first.")
    
    plt.figure()
    for em in self.em_ls[0]:
      plt.plot(self.reference_spec.wavelengths, em)
    plt.show()


    
    


    
  
    