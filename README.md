# HSI Workshop

Workshop notebooks and helper code for loading, visualizing, and analyzing hyperspectral images (HSI), with examples for spectral unmixing, reference spectrum comparison, and grid or region-based analysis.

This repository is organized around Jupyter notebooks for interactive analysis and a small `hsi_detect` Python package that contains the reusable image, spectrum, classification, and plotting utilities used by the notebooks.

## Repository Contents

```text
.
|-- auto_find_plates.ipynb
|-- grid_analysis.ipynb
|-- hsi_workshop.ipynb
|-- manual_define_circle.ipynb
|-- manual_define_rectangles.ipynb
|-- requirements.txt
`-- hsi_detect/
    |-- __init__.py
    |-- classifier.py
    |-- grid_analysis.py
    |-- image.py
    |-- spectrum.py
    `-- utils.py
```

## Notebooks

- `hsi_workshop.ipynb` introduces HSI image loading, `.hdr`/data-file handling, basic pixel and spectrum inspection, and reference spectrum workflows.
- `grid_analysis.ipynb` is intended for experiments where samples are arranged in a grid, such as plates or pellet layouts.
- `auto_find_plates.ipynb` explores automated plate/sample detection using OpenCV template matching and spectral band ratios.
- `manual_define_rectangles.ipynb` provides an interactive Dash workflow for selecting rectangular regions of interest and analyzing classification scores or spectra inside those regions.
- `manual_define_circle.ipynb` provides a similar interactive workflow for selecting circular regions of interest.

Some notebooks include hard-coded local paths for image files, concentration maps, reference spectra, and output directories. Update those paths before running the notebooks on a new machine.

## Python Package

The `hsi_detect` package contains reusable code used by the notebooks:

- `hsi_detect.image.HyperspectralImage` loads ENVI-compatible hyperspectral image files through `spectral`, smooths spectra, reconstructs RGB views, flattens image cubes, and displays images.
- `hsi_detect.spectrum.Spectrum` loads reference spectra from `.npy` or `.csv`, interpolates spectra to image wavelengths, and plots spectra.
- `hsi_detect.classifier.HierarchicalKMeansUnmixer` extracts endmembers with hierarchical k-means utilities and classifies hyperspectral images using unmixing against a reference spectrum.
- `hsi_detect.grid_analysis.GridImage` and `GridAnalysis` support grid-based sample layouts, coordinate selection, saved analysis parameters, masking, average spectra extraction, unmixing scores, narrow-band absorbance, and dose-response plotting.
- `hsi_detect.utils` contains lower-level spectral, clustering, masking, fitting, plotting, and image-processing helpers.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name hsi-workshop --display-name "HSI Workshop"
```

Then start Jupyter:

```bash
jupyter notebook
```

Select the `HSI Workshop` kernel when opening the notebooks.

If Jupyter is not already installed in your environment, install your preferred notebook frontend, for example `pip install jupyterlab`, or open the notebooks from VS Code.

## Data Expectations

The GitHub repository tracks the notebooks, package code, and `requirements.txt`. It does not track the large hyperspectral datasets or generated analysis outputs.

The notebooks expect local files such as:

- ENVI `.hdr` files with matching raw/data files in the same directory.
- Reference spectra stored as `.npy` or `.csv` files.
- Concentration maps stored as CSV files for grid or region-based analyses.
- Optional output directories for generated RGB images, plots, JSON coordinate files, and scored images.

For ENVI files, pass the `.hdr` path to the code. The associated data file must remain next to the `.hdr` file so `spectral.envi.open(...)` can load the image.

## Typical Workflow

1. Open `hsi_workshop.ipynb` to inspect an HSI image, reconstruct an RGB view, and understand the reference spectrum workflow.
2. Load or define a reference spectrum with `Spectrum`.
3. Load an image with `HyperspectralImage`.
4. Fit `HierarchicalKMeansUnmixer` to extract endmembers and classify the image.
5. Use either grid analysis or manual region selection to summarize spectra and scores across samples.
6. Export plots, coordinate JSON files, and scored images to an output directory outside the tracked source tree.

## Minimal Example

```python
from hsi_detect.image import HyperspectralImage
from hsi_detect.spectrum import Spectrum
from hsi_detect.classifier import HierarchicalKMeansUnmixer

image = HyperspectralImage("path/to/REFLECTANCE_001.hdr")
image.make_rgb()
image.show()

reference = Spectrum("path/to/reference_spectrum.npy")
reference.interpolate_spectrum(image.centers)

classifier = HierarchicalKMeansUnmixer()
classifier.fit(image, reference)
scored_image = classifier.classify(reference)
```

## Notes

- The package assumes hyperspectral image data can be read by the `spectral` Python package.
- Interactive selection notebooks use Dash/JupyterDash and may open a local browser server on a random port.
- Several analysis notebooks are experiment-specific and should be treated as editable templates.
- Large input data and generated outputs should stay out of Git unless intentionally added through Git LFS or another data-management workflow.
