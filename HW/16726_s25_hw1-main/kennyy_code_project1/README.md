# Image Alignment and Contrast Adjustment

This project provides Python scripts for aligning and adjusting the contrast of images. The code is designed to work with images split into three color channels and can perform both single-scale and multi-scale (pyramid) alignment using either NumPy or PyTorch.

## Installation

To run the scripts, you need to have Python installed along with the following libraries:

- NumPy
- scikit-image
- PyTorch
- os

You can install the required Python packages using pip:

```bash
pip install numpy scikit-image torch
```

## Usage

The main script is `main_hw1.py`, which contains several functions for image processing:

- `contrast(image)`: Adjusts the contrast of an image.
- `single_scale_align(im1, im2)`: Aligns two images using a single-scale approach.
- `pyramid_align(im1, im2)`: Aligns two images using a multi-scale (pyramid) approach.
- `pyramid_align_torch(im1, im2)`: Similar to `pyramid_align`, but uses PyTorch for computations.

### Running the Script

The script can be executed directly. By default, it will run the `main()` function, which processes images in the `data` folder using the pyramid alignment method.

```bash
python main_hw1.py
```

### Functions

- **main()**: Processes images using NumPy for alignment. It saves the aligned images in the `out_image` directory.
- **main_torch()**: Similar to `main()`, but uses PyTorch for image alignment.
- **main_contrast()**: Aligns images and adjusts their contrast before saving.

### Configuration

- The `SINGLE_SCALE` variable in the `main()` and `main_torch()` functions determines whether to use single-scale or pyramid alignment.
- The `data` folder should contain the images you want to process.

## Output

Processed images are saved in the `out_image` directory. The naming convention for the output files indicates whether single-scale or pyramid alignment was used, and whether contrast adjustment was applied.

## Acknowledgments

This project is based on starter code from the 16-726 Project 1 and credits the course materials from [UC Berkeley's CS194-26](https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj1/data/colorize_skel.py).
```
