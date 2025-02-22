# (16-726): Project 1 starter Python code
# credit to https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj1/data/colorize_skel.py
# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import os
import skimage.io as skio
import skimage.color as skcolor
import skimage.filters as skfilters
from skimage import exposure
import torch
import skimage as sk
# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def contrast(image):
    contrast_c = 1.2
    mean_image = np.mean(image)
    adjusted_image = np.copy(image)
    
    adjusted_image[adjusted_image < mean_image] /= contrast_c
    adjusted_image[adjusted_image > mean_image] *= contrast_c

    max_image, min_image = np.max(adjusted_image), np.min(adjusted_image)
    
    if max_image == min_image:
        return adjusted_image
    
    adjusted_image = (adjusted_image - min_image) / (max_image - min_image)
    return adjusted_image

def single_scale_align(im1, im2):
    # align im1 on topic of im2
    windows = [-15, 15]
    shifting = [0,0]
    min_ssd = np.Inf
    best_im1 = np.copy(im1)  # Initialize with the original image

    for shift_x in range(windows[0], windows[1]):
        for shift_y in range(windows[0], windows[1]):
            shift_im1 = np.roll(im1, (shift_x,shift_y), axis=(1,0))

            ssd = np.sum((shift_im1 - im2)**2)
            if ssd < min_ssd:
                min_ssd = ssd
                shifting = [shift_x, shift_y]
                best_im1 = shift_im1

    return best_im1, shifting

def pyramid_align(im1, im2):
    # align im1 on topic of im2
    windows = [-15, 15]
    total_shift = [0,0]
    shifting = [0,0]
    best_im1 = np.copy(im1)  # Initialize with the original image

    scale = 2

    layers = np.floor(np.log(im1.shape[0] / (100 * scale)) / np.log(scale)).astype(int) + 1
    print(layers)

    # crop the edges of the image
    portion = 0.15

    for layer in range(layers, -1, -1):
        min_ssd = np.Inf
        scale_factor = 1 / (scale ** layer)
        scaled_im1 = sk.transform.rescale(im1, scale_factor, anti_aliasing=True)
        scaled_im2 = sk.transform.rescale(im2, scale_factor, anti_aliasing=True)

        # crop the edges of the image
        scaled_im1 = scaled_im1[int(portion * scaled_im1.shape[0]): int((1 - portion) * scaled_im1.shape[0]), int(portion * scaled_im1.shape[1]): int((1 - portion) * scaled_im1.shape[1])]
        scaled_im2 = scaled_im2[int(portion * scaled_im2.shape[0]): int((1 - portion) * scaled_im2.shape[0]), int(portion * scaled_im2.shape[1]): int((1 - portion) * scaled_im2.shape[1])]

        for shift_x in range(shifting[0] + windows[0], shifting[0] + windows[1] + 1):
            for shift_y in range(shifting[1] + windows[0], shifting[1] + windows[1] + 1):
                shift_scaled_im1 = np.roll(scaled_im1, (shift_x,shift_y), axis=(1,0))

                ssd = np.sum((shift_scaled_im1 - scaled_im2)**2)
                if ssd < min_ssd:
                    min_ssd = ssd
                    shifting = [shift_x , shift_y]
        
        shifting = [shifting[0] * scale, shifting[1] * scale]
        total_shift = [total_shift[0] + shifting[0], total_shift[1] + shifting[1]]
        print("Shift ", total_shift, "at scale: 1/", scale ** layer)

    total_shift = [total_shift[0] - int(shifting[0]), total_shift[1] - int(shifting[1])]
    shifting = [int(shifting[0] / scale), int(shifting[1] / scale)]
    best_im1 = np.roll(best_im1, shift=(shifting[0], shifting[1]), axis=(1,0))

    return best_im1, total_shift

def pyramid_align_torch(im1, im2):
    # Convert images to PyTorch tensors
    im1 = torch.tensor(im1, dtype=torch.float32)
    im2 = torch.tensor(im2, dtype=torch.float32)

    windows = [-15, 15]
    total_shift = [0, 0]
    shifting = [0, 0]
    best_im1 = im1.clone()  # Initialize with the original image

    scale = 2

    layers = int(torch.floor(torch.log(torch.tensor(im1.shape[0] / (100 * scale))) / torch.log(torch.tensor(scale)))) + 1
    print(layers)

    # crop the edges of the image
    portion = 0.15

    for layer in range(layers, -1, -1):
        min_ssd = float('inf')
        scale_factor = 1 / (scale ** layer)
        scaled_im1 = sk.transform.rescale(im1.numpy(), scale_factor, anti_aliasing=True)
        scaled_im2 = sk.transform.rescale(im2.numpy(), scale_factor, anti_aliasing=True)

        # Convert back to tensors
        scaled_im1 = torch.tensor(scaled_im1, dtype=torch.float32)
        scaled_im2 = torch.tensor(scaled_im2, dtype=torch.float32)

        # crop the edges of the image
        scaled_im1 = scaled_im1[int(portion * scaled_im1.shape[0]): int((1 - portion) * scaled_im1.shape[0]), int(portion * scaled_im1.shape[1]): int((1 - portion) * scaled_im1.shape[1])]
        scaled_im2 = scaled_im2[int(portion * scaled_im2.shape[0]): int((1 - portion) * scaled_im2.shape[0]), int(portion * scaled_im2.shape[1]): int((1 - portion) * scaled_im2.shape[1])]

        for shift_x in range(shifting[0] + windows[0], shifting[0] + windows[1] + 1):
            for shift_y in range(shifting[1] + windows[0], shifting[1] + windows[1] + 1):
                shift_scaled_im1 = torch.roll(scaled_im1, shifts=(shift_x, shift_y), dims=(1, 0))

                ssd = torch.sum((shift_scaled_im1 - scaled_im2) ** 2).item()
                if ssd < min_ssd:
                    min_ssd = ssd
                    shifting = [shift_x, shift_y]

        shifting = [shifting[0] * scale, shifting[1] * scale]
        total_shift = [total_shift[0] + shifting[0], total_shift[1] + shifting[1]]
        print("Shift ", total_shift, "at scale: 1/", scale ** layer)

    total_shift = [total_shift[0] - int(shifting[0]), total_shift[1] - int(shifting[1])]
    shifting = [int(shifting[0] / scale), int(shifting[1] / scale)]
    best_im1 = torch.roll(best_im1, shifts=(shifting[0], shifting[1]), dims=(1, 0))

    return best_im1.numpy(), total_shift

def main():
    folder = "data"
    data = [folder + "/" + item for item in os.listdir(folder)]

    SINGLE_SCALE = False

    for imname in data:
        # name of the input file
        print("Process alignment the file: " + imname)
        # read in the image
        im = skio.imread(imname)

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)
            
        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(np.int32)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        print("image resolution: ", b.shape)
        if "emir" in imname:
            if SINGLE_SCALE:    
                ab, shifting_b = single_scale_align(b, g)
                ar, shifting_r = single_scale_align(r, g)      
            else:
                ab, shifting_b = pyramid_align(b, g)
                ar, shifting_r = pyramid_align(r, g)
            im_out = np.dstack([ar,g,ab])

        else:
            if SINGLE_SCALE:    
                ag, shifting_g = single_scale_align(g, b)
                ar, shifting_r = single_scale_align(r, b)                
            else:
                ag, shifting_g = pyramid_align(g, b)
                ar, shifting_r = pyramid_align(r, b)
            # create a color image
            im_out = np.dstack([ar, ag, b])
        
        im_out *= 255
        im_out = im_out.astype(np.uint8)

        # save the image
        if SINGLE_SCALE:
            fname = './out_image/' + "single_scale_" + imname.split('/')[-1].rsplit('.', 1)[0] + '.jpg'
        else:
            fname = './out_image/' + "pyramid_" + imname.split('/')[-1].rsplit('.', 1)[0] + '.jpg'
        skio.imsave(fname, im_out)


def main_torch():
    folder = "data"
    data = [folder + "/" + item for item in os.listdir(folder)]

    SINGLE_SCALE = False

    for imname in data:
        print("Process alignment the file: " + imname)
        im = skio.imread(imname)
        im = sk.img_as_float(im)
        
        # Convert image to PyTorch tensor
        im = torch.tensor(im, dtype=torch.float32)
            
        # compute the height of each part (just 1/3 of total)
        height = torch.floor(torch.tensor(im.shape[0] / 3.0)).int()

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        print("image resolution: ", b.shape)
        if "emir" in imname:
            if SINGLE_SCALE:    
                ab, shifting_b = single_scale_align(b.numpy(), g.numpy())
                ar, shifting_r = single_scale_align(r.numpy(), g.numpy())
                ab, ar = torch.tensor(ab), torch.tensor(ar)
            else:
                ab, shifting_b = pyramid_align_torch(b, g)
                ar, shifting_r = pyramid_align_torch(r, g)
            im_out = torch.stack([ar, g, ab], dim=2)

        else:
            if SINGLE_SCALE:    
                ag, shifting_g = single_scale_align(g.numpy(), b.numpy())
                ar, shifting_r = single_scale_align(r.numpy(), b.numpy())
                ag, ar = torch.tensor(ag), torch.tensor(ar)
            else:
                ag, shifting_g = pyramid_align_torch(g, b)
                ar, shifting_r = pyramid_align_torch(r, b)
            # create a color image
            im_out = torch.stack([ar, ag, b], dim=2)
        
        im_out *= 255
        im_out = im_out.to(torch.uint8)

        # Convert back to numpy for saving
        im_out = im_out.numpy()

        # save the image
        if SINGLE_SCALE:
            fname = './out_image/' + "single_scale_" + imname.split('/')[-1].rsplit('.', 1)[0] + '.jpg'
        else:
            fname = './out_image/' + "pyramid_" + imname.split('/')[-1].rsplit('.', 1)[0] + '.jpg'
        skio.imsave(fname, im_out)


def main_constract():
    folder = "data"
    data = [folder + "/" + item for item in os.listdir(folder)]

    for imname in data:
        print("Process alignment the file: " + imname)
        im = skio.imread(imname)

        im = sk.img_as_float(im)
            
        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(np.int32)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        print("image resolution: ", b.shape)

        if "emir" in imname:

            ab, shifting_b = pyramid_align(b, g)
            ar, shifting_r = pyramid_align(r, g)
            im_out = np.dstack([ar,g,ab])


        else:
            ag, shifting_g = pyramid_align(g, b)
            ar, shifting_r = pyramid_align(r, b)
            # create a color image
            im_out = np.dstack([ar, ag, b])

        im_out = contrast(im_out)

        im_out *= 255   
        im_out = im_out.astype(np.uint8)

        fname = './out_image/' + "pyramid_" + imname.split('/')[-1].rsplit('.', 1)[0] + '_contrast' + '.jpg'
        skio.imsave(fname, im_out)


if __name__ == "__main__":
    main()
    # main_torch()
    # main_constract() 