# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import scipy

def toy_recon(image):
    """
    Toy problem for gradient-domain reconstruction.
    :param image: (H, W) input image
    :return: (H, W) output image
    """
    imh, imw = image.shape
    # im2var(x,y) = x*imh + y
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    print("im2var:", im2var)

    # x dimention + y dimention + 1 (top left corner)
    A = np.zeros((imh * imw * 2 + 1 , imh * imw))
    b = np.zeros((imh * imw * 2 + 1))

    e = 0
    for i in range(imh):
        # row-wise x-gredient
        for j in range(imw - 1):
            A[e, im2var[i, j]] = -1
            A[e, im2var[i, j + 1]] = 1
            b[e] = image[i, j + 1] - image[i, j]
            e += 1

    for j in range(imw):
        # column-wise y-gredient
        for i in range(imh - 1):
            A[e, im2var[i, j]] = -1
            A[e, im2var[i + 1, j]] = 1

            b[e] = image[i + 1, j] - image[i, j]
            e += 1

    # top left corner
    A[e, im2var[0, 0]] = 1
    b[e] = image[0, 0]
    
    sA = scipy.sparse.csr_matrix(A)
    x = scipy.sparse.linalg.lsqr(sA, b)
    x = x[0] * 255
    # var2im
    res = x.reshape((imh, imw)).astype(int)
    return res

def img2var_f(img):
    """
    Convert an image to a variable.
    :param img: (H, W) input image
    :return: (H, W) output image
    """
    try:
        imh, imw = img.shape
    except:
        imh, imw, _ = img.shape
    # im2var(x,y) = x*imh + y
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    return im2var

def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    imh, imw, c = fg.shape
    # im2var(x,y) = x*imh + y
    im2var = img2var_f(fg)
    print("im2var:", im2var)

    A = scipy.sparse.lil_matrix((imh * imw , imh * imw))
    mask_xi, mask_yi, _ = np.where(mask == 1) # get the x,y dimension index out of the mask

    res = np.zeros((imh, imw, c))

    # solving Ax = b_c for each channel
    for channel in range(c):
        b = np.zeros((imh * imw, 1))
        for i in range(len(mask_xi)):
            x, y = mask_xi[i], mask_yi[i]
            e = (x - 1) * imw + y
            point = im2var[x, y]

            A[e, point] = 4
            b[e] += 4 * fg[x,y,channel]

            # top neighbor
            if x - 1 >= 0:
                x_next, y_next = x - 1, y
                b[e] -= 1 * fg[x_next, y_next, channel]
                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                    b[e] += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b
            
            # bottom neighbor
            if x + 1 < imh:
                x_next, y_next = x + 1, y
                b[e] -= 1 * fg[x_next, y_next, channel]
                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                    b[e] += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b
            
            # left neighbor
            if y - 1 >= 0:
                x_next, y_next = x, y - 1
                b[e] -= 1 * fg[x_next, y_next, channel]
                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                    b[e] += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b

            # right neighbor
            if y + 1 < imw:
                x_next, y_next = x, y + 1
                b[e] -= 1 * fg[x_next, y_next, channel]
                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                    b[e] += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b

    
        sA = A.tocsr()
        x = scipy.sparse.linalg.lsqr(sA, b)
        x = x[0] * 255
        # var2im
        res[:,:, channel] = x.reshape((imh, imw)).astype(int)

    return  (res / 255) * mask + bg * (1 - mask)

def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    imh, imw, c = fg.shape
    # im2var(x,y) = x*imh + y
    im2var = img2var_f(fg)
    print("im2var:", im2var)

    A = scipy.sparse.lil_matrix((imh * imw , imh * imw))
    mask_xi, mask_yi, _ = np.where(mask == 1) # get the x,y dimension index out of the mask

    res = np.zeros((imh, imw, c))

    # solving Ax = b_c for each channel
    for channel in range(c):
        b = np.zeros((imh * imw, 1))
        for i in range(len(mask_xi)):
            x, y = mask_xi[i], mask_yi[i]
            e = (x - 1) * imw + y
            point = im2var[x, y]

            A[e, point] = 4

            # top neighbor
            if x - 1 >= 0:
                x_next, y_next = x - 1, y

                ds = fg[x, y, channel] - fg[x_next, y_next, channel]
                dt = bg[x, y, channel] - bg[x_next, y_next, channel]

                if abs(ds) >= abs(dt):
                    b[e] += 1 * ds
                else:
                    b[e] += 1 * dt
                
                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                   b[e]  += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b
            else:
                b[e] += 1 * bg[x, y, channel] # if the point is at boarder, add the background value to the b
            
            # bottom neighbor
            if x + 1 < imh:
                x_next, y_next = x + 1, y
                
                ds = fg[x, y, channel] - fg[x_next, y_next, channel]
                dt = bg[x, y, channel] - bg[x_next, y_next, channel]
                if abs(ds) >= abs(dt):
                    b[e] += 1 * ds
                else:
                    b[e] += 1 * dt
                
                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                    b[e] += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b
            else:
                b[e] += 1 * bg[x, y, channel] # if the point is at boarder, add the background value to the b

            # left neighbor
            if y - 1 >= 0:
                x_next, y_next = x, y - 1
                ds = fg[x, y, channel] - fg[x_next, y_next, channel]
                dt = bg[x, y, channel] - bg[x_next, y_next, channel]

                if abs(ds) >= abs(dt):
                    b[e] += 1 * ds
                else:
                    b[e] += 1 * dt

                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                    b[e] += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b
            else:
                b[e] += 1 * bg[x, y, channel] # if the point is at boarder, add the background value to the b
            # right neighbor
            if y + 1 < imw:
                x_next, y_next = x, y + 1
                ds = fg[x, y, channel] - fg[x_next, y_next, channel]
                dt = bg[x, y, channel] - bg[x_next, y_next, channel]
                if abs(ds) >= abs(dt):
                    b[e] += 1 * ds
                else:
                    b[e] += 1 * dt
                if mask[x_next, y_next]:
                    point_next = im2var[x_next, y_next] # get the index of the point
                    if channel == 0:
                        A[e, point_next] = -1
                else:
                    b[e] += 1 * bg[x_next, y_next, channel] # if the neighbor is not in the mask, add the background value to the b
            else:
                b[e] += 1 * bg[x, y, channel] # if the point is at boarder, add the background value to the b
    
        sA = A.tocsr()
        x = scipy.sparse.linalg.lsqr(sA, b)
        x = x[0] * 255
        # var2im
        res[:,:, channel] = x.reshape((imh, imw)).astype(int)

    return  (res / 255) * mask + bg * (1 - mask)

def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.savefig('results/toy_result.png')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.savefig("results/blend.png")
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')

        # save the plot
        plt.savefig('results/mixed_blend.png')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
