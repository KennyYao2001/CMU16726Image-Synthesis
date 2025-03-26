import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss
import torchvision.transforms as T
import time


"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# seed 
torch.manual_seed(3)

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_3']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10']


def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)

    i = 0
    name = ""
    
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
            model.add_module(name, layer)
            
            # add content loss
            if name in content_layers:
                # get the output of the current layer
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            # add style loss
            if name in style_layers:
                # get the output of the current layer
                target = model(style_img).detach()
                style_loss = StyleLoss(target)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            model.add_module(name, nn.ReLU(inplace=False))

        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            model.add_module(name, layer)

        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
            model.add_module(name, layer)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    model = model[:i+1]
    print(model)
    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a "closure"
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)
    model.requires_grad_(False)
    # get the optimizer
    input_img.requires_grad_(True)
    optimizer = get_image_optimizer(input_img)
    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function

    step = [0]

    def closure():
        with torch.no_grad():
            input_img.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)

        style_score = 0
        content_score = 0
        loss = 0

        if use_content:
            for cl in content_losses:
                content_score += cl.loss*content_weight
            loss += content_score

        if use_style:
            for sl in style_losses:
                style_score += sl.loss*style_weight
            loss += style_score

        loss.backward()
        step[0] += 1
        if step[0] % 10 == 0:
            print(f"step {step[0]}: Style Loss: {style_score}, Content Loss: {content_score}, Total Loss: {loss}")

        return loss
    
    while step[0] < num_steps:
        optimizer.step(closure)
            
    


    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step

    # make sure to clamp once you are done
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def main(style_img_path, content_img_path):
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)

    # interative MPL
    plt.ion()

    # resize style image to content image size
    SIZE = (content_img.shape[-2], content_img.shape[-1])
    style_size = style_img.shape

    
    style_img = T.Resize(size=SIZE)(style_img)
    content_img = T.Resize(size=SIZE)(content_img)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # ============================ part1 ============================
    # # image reconstruction
    # print("Performing Image Reconstruction from white noise initialization")
    # # input_img = random noise of the size of content_img on the correct device
    # input_img = torch.randn(content_img.data.size(), device=device)
    # # output = reconstruct the image from the noise
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=False, num_steps=1000)

    # plt.figure()
    # imshow(output, title='Reconstructed Image')
    # plt.savefig("results/reconstructed_image_layer3_seed12.png")

    # ============================ part2 ============================
    # # texture synthesis
    # print("Performing Texture Synthesis from white noise initialization")
    # # input_img = random noise of the size of content_img on the correct device
    # # output = synthesize a texture like style_image
    # input_img = torch.randn(content_img.data.size(), device=device)
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=False, use_style=True, num_steps=1000)

    # plt.figure()
    # imshow(output, title='Synthesized Texture')
    # plt.savefig("results/synthesized_texture_1-15_seed0.png")
    # ============================ part3 ============================
    # style transfer
    # input_img = random noise of the size of content_img on the correct device
    # output = transfer the style from the style_img to the content image

    # print("Performing Style Transfer from random noise initialization")
    # time_start_noise = time.time()
    # input_img = torch.randn(content_img.data.size(), device=device)
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=1000, style_weight=10000, content_weight=1)
    
    # plt.figure()
    # imshow(output, title='Output Image from noise')
    # # adding the name of the style and content image to the path
    # plt.savefig(f"results/style_transfer_noise_{style_img_path.split('/')[-1].split('.')[0]}_{content_img_path.split('/')[-1].split('.')[0]}.png")
    # time_end_noise = time.time()

    print("Performing Style Transfer from content image initialization")
    # input_img = content_img.clone()
    # output = transfer the style from the style_img to the content image
    time_start_content = time.time()

    input_img = content_img.clone()
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=1000, style_weight=10000, content_weight=1)
    plt.figure()
    imshow(output, title='Output Image from content')
    plt.savefig(f"results/style_transfer_content_{style_img_path.split('/')[-1].split('.')[0]}_{content_img_path.split('/')[-1].split('.')[0]}.png")
    time_end_content = time.time()

    # print(f"Time taken: {time_end_noise - time_start_noise} seconds")
    print(f"Time taken: {time_end_content - time_start_content} seconds")

    # ================================================================

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:3]
    main(*args)
