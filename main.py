#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp
import os

from PIL import Image

import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from skimage import color


import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

TARGET_CLASSES = {'acousticGuitar.jpg': 402, 
                  'artichoke.jpg': 944, 
                  'snake.jpg': 56, 
                  'bee.jpg': 309, 
                  'bellPepper.jpg': 945, 
                  'bicycle.jpg': 444, 
                  'broccoli.jpg': 937, 
                  'camera.jpg': 759, 
                  'canoe.jpg': 472, 
                  'car.jpg': 705, 
                  'castle.jpg': 483, 
                  'cat.jpg': 281, 
                  'corn.jpg': 987, 
                  'cucumber.jpg': 943, 
                  'dog.jpg': 267, 
                  'dogs.jpg': 199, 
                  'drum.jpg': 822, 
                  'electricFan.jpg': 545, 
                  'fig.jpg': 952, 
                  'flute.jpg': 558, 
                  'frog.jpg': 31, 
                  'harmonica.jpg': 593, 
                  'jellyfish.jpg': 107,
                  'keyboard.jpg': 508, 
                  'kingSnake.jpg': 56, 
                  'laptop.jpg': 620, 
                  'mudturtle.jpg': 35, 
                  'orange.jpg': 950, 
                  'parrot.jpg': 88, 
                  'piano.jpg': 579, 
                  'pineapple.jpg': 953, 
                  'pomegranate.jpg': 957, 
                  'seaAnemone.jpg': 108, 
                  'soccerball.jpg': 805, 
                  'sock.jpg': 806, 
                  'spiderMonkey.jpg': 381, 
                  'starfish.jpg': 327, 
                  'strawberry.jpg': 949, 
                  'tiger.jpg': 292, 
                  'toaster.jpg': 859, 
                  'tractor.jpg': 866, 
                  'train.jpg': 705, 
                  'trumpet.jpg': 513, 
                  'turtle.jpg': 33, 
                  'violin.jpg': 889, 
                  'warplane.jpg': 895, 
                  'mudturtle.jpg' : 35
                #   'acorn.jpg':988,
                #   'asparagus.jpg':
                #   'lettuce.jpg':
                #   'raspberry.jpg':
                #   'sunflower.jpg': 
            }

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def load_images_from_folder(folder_path):
    images = []
    raw_images = []
    target_classes = []
    names = []
    print("Images:")
    imgs = os.listdir(folder_path)

    for i, image_path in enumerate(imgs):
        image_name = image_path
        if (image_name in TARGET_CLASSES):
            names.append(image_name.split('.')[0])
            image_path = folder_path + image_path
            print("\t#{}: {}".format(i, image_path))
            image, raw_image = preprocess(image_path)
            images.append(image)
            raw_images.append(raw_image)
            target_classes.append([TARGET_CLASSES[image_name]])

    return images, raw_images, target_classes, names


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filepath, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    maskpos = np.zeros(gcam.shape)
    maskneg = np.zeros(gcam.shape)

    top_five = []
    bottom_five = []


    # mask[mask >= 0.8] = 0
    # mask[(mask >= 0.6) & (mask < 0.8) ] = 1 
    # mask[(mask >= 0.4) & (mask < 0.6) ] = 2
    # mask[(mask >= 0.2) & (mask < 0.4) ] = 3
    # mask[(mask >= 0.10) & (mask < 0.2) ] = 4

    # mask[(mask >= 0.05) & (mask < 0.10) ] = 5
    # mask[(mask >= 0.010) & (mask < 0.05) ] = 6

    # mask[mask < 0.010 ] = 7
    alpha = np.full((gcam.shape[0], gcam.shape[1]), 255) 

    print(raw_image.shape)
    raw_image = np.dstack((raw_image, alpha))
    print(raw_image.shape)

    for i in range(0, 5):
        upper = (i * 5000)
        lower = (i+1) * 5000
        # print(-lower, -upper)
        if (upper == 0):
            ii = np.unravel_index(np.argsort(gcam.ravel())[-lower:], gcam.shape)
        else:
            ii = np.unravel_index(np.argsort(gcam.ravel())[-lower:-upper], gcam.shape)
        
        tempmask = np.ones(gcam.shape).astype(int)
        tempmask[ii] = 0
        tempimg = np.copy(raw_image)
        tempimg[tempmask == 1] = [255,255,255,0]
    
        top_five.append(tempimg)


    for i in range(0, 5):
        upper = (i+1) * 5000
        lower = (i) * 5000
        # print(lower, upper)
        ii = np.unravel_index(np.argsort(gcam.ravel())[lower:upper], gcam.shape)
        maskneg[ii] = i+1
    
        tempmask = np.ones(gcam.shape).astype(int)
        tempmask[ii] = 0
        tempimg = np.copy(raw_image)
        tempimg[tempmask == 1] = [255,255,255,0]
    
        bottom_five.append(tempimg)



    # threshold = np.sort(mask.ravel())[-5000]

    # mask[mask < threshold] = 0
    # mask[mask >= threshold] = 1
    # print(mask)



    # print(mask)
    # temp = mask.reshape(50176)
    
    # plt.hist(temp,  bins=10)
    # plt.savefig('foo.png')

    # cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        # gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        # print(cmap.shape, raw_image.shape)

        # cmap = cmap.reshape(224*224, 3)
        # raw_image = raw_image.reshape(224*224, 3)
        # print(raw_image)
        # new_image = mark_boundaries(raw_image, np.uint8(mask))
        # new_image_pos = color.label2rgb(maskpos, raw_image)
        # new_image_neg = color.label2rgb(maskneg, raw_image)

        # raw_image[mask == 0] = [255, 255, 255] 
        # print(cm.jet())
        # for (i, c) in enumerate(cmap):
        #     if (not (c[2] > 200)): 
        #         raw_image[i] = [0,0,0]
        #         # cmap[i] = [0, 0, 0]
        #     # else: 
        #     #     print(c)
            
        # cmap = cmap.reshape(224, 224, 3)
        # raw_image = raw_image.reshape(224, 224, 3)

        # gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) /2
        # gcam =  raw_image.astype(np.float)
        pass

    filename = os.path.basename(filepath).split(".")[0]
    directory = os.path.dirname(filepath) + "/" + filename
    print(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)



    for i, img in enumerate(bottom_five):
        cv2.imwrite(directory + "/" + "bottom_"+str(i)+ ".png" , img)

    for i, img in enumerate(top_five):
        cv2.imwrite(directory + "/" + "top_"+str(i)+ ".png" , img)

    # print(filename.split("."))




def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-f", "--folder-path", type=str, multiple=False, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-t", "--target-layer", type=str, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo1(folder_path, target_layer, arch, topk, output_dir, cuda):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model.to(device)
    model.eval()

    # Images
    images, raw_images, target_classes, names = load_images_from_folder(folder_path)
    print(target_classes)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================


    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted
    # print("Vanilla Backpropagation:")
    # for i in range(topk):
    #     bp.backward(ids=ids[:, [i]])
    #     gradients = bp.generate()

    #     # Save results as image files
    #     for j in range(len(images)):
    #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

    #         save_gradient(
    #             filename=osp.join(
    #                 output_dir,
    #                 "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
    #             ),
    #             gradient=gradients[j],
    #         )

    # # Remove all the hook function in the "model"
    # bp.remove_hook()

    # # =========================================================================
    # print("Deconvolution:")

    # deconv = Deconvnet(model=model)
    # _ = deconv.forward(images)

    # for i in range(topk):
    #     deconv.backward(ids=ids[:, [i]])
    #     gradients = deconv.generate()

    #     for j in range(len(images)):
    #         print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

    #         save_gradient(
    #             filename=osp.join(
    #                 output_dir,
    #                 "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
    #             ),
    #             gradient=gradients[j],
    #         )

    # deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    ids_ = torch.LongTensor(target_classes).to(device)
    ids = ids_
    # for i in range(topk):
        # Guided Backpropagation
    gbp.backward(ids=ids_)
    gradients = gbp.generate()

    # Grad-CAM
    gcam.backward(ids=ids_)
    regions = gcam.generate(target_layer=target_layer)

    for j in range(len(images)):
        print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, 0]], probs[j, 0]))

        # Guided Backpropagation
        # save_gradient(
        #     filename=osp.join(
        #         output_dir,
        #         "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, 0]]),
        #     ),
        #     gradient=gradients[j],
        # )

        # Grad-CAM
        # save_gradcam(
        #     filename=osp.join(
        #         output_dir,
        #         "{}-{}-gradcam-{}-{}.png".format(
        #             j, arch, target_layer, classes[ids[j, 0]]
        #         ),
        #     ),
        #     gcam=regions[j, 0],
        #     raw_image=raw_images[j],
        # )
        save_gradcam(
            filepath=osp.join(
                output_dir,
                "{}.png".format(
                   names[j]
                ),
            ),
            gcam=regions[j, 0],
            raw_image=raw_images[j],
        )


        # Guided Grad-CAM
        # save_gradient(
        #     filename=osp.join(
        #         output_dir,
        #         "{}-{}-guided_gradcam-{}-{}.png".format(
        #             j, arch, target_layer, classes[ids[j, 0]]
        #         ),
        #     ),
        #     gradient=torch.mul(regions, gradients)[j],
        # )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo2(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-s", "--stride", type=int, default=1)
@click.option("-b", "--n-batches", type=int, default=128)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo3(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    Generate occlusion sensitivity maps
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images, _ = load_images(image_paths)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )


if __name__ == "__main__":
    main()
