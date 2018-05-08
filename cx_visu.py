"""
cx_visu.py

Visualization methods for counterexample task
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import numpy as np

from torchvision.transforms import ToTensor, CenterCrop, Compose, Pad
import torchvision.utils as tvu

from PIL import Image


def viz_knns(datadir, img_name, knn_names, comp_name, question, answer,
             n_display, outfile='viz_knns.jpg'):
    """
    Outputs image with original image (img_name) on left and tiled
    KNNs on right. KNN images appear in row-major order and can be passed in
    order by score or by KNN-distance. comp_name is the name of the complement
    image. question and answer are strings printed in the image. The complement
    is bordered in red in the tiling.
    """

    def show(img1, img2):
        fig = plt.figure(figsize=(20, 10))
        fig.text(.5, .140, "Q: " + question + "\nA: " +
                 answer, ha='center', fontsize=24)
        gridspec.GridSpec(1, 4)
        plt.subplot2grid((1, 4), (0, 0))
        npimg = img1.numpy()
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

        plt.subplot2grid((1, 4), (0, 1), colspan=3, rowspan=1)
        npimg = img2.numpy()
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
#         plt.show()
        fig.tight_layout()
        plt.savefig(outfile)
        plt.close(fig)
    names = knn_names[:n_display]
    if comp_name not in names:
        names[-1] = comp_name
    comp_index = names.index(comp_name)
    imgs = [Image.open(os.path.join(datadir, name))
            for name in names]
    img_tensors = []
    for i, img in enumerate(imgs):
        if i == comp_index:
            border_width = 10
            transform = Compose([
                CenterCrop(360 - 2 * border_width),
                Pad(border_width, (0, 256, 0)),
                ToTensor(),
            ])
        else:
            transform = Compose([
                CenterCrop(360),
                ToTensor(),
            ])

        img_tensors.append(transform(img))

    orig = transform(Image.open(os.path.join(
        datadir, img_name)))
    img_tensors = [it for it in img_tensors if it.size() == (3,360,360)]

    show(orig, tvu.make_grid(img_tensors))


def viz_qa(datadir, img_name, knn_names, comp_name, question, answer,
           nn_answers, n_display, outfile='viz_qa.jpg'):
    def show(img1, img2, size1, size2):
        fig = plt.figure(figsize=(20, 10))
        fig.text(.03, .180, "Q: " + question + "\nA: " +
                 answer, ha='left', fontsize=20)
        gridspec.GridSpec(1, 4)
        plt.subplot2grid((1, 4), (0, 0))
        npimg = img1.numpy()
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

        plt.subplot2grid((1, 4), (0, 1), colspan=3, rowspan=1)
        for i in range(len(nn_answers)):
            fig.text(.275 + i * .145, .3, "A: " +
                     nn_answers[i], ha='left', fontsize=20)
        npimg = img2.numpy()
        plt.axis('off')
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
#         plt.show()
        fig.tight_layout()
        plt.savefig(outfile)
        plt.close(fig)
    names = knn_names[:n_display]
    if comp_name not in names:
        names[-1] = comp_name

    comp_index = names.index(comp_name)
    imgs = [Image.open(os.path.join(datadir, name)) for name in names]
    img_tensors = []
    for i, img in enumerate(imgs):
        if i == comp_index:
            border_width = 10
            transform = Compose([
                CenterCrop(360 - 2 * border_width),
                Pad(border_width, (0, 256, 0)),
                ToTensor(),
            ])
        else:
            transform = Compose([
                CenterCrop(360),
                ToTensor(),
            ])

        img_tensors.append(transform(img))

    orig = transform(Image.open(os.path.join(datadir, img_name)))
    img_tensors = [it for it in img_tensors if it.size() == (3,360,360)]
    show(orig, tvu.make_grid(img_tensors), (5, 5), (20, 10))
