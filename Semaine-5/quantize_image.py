# -*- coding: utf-8 -*-

import os
from random import sample
from time import clock
# PIL Library
import Image

# You have to implement the k-means algorithm in a function kmeans
# in a file kmeans.py
from kmeans import kmeans

# available on the lecture websit
from colorDist import *

# Parameters
in_filename = "sea.png"
n_samples = 2000  # number of points to consider to learn the quantification
n_colors = 16  # number of colors in the quantified image
use_perceptualColorDistance = False  # Otherwise, use eucleadean distance.
verbose_error = True  # error is printed at each iteration of the k-means if set to True
maxerr = 0.001  # first stopping criteria for k-means
maxiter = 20  # maximum number of iterations in the k-means


def quantize(data, palette_short):
    out_data = []
    for rgb in data:
        closest_col = min(enumerate(palette_short),
                          key=lambda (pos, col): perceptualColorDistance(col, rgb))
        out_data.append(closest_col[0])
    return out_data


if __name__ == "__main__":

    assert 257 > n_colors > 1

    in_name, suffix = os.path.splitext(in_filename)
    if use_perceptualColorDistance:
        ext = "_percept"
    else:
        ext = "eucl"

    out_filename = "_".join([str(in_name),
                             str(n_samples),
                             str(n_colors),
                             ext]) + ".png"

    im = Image.open(in_filename)

    if im.mode != "RGB":
        im = im.convert(mode='RGB')

    data = im.getdata()

    points = sample(data, n_samples)

    # create features vectors
    pointsFP = [tuple(comp / 255.0 for comp in point) for point in points]

    clusters, centroids = kmeans(data=pointsFP,
                                 k=n_colors,
                                 t=maxerr,
                                 maxiter=maxiter)

    # convert results of kmeans into a color palette
    palette = [tuple(
        int(round(coord * 255)) for coord in point) for point in centroids]
    palette = sorted(set(palette), key=sum)

    # ensure the color palette has 255 components
    palette_short = list(palette)
    palette.extend(palette[-1] for i in xrange(256 - len(palette)))

    # create output image
    im_out = Image.new("P", im.size, 0)
    flattened_palette = [component for color in palette for component in color]
    im_out.putpalette(flattened_palette)
    out_data = quantize(data, palette_short)
    im_out.putdata(out_data)
    im_out.save(out_filename)
