
import os
import io
from scipy import ndimage

import numpy as np
def relabel(labels):
    '''' '''

    if not isinstance(labels, np.ndarray):
        raise TypeError('Input image has to be numpy.ndarray!')

    if labels.ndim != 3:  # only for 3D images

        raise TypeError('Only 3D image is supported!')

    current_label = np.max(labels) + 1

    props = measure.regionprops(labels)

    for obj in props:

        # cropping

        sub_image = np.uint8(
            labels[obj.bbox[0]:obj.bbox[3], obj.bbox[1]:obj.bbox[4], obj.bbox[2]:obj.bbox[5]] == obj.label)

        sub_labels, num_sub_obj = ndimage.label(sub_image, structure=ndimage.generate_binary_structure(3, 3))

        if num_sub_obj > 1:

            sub_labels[sub_labels == 1] = 0

            for i in range(2, num_sub_obj + 1):
                sub_labels[sub_labels == i] = current_label - obj.label

                current_label += 1

            labels[obj.bbox[0]:obj.bbox[3], obj.bbox[1]:obj.bbox[4], obj.bbox[2]:obj.bbox[5]] += sub_labels

        del sub_image

    del props

    return labels, (current_label-1)

