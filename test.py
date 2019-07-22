from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from colormath.color_objects import XYZColor, sRGBColor,LabColor
from colormath.color_conversions import convert_color
import os

# 4 first color are traffic related (from fastest to slowest) + gray bg + water color + landscape(man made) + landscape(nature) + road color + road margin
colors = ["84ca50", "f07d02", "e60000", "9e1313", "ededee", "aadaff" , "c0ecae", "c3ecb1", "fff0ac", "f7d36e"]

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image

def RGB2HEX(color):
   return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def HEX2RGB(h):
   return list(int(h[i:i+2], 16) for i in (0, 2, 4))

def HEX2LAB(h):
   color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
   rgb = sRGBColor(rgb_r = color[0], rgb_g =color[1], rgb_b = color[2], is_upscaled=True)
   xyz = convert_color(rgb, XYZColor)
   out = convert_color(xyz,LabColor)
   return [out.lab_l, out.lab_a, out.lab_b]

def closest_node(X, C):
    dist_2 = np.sum((X - C)**2, axis=1)
    return np.argmin(dist_2)

def get_colors(image, number_of_colors, show_chart):
    # modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    C = np.asarray([HEX2LAB(h) for h in colors])
    modified_image = image
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    X = np.asarray(modified_image)

   #  clf = KMeans(n_clusters = number_of_colors, max_iter=1, init=C)
   #  labels = clf.fit_predict(modified_image)
    label = closest_node(X, C)
    print(label[:10])
    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.figure()
        plt.imshow(image)
        plt.show()

    return rgb_colors


# im = get_image("map.png")
# plt.imshow(im)
# plt.show()
get_colors(get_image("map.png"), 10, True)