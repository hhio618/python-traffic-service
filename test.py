from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from colormath.color_objects import XYZColor, sRGBColor,LabColor
from colormath.color_conversions import convert_color
import os
import math

# 4 first color are traffic related (from fastest to slowest) + gray bg + water color + landscape(man made) + landscape(nature) + road color + road margin
colors = ["84ca50", "f07d02", "e60000", "9e1313", "ededee", "aadaff" , "c0ecae", "c3ecb1", "fff0ac", "f7d36e"]

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def RGB2HEX(color):
   return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def LAB2HEX(color):
    # Scale up to 0-255 values.
    rgb_r = int(math.floor(0.5 + color[0] * 255))
    rgb_g = int(math.floor(0.5 + color[1] * 255))
    rgb_b = int(math.floor(0.5 + color[2] * 255))
    return '#%02x%02x%02x' % (rgb_r, rgb_g, rgb_b)

def HEX2RGB(h):
   return list(int(h[i:i+2], 16) for i in (0, 2, 4))

def HEX2LAB(h):
   color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
   return rgb2lab(np.uint8(np.asarray([[color]])))[0][0]
    

def closest_node(X, C):
   counts = {c:0 for c in range(C.shape[0])}
   y = []
   n = X.shape[0]
   for i in range(n):
      lowest_dist = 1000000
      best_c = 0
      for j in range(C.shape[0]):
         dist = np.linalg.norm(X[i]-C[j])
         if dist < lowest_dist:
            best_c = j
            lowest_dist = dist
      # print(lowest_dist)
      counts[best_c] = counts[best_c] +1
      y.append(best_c)
      if i%(n/10) == 0:
         print("%d percent is done" %  (i*100/n))
   return counts, np.asarray(y)


def get_colors(image, number_of_colors, show_chart, image_name):
    # modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    C = np.asarray([HEX2RGB(h) for h in colors])
    modified_image = image
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    X = np.asarray(modified_image)

   #  clf = KMeans(n_clusters = number_of_colors, max_iter=1, init=C)
   #  labels = clf.fit_predict(modified_image)
    counts, label = closest_node(X, C)
    

    center_colors = C
    # We get ordered colors by iterating through the keys
    hex_colors = ["#%s"%c for c in colors]
    rgb_colors = [HEX2RGB(c) for c in colors]

    if (show_chart):
        fig = plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        fig.savefig(image_name)

    return rgb_colors


def test():
   color1 = [20,20,200]
   color2 = [20,20,100]

   c1 = rgb2lab(np.uint8(np.asarray([[color1]])))[0][0]
   c2 = rgb2lab(np.uint8(np.asarray([[color2]])))[0][0]
   print(deltaE_cie76(c1,c2))


# im = get_image("map.png")
# plt.imshow(im)
# plt.show()
# for i in range(1,6):
#     get_colors(get_image("map%d.png"%i), 10, True, "pie%d.png" %i)
i = 6
get_colors(get_image("map%d.png"%i), 10, True, "pie%d.png" %i)
# test()