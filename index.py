import cv2
import os
import pandas as pd
from color_descriptor import ColorDescriptor


directory_path = '/Users/Ziyu//OneDrive - Clarivate Analytics/Desktop/maxmara_images/'
# initialize the color descriptor
cd = ColorDescriptor((8, 8, 8))

with open('/Users/Ziyu//OneDrive - Clarivate Analytics/Desktop/maxmara_images/index.csv', 'w') as f:
    for file_name in os.listdir(directory_path):
        index = file_name.replace('.jpg', '')
        file_path = directory_path + file_name
        image = cv2.imread(file_path, 1)
        features = cd.describe(image)
        print(file_name)
        f.write('%s,%s\n' % (index, ','.join(map(str, features))))