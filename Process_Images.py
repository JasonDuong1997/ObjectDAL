import os
import cv2

directory_to_images = ""
dir_list = os.listdir(directory_to_images)

i = 0
for item in dir_list:
  if (".jpg" in item):
    img = cv2.imread(item)

    resize = cv2.resize(img, (0,0), fx=.25, fy=.25)
    resize_name = "img" + str(i) + ".jpg"
    cv2.imwrite(resize_name, resize)
    i += 1

    h_flip = cv2.flip(resize, 1)
    h_flip_name = "img" + str(i) + ".jpg"
    cv2.imwrite(h_flip_name, h_flip)
    i += 1
