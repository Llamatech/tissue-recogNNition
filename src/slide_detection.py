import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

slice_dim = (400, 400)

lower_purple = np.array([[[156,  50, 50]]])
upper_purple = np.array([[[176,  255, 255]]])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

def rgb2gray(rgb):
    r, g, b = np.rollaxis(rgb[...,:3], axis = -1)
    return 0.299 * r + 0.587 * g + 0.114 * b

def rgba2rgb(rgba):
    r, g, b, a = np.rollaxis(rgba[...,:], axis = -1)
    return np.dstack((r,g,b))

def map_px(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def flatten_matrix(M):
    M = M.flatten(1)
    M = M.reshape((len(M), 1))
    return M

def flatten_cube(M):
    c = list(np.rollaxis(M[...,:3], axis = -1))
    c = map(flatten_matrix, c)
    return np.vstack(c)

slide = openslide.open_slide('../data/Normal_006.tif')
dims = slide.dimensions

lvl_0 = np.array(slide.read_region((0,0), 5, slide.level_dimensions[5]))
hsv = cv2.cvtColor(lvl_0[:, :, 0:3], cv2.COLOR_RGB2HSV)

mask = cv2.inRange(hsv, lower_purple, upper_purple)
mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
mask1 = cv2.dilate(mask1,kernel,iterations = 1)

mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

mask = (mask1+mask2)/2
mask1, mask2 = None, None
contours, hier = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#C = []
M = None

for i,cnt in enumerate(contours):
    if hier[0, i][-1] == -1:
       area = cv2.contourArea(cnt)
       equi_diameter = np.sqrt(4*area/np.pi)
       if equi_diameter >= 100:
          x,y,w,h = cv2.boundingRect(cnt)
          x_a = map_px(x, 0, lvl_0.shape[1], 0, dims[0])
          y_a = map_px(y, 0, lvl_0.shape[0], 0, dims[1])
          w_m = map_px(w, 0, lvl_0.shape[1], 0, dims[0])
          h_m = map_px(h, 0, lvl_0.shape[0], 0, dims[1])
          count = 1
          for k in range(0, h_m, 400):
              for j in range(0, w_m, 400):
                  print "Tumor %d - Slice: %d/%d" % (i, count, h_m/slice_dim[0] * w_m/slice_dim[1])
                  patch = np.array(slide.read_region((x_a+j, y_a+k), 0, slice_dim))
                  gray = rgb2gray(patch)
                  hist = cv2.calcHist([np.float32(gray)],[0],None,[256],[0,256])
                  if np.sum(hist[0:200]) > np.sum(hist[200:]):
                     mpimg.imsave('./Normal_006/T%d_%d.png' % (i, count), patch)
                     count += 1

#np.savez('features.npz', )




