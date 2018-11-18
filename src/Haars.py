#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:08:57 2018

@author: lida kuang
"""
#%%
import numpy as np
import cv2 
#%%
#generate features for image
path = '../data/processed_images/train/argry/0000.jpg'
img = cv2.imread(path)
img = cv2.resize(img,(24,24))
img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#%%
def enum(**enums):
    return type('Enum', (), enums)
FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]

#%%
def to_integral_image(img_arr):
    row_sum = np.zeros(img_arr.shape)
    integral_image = np.zeros((img_arr.shape[0],img_arr.shape[1]))
    for i in range(img_arr.shape[1]):
        for j in range(img_arr.shape[0]):
            row_sum[i, j] = row_sum[i, j - 1] + img_arr[i, j]
            integral_image[i, j] = integral_image[i - 1, j] + row_sum[i, j]
    return integral_image

def sum_region(ii, tl, br):
    top_left = tl
    bottom_right = br
    if(top_left == bottom_right):
        return ii[bottom_right]
    top_right = (tl[0], br[1])
    bottom_left = (br[0], tl[1])
#    print(top_left, bottom_right, top_right, bottom_left)
    return (ii[bottom_right] + ii[top_left] - ii[top_right] - ii[bottom_left])

#%%
#count = 0;
def get_feature_properties():
    feature_properties = []
    for i in range(len(FeatureTypes)):
        ft = FeatureTypes[i]
        sizeX = ft[1]
        sizeY = ft[0]
#        print("%dx%d shapes:\n" % (sizeX, sizeY))
        for w in range(sizeX, max_width + 1, sizeX):
            for h in range(sizeY, max_height + 1, sizeY):
    #            print("\tsize: %dx%d => " % (w, h))
    #            c=count
                for x in range(max_width - w ):
                    for y in range(max_height - h ):
                        feature_properties.append((ft, (x, y), (h, w)))
    return feature_properties
#                    count+=1
#            print("count: %d\n" % (count-c))
#                    hf = HaarLikeFeature(ft, (x, y), w, h)
#                    print(hf.get_score(ii))
                    
    
def get_feature_values(ii ,feature_properties):
    feature_values = []
    for i in range(len(feature_properties)):
        feature_type = feature_properties[i][0]
        top_left = feature_properties[i][1]
        height, width = feature_properties[i][2]
        bottom_right = (top_left[0] + width, top_left[1] + height)
    #    print(feature_properties[i])
        score = 0
        if feature_type == (1, 2):
            first =  sum_region(ii, top_left, (int(top_left[0] + width / 2), bottom_right[1]))
            second = sum_region(ii, (int(top_left[0] + width / 2), top_left[1]), bottom_right)
            score = first - second
        elif feature_type == (2, 1):
            first = sum_region(ii, top_left, (bottom_right[0], int(top_left[1] + height / 2)))
            second = sum_region(ii, (top_left[0], int(top_left[1] + height / 2)), bottom_right)
            score = first - second
        elif feature_type == (3, 1):
            first = sum_region(ii, top_left, (bottom_right[0], int(top_left[1] + height / 3)))
            second = sum_region(ii, (top_left[0], int(top_left[1] + height / 3)), (bottom_right[0], int(top_left[1] + 2 * height / 3) ))
            third = sum_region(ii, (top_left[0], int(top_left[1] + 2 * height / 3)), bottom_right)
            score = first - second + third
        elif feature_type == (1, 3):
            first = sum_region(ii, top_left, (int(top_left[0] + width / 3), bottom_right[1]))
            second = sum_region(ii, (int(top_left[0] + width / 3), top_left[1]), (int(top_left[0] + 2 * width / 3), bottom_right[1]))
            third = sum_region(ii, (int(top_left[0] + 2 * width / 3), top_left[1]), bottom_right)
            score = first - second + third
        elif feature_type == (4, 4):
            # top left area
            first = sum_region(ii, top_left, (int(top_left[0] + width / 2), int(top_left[1] + height / 2)))
            # top right area
            second = sum_region(ii, (int(top_left[0] + width / 2), top_left[1]), (bottom_right[0], int(top_left[1] + height / 2)))
            # bottom left area
            third = sum_region(ii, (top_left[0], int(top_left[1] + height / 2)), (int(top_left[0] + width / 2), bottom_right[1]))
            # bottom right area
            fourth = sum_region(ii, (int(top_left[0] + width / 2), int(top_left[1] + height / 2)), bottom_right)
            score = first - second - third + fourth
        feature_values.append(score)
    return feature_values
    
#%%
ii = to_integral_image(img_g)
max_height,max_width = ii.shape
feature_properties = get_feature_properties()
feature_values = get_feature_values(ii, feature_properties)