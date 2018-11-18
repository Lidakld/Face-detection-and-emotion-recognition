#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 07:23:05 2018

@author: lida kuang
"""
#%%
import numpy as np
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
    top_left = (tl[0], tl[1])
    bottom_right = (br[0], br[1])
    if(top_left == bottom_right):
        return ii[bottom_right]
    top_right = (tl[0], br[1])
    bottom_left = (br[0], tl[1])
    return (ii[bottom_right]+ii[top_left]-ii[top_right]-ii[bottom_left])

def enum(**enums):
    return type('Enum', (), enums)

FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1), THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]

class HaarLikeFeature(object):
    '''
    create Haar like feature
    '''
    def __init__(self, feature_type, position, width, height):
        '''
        feature_type: see FeatureType
        position: postion of top left(x, y)
        width: width of the feature
        height: height of the feature
        '''
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
#        self.height = height
#        self.polarity = polarity
#        self.threshold = threshold
        
    def get_score(self, ii):
        '''
        Get the score of given feature:
        
        :para ii: integral image array 
        :return score of the given feature
        
        '''
        score = 0
        if self.type == FeatureType.TWO_VERTICALL:
            first =  sum_region(ii, self.top_left, (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            second = sum_region(ii, (int(self.top_left[0] + self.width / 2), self.top_left[1]), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = sum_region(ii, self.top_left, (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            second = sum_region(ii, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = sum_region(ii, self.top_left, (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = sum_region(ii, (self.top_left[0], int(self.top_left[1] + self.height / 3)), (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3) ))
            third = sum_region(ii, (self.top_left[0], self.top_left[1] + 2 * self.height / 3), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = sum_region(ii, self.top_left, (int(self.top_left[0] + self.width / 3), self.bottom_right[1]))
            second = sum_region(ii, (int(self.top_left[0] + self.width / 3), self.top_left[1]), (int(self.top_left[0] + 2 * self.width / 3), self.bottom_right[1]))
            third = sum_region(ii, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            # top left area
            first = sum_region(ii, self.top_left, (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            # top right area
            second = sum_region(ii, (int(self.top_left[0] + self.width / 2), self.top_left[1]), (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            # bottom left area
            third = sum_region(ii, (self.top_left[0], int(self.top_left[1] + self.height / 2)), (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            # bottom right area
            fourth = sum_region(ii, (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)), self.bottom_right)
            score = first - second - third + fourth
        return score
    
#    def get_vote(self, ii):
#        """
#        Get vote of this feature for given integral image.
#        :param int_img: Integral image array
#        :type int_img: numpy.ndarray
#        :return: 1 iff this feature votes positively, otherwise -1
#        :rtype: int
#        """
#        score = self.get_score(ii)
#        return (1 if score < self.polarity * self.threshold else 0)
#%%
#import cv2 
#path = '../data/processed_images/train/argry/0000.jpg'
#img = cv2.imread(path)
#img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',img_g)
##%%
#img_ii = to_integral_image(img_g)
#cv2.imshow('ii',img_ii)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()   
            