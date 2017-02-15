########################################################################
#
# File:   project2_cv.py
# Author: Tom Wilmots and John Larkin
# Date:   February 15th, 2017
#
# Written for ENGR 27 - Computer Vision
########################################################################

'''
OVERVIEW:

In this project we are going to investigate two different applications: Laplacian Pyramid Blending and
Hybrid Images. Can use blending to make smooth transistions between two very different iamges, and 
hyprid images allow us to creat intersting optical illusions. 

'''

import cv2
import numpy as np
import sys
import cvk2

def pyr_building(img):
	image = cv2.imread('img')
	n = int(raw_input('Number of levels?: '))



if __name__ == "__main__":
	pyr_building('sunset.png')