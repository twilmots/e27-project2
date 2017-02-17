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

window = 'Window'
def fixKeyCode(code):
	# need this to fix our opencv bug
    return np.uint8(code).view(np.int8)

def labelAndWaitForKey(image, text):

    # Get the image height - the first element of its shape tuple.
    h = image.shape[0]

    display = image.copy()


    text_pos = (16, h-16)                # (x, y) location of text
    font_face = cv2.FONT_HERSHEY_SIMPLEX # identifier of font to use
    font_size = 1.0                      # scale factor for text
    
    bg_color = (0, 0, 0)       # RGB color for black
    bg_size = 3                # background is bigger than foreground
    
    fg_color = (255, 255, 255) # RGB color for white
    fg_size = 1                # foreground is smaller than background

    line_style = cv2.LINE_AA   # make nice anti-aliased text

    # Draw background text (outline)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                bg_color, bg_size, line_style)

    # Draw foreground text (middle)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                fg_color, fg_size, line_style)

    cv2.imshow('Image', display)

    # We could just call cv2.waitKey() instead of using a while loop
    # here, however, on some platforms, cv2.waitKey() doesn't let
    # Ctrl+C interrupt programs. This is a workaround.
    while fixKeyCode(cv2.waitKey(15)) < 0: pass

def pyr_building(name):
	"""
	This function generates a LaPlacian pyramid.
	A LaPlacian pyramid is one that encodes an image of successfully smaller LaPlacian images, built atop a base layer
	consisting of a blurred and reduced copy of the original image.
	"""

	# result list of laplacian images
	lp = []

	# read in our original image
	image = cv2.imread(name)

	# query the user for number of levels
	# NOTE: it might be beneficial to just have this be a set calculated value
	n = int(raw_input('Number of levels?: '))

	# Initially call our original image g0
	# we'll create the iterations following
	g0 = image.astype(np.float32)

	# doing this for naming convention and to make sure we're not destorying anything
	g_prev = g0.copy()

	# just iterating and creating our laplacian images
	for i in range(n+1):
		# use our pyrDown function
		g_next = cv2.pyrDown(g_prev)

		# recalculate the height and width
		height, width = g_prev.shape[0:2]
		g_next_up = cv2.pyrUp(g_next, dstsize = (width, height))
		
		# if we're less than n
		if i < n:
			# we need to subtract off g_i - g_i+1^+
			li = g_prev - g_next_up
		else: 
			# otherwise, we can just use g_n (l_n = g_n)
			li = g_next
		
		# let's make sure we go onto the next image
		g_prev = g_next

		# let's store our laplacian pyramid
		lp.append(li)
	return lp


def show_image_32bit(img):
	"""
	This function is going to show an image that is passed in as a float32 data type image.
	"""
	cv2.imshow(window, 0.5 + 0.5*(img / np.abs(img).max()))
	cv2.waitKey()



if __name__ == "__main__":
	filename = raw_input("What file would you like to use? (0 for def): ")
	if filename == '0':
		fname = 'sunset.png'
	print(fname)
	# image = cv2.imread(fname)
	# cv2.imshow('Window', image)
	# cv2.waitKey()
	lp_images = pyr_building(fname)
	for image in lp_images:
		show_image_32bit(image)


