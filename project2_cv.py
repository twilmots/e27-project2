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
	This will also reduce until the coarsest level be about 8-16 pixels in its minimum dimension.
	"""

	# result list of laplacian images
	lp = []

	# read in our original image
	image = cv2.imread(name)

	# query the user for number of levels
	# NOTE: it might be beneficial to just have this be a set calculated value
	#n = int(raw_input('Number of levels?: '))

	# Initially call our original image g0
	# we'll create the iterations following
	g0 = image.astype(np.float32)

	# doing this for naming convention and to make sure we're not destorying anything
	g_prev = g0.copy()

	number_of_loops_needed = 1
	temp_height, temp_width = g_prev.shape[0:2]
	while min(temp_height, temp_width) > 16:
		temp_height /= 2
		temp_width  /= 2
		number_of_loops_needed += 1

	print("This is how many times we're going to iterate: {}".format(number_of_loops_needed))

	# just iterating and creating our laplacian images
	for i in range(number_of_loops_needed+1):
		# use our pyrDown function
		g_next = cv2.pyrDown(g_prev)

		# recalculate the height and width
		height, width = g_prev.shape[0:2]
		g_next_up = cv2.pyrUp(g_next, dstsize = (width, height))
		
		# if we're less than n
		if i < number_of_loops_needed:
			# we need to subtract off g_i - g_i+1^+
			li = g_prev - g_next_up
		else: 
			# otherwise, we can just use g_n (l_n = g_n)
			li = g_prev
		
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

def pyr_reconstruct(lp):
	"""
	This function takes in the LaPlacian pyramid list.
	"""
	final_lp = lp[-1]
	r_next = final_lp.copy()
	for i in reversed(range(1,len(lp))):
		# this is changing from n-1 to 1
		l_prev = lp[i-1]

		# Let's get the ideal height. It should be double the size of 
		# the original
		height, width = l_prev.shape[0:2]

		# This is essentially r_i+1^+
		r_next_up = cv2.pyrUp(r_next, dstsize = (width, height))

		# We're just inverting the option above
		r_prev = r_next_up + l_prev

		# We want to keep reversing until we're at the beginning 
		r_next = r_prev

	r0 = r_prev.copy()
	return r0

def alpha_blend(A,B,alpha):
	A = A.astype(alpha.dtype)
	B = B.astype(alpha.dtype)

	# if A and B are RGB images, we must pad out alpha to be the right shape
	if len(A.shape) == 3:
		alpha = np.expand_dims(alpha,2)
	return A + alpha*(B-A)

def traditional_blend(imname1 = 'sunset.png', imname2 = 'minority-report.png'):
	imageA = cv2.imread(imname1)
	imageB = cv2.imread(imname2)

	height, width = imageA.shape[0:2]
	mask = np.zeros((height, width), np.uint8)
	center = (width/2, height/2)       # point specified as (x, y)
	ellipse_size = (width/4, height/2) # size specified as (width, height)
	rotation = 0                      # rotation angle, degrees
	start_angle = 0
	end_angle = 360
	white = (255, 255, 255)            # RGB triple for pure white
	line_style = -1                    # denotes filled ellipse

	cv2.ellipse(mask, center,
            ellipse_size, rotation,
            start_angle, end_angle,
            white, line_style)

	return mask


def image_blend(imname1 = 'sunset.png', imname2 = 'minority-report.png'):
	imageA = cv2.imread(imname1)
	imageB = cv2.imread(imname2)

	height, width = imageA.shape[0:2]
	mask = np.zeros((height, width), np.uint8)
	center = (width/2, height/2)       # point specified as (x, y)
	ellipse_size = (width/4, height/2) # size specified as (width, height)
	rotation = 0                      # rotation angle, degrees
	start_angle = 0
	end_angle = 360
	white = (255, 255, 255)            # RGB triple for pure white
	line_style = -1                    # denotes filled ellipse

	cv2.ellipse(mask, center,
            ellipse_size, rotation,
            start_angle, end_angle,
            white, line_style)

	# Make a Kernel before running the blur
	kernel_size = (5,5)

	# NOTE: RIGHT NOW THIS IS JUST BLURRING OUR MASK...
	# WE NEED TO DO THIS FOR LIKE EACH IMAGE IN OUR LP PYRAMID
	alphamask = cv2.GaussianBlur(mask, kernel_size, 0)

	# labelAndWaitForKey(alphamask, 'Our Alpha-Mask')
	
	lpA = pyr_building(imname1)
	lpB = pyr_building(imname2)

	for i in range(len(lpA)):
		layerA = lpA[i]
		layerB = lpB[i]
		size = layerA.shape[0:2]

		cv2.resize(alphamask, size, interpolation=cv2.INTER_AREA)
		cv2.imshow('win',alphamask)
		cv2.waitKey()
	x = alpha_blend(imageA,imageB,mask)

	labelAndWaitForKey(x,'Our blended image')


if __name__ == "__main__":
	filename = raw_input("What file would you like to use? (0 for def): ")
	if filename == '0':
		fname = 'sunset.png'
	print(fname)

	# Laplacian image pyramid list
	lp_images = pyr_building(fname)

	# How many images are in our laplacian pyramid list
	print("Number of image in our Laplacian pyramid: {}".format(len(lp_images)))

	# Just show all of them for convenience sake
	for image in lp_images:
		show_image_32bit(image)

	r0 = pyr_reconstruct(lp_images)

	# This is our image reconstructed... still in 32 bit
	show_image_32bit(r0)

	# Let's get a shitty blend before using the pyramid
	#rough_blend = traditional_blend()
	#labelAndWaitForKey(rough_blend, 'Rough Blend')
	# Let's blend some images. This is the alpha blending.
	image_blend()







