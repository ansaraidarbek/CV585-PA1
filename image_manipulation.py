#credit: Juan Carlos Niebles and Ranjay Krishna

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    # Use skimage io.imread
    out = io.imread(image_path)

    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    image = image/255
    out =  0.5*np.multiply(image,image)


    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width)
    """
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = np.dot(image[...,:3], [0.299, 0.587, 0.144])


    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def rgb_decomposition(image, channel):
    """ Return image with the rgb channel specified
    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: string specifying the channel
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = image
    if(channel == 'R'):
        out = np.multiply(image[...,:3], [1, 0, 0])
    if(channel == 'G'):
        out = np.multiply(image[...,:3], [0, 1, 0])
    if(channel == 'B'):
        out =  np.multiply(image[...,:3], [0, 0, 1])


    ######################################
    #        END OF YOUR CODE            #
    ######################################
    
    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding 
    the specified channels for each image
    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: string specifying channel used for image1
        channel2: string specifying channel used for image2
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    width_cutoff = image1.shape[1] // 2
    a = image1[:, :width_cutoff:]
    b = image2[:, :width_cutoff:]
    a = rgb_decomposition(a, channel1)
    b = rgb_decomposition(b, channel2)
    out = np.concatenate((a, b), axis=1)


    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out