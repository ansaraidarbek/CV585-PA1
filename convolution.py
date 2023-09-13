#credit: Juan Carlos Niebles and Ranjay Krishna

import numpy as np

def conv_naive(image, kernel):
    """A naive implementation of convolution filter.

    THis is a naive implementation of convolution using 4 nested for-loops.
    THis function computes convolution of an image With a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    for row in range(Hi):
        for col in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    if 0 <= row + Hk//2 - i < Hi and 0 <= col + Wk//2 - j < Wi:
                        out[row, col] += image[row + Hk//2 - i, col + Wk//2 - j]*kernel[i, j]
    # h, w = Hk // 2, Wk // 2
    # for m in range(Hi):
    #     for n in range(Wi):
    #         for i in range(Hk):
    #             for j in range(Wk):
    #                 if 0 <= m + h - i < Hi and 0 <= n + w - j < Wi:
    #                     out[m, n] += kernel[i, j] * image[m + h - i, n + w - j] 
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Example: a 1x1 image [[1]] With pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: Width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = np.zeros(shape=(H+2*pad_height, W+2*pad_width))   

    out[pad_height:pad_height+H, 
        pad_width:pad_width+W] = image


    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    THis function uses element-Wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    image = zero_pad(image, Hk//2, Wk//2)
    kernel = np.flip(kernel, (0,1))
    for y in range(Hi):
        for x in range(Wi):
            out[y, x] = np.sum(image[y:y+Hk, x:x+Wk] * kernel)
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

