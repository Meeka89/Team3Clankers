#----------conv.py----------#
#-------Andrew_Martin-------#

import numpy as np
from forms import tanh
#part 3 is needed for kernels


def get_patches(layer): # layer: (batch, 28, 28)
    batch_size = layer.shape[0]
    patches = []
    for r in range(layer.shape[1] - 3 + 1): # 0..25
        for c in range(layer.shape[2] - 3 + 1): # 0..25
            patch = layer[:, r:r+3, c:c+3] # (batch, 3, 3)
            # the 1 is a slot for the patch-position axis
            patches.append(patch.reshape(batch_size, 1, 3, 3))
    # concatenate fills that slot: 676 positions side by side
    expanded = np.concatenate(patches, axis=1) # (batch, 676, 3, 3)
    es = expanded.shape
    # flatten EACH patch to 9 numbers, one row per (image, position)
    return expanded.reshape(es[0]*es[1], 9), es # (batch*676, 9)

# convulution forward pass from good ol joe
def conv_forward(layer_0, kernels):

    n = layer_0.shape[0] # images in THIS batch (the last one is short)
    # the SAME 16 kernels multiply every patch, at every position
    
    flat_patches, es = get_patches(layer_0) # (n*676, 9)
    
    kernel_out = flat_patches.dot(kernels) # (n*676, 16)
    # group the responses by image (nothing averaged or discarded)
    
    layer_1 = tanh(kernel_out.reshape(n, 676*16)) # (n, 10816)

    return flat_patches, kernel_out, layer_1



if __name__ == "__main__":
    # Shape check 100 images -> (67600, 9)
    layer = np.random.rand(100, 28, 28)
    flat, es = get_patches(layer)
    print("flat shape:", flat.shape, "| expanded shape:", es)
    # flat shape: (67600, 9) | expanded shape: (100, 676, 3, 3)
 
    # Content check: first patch is image 0's top left 3x3 corner
    print("first patch matches top-left corner:",
          np.allclose(flat[0], layer[0, 0:3, 0:3].flatten())) # true
 
    # Ordering check: row 676 is image 1's top left corner
    print("row 676 is image 1's top-left corner:",
          np.allclose(flat[676], layer[1, 0:3, 0:3].flatten())) # true
 
    # Last patch of image 0 is position (25, 25)
    print("row 675 is image 0 at (25,25):", 
          np.allclose(flat[675], layer[0, 25:28, 25:28].flatten())) # true

    #---------------------------------------------------------------------
    # part 2 test case
    # Forward check: 16 kernels -> layer_1 is (n, 10816), all in (-1, 1)
    kernels = 0.02 * np.random.random((9, 16)) - 0.01
    fp, ko, l1 = conv_forward(np.random.rand(3, 28, 28), kernels)
    print("forward shapes:", fp.shape, ko.shape, l1.shape)  # (2028, 9) (2028, 16) (3, 10816)
    print("layer_1 in (-1, 1):", l1.min() > -1 and l1.max() < 1)  # True

    #AI NOTICE
    #
    # Claude was used to generate some of the print statements above and debugg. 
    # all other code is my own or from the handout. 
