# SLP
    """
    A class that manages cleaned data cubes and estiamtes the spectral line profile of that cube given a mask to sum over
    Input:
        cube (channel, image, image): Cleaned data in units of Jy/pixel
        mask (image, image): A mask that matches the size of the image. The mask is used to some over the pixels.
    Output (with function execute):
        slp (len(channels)): Intensity, units of Jy.
        bootstrap_std: bootstrapped std in units of Jy. 
    
    """
