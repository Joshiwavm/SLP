from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class SLP():
    """
    A class that manages cleaned data cubes and estiamtes the spectral line profile of that cube given a mask to sum over
    Input:
        cube (channel, image, image): Cleaned data in units of Jy/pixel
        mask (image, image): A mask that matches the size of the image. The mask is used to some over the pixels.
    Output (with function execute):
        slp (len(channels)): Intensity, units of Jy.
        bootstrap_std: bootstrapped std in units of Jy. 
    
    """
    
    def __init__(self, cube, mask, amount = 200, visualize = False):
        self.visualize = visualize
        self.cube   = cube
        self.mask   = mask
        self.amount = amount
        
        xx, yy         = np.meshgrid(np.arange(0,self.cube.shape[-2], 1), np.arange(0, self.cube.shape[-1],1))
        self.CoM_mask  =  (np.mean(xx[self.mask]), np.mean(yy[self.mask]))
        self.rr        = ((xx-self.CoM_mask[0])**2 + (yy-self.CoM_mask[1])**2)**0.5
        
        self.mask_size = int(np.sum(self.mask)**0.5+0.1*self.cube.shape[-2]) # --> quick fix

    def _position(self):

        r =  np.sqrt(np.random.uniform(0.1,1, size = self.amount))
        theta = np.random.uniform(0,1, size = self.amount) * 2 * np.pi

        x = self.CoM_mask[0] + self.cube.shape[1]/3*r * np.cos(theta) #hard coded
        y = self.CoM_mask[1] + self.cube.shape[1]/3*r * np.sin(theta) #hard coded
        
        pos = np.array([x,y], dtype = np.int).T
        return np.vstack(([int(self.CoM_mask[0]), int(self.CoM_mask[1])], pos))
    
    def _make_mask(self, pos):
        new_masks = np.zeros((len(pos[1:]),self.im.shape[-2], self.im.shape[-1]))
        for idx, xy in enumerate(pos[1:]):
            new_masks[idx, 
                      xy[0]-self.mask_size//2:xy[0]+self.mask_size//2, 
                      xy[1]-self.mask_size//2:xy[1]+self.mask_size//2] = self.mask[pos[0][0]-self.mask_size//2:pos[0][0]+self.mask_size//2, 
                                                                                   pos[0][1]-self.mask_size//2:pos[0][1]+self.mask_size//2]
            
            
        return new_masks.astype(bool)    
    
    def _estimate_std(self, new_masks):
        estimates = []
        for m in new_masks:
            estimates.append(np.sum(self.im * m))
        return np.std(estimates)
    
    def _estimate_slp(self):
        return np.nansum(self.cube*self.mask, axis = (1,2))

    def _visualize(self, masks):
        
        for idx, m in enumerate(masks):
            clear_output(True)
            plt.imshow(m)
            plt.show()
            if idx>30: break
    
    def execute(self):

        bootstrap_std = []
        for i in tqdm(range(len(self.cube))):
            self.im = self.cube[i]
            pos     = self._position()
            masks   = self._make_mask(pos)            
            bootstrap_std.append(self._estimate_std(masks))
            
        slp = self._estimate_slp()

        if self.visualize: self._visualize(masks)

        return slp, np.array(bootstrap_std)
