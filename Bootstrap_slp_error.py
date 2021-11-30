from tqdm import tqdm
import numpy as np

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
    
    def __init__(self, cube, mask, amount = 1000):
        self.cube   = cube
        self.mask   = mask
        self.amount = amount
        
        xx, yy         = np.meshgrid(np.arange(0,self.cube.shape[-2], 1), np.arange(0, self.cube.shape[-1],1))
        self.CoM_mask  =  (np.mean(xx[self.mask]), np.mean(yy[self.mask]))
        self.rr        = ((xx-self.CoM_mask[0])**2 + (yy-self.CoM_mask[1])**2)**0.5
        
        self.mask_size = int(np.sum(self.mask)**0.5+0.1*self.cube.shape[-2]) # --> quick fix

    def _position(self):

        idx  = 0
        seed = 0
        pos  = []
        while idx < self.amount:
            np.random.seed(seed)
            seed +=1 
            
            x, y = np.random.uniform(self.mask_size//2, self.im.shape[-1] - self.mask_size//2, size = 2).astype(np.int) 
            dist = self.rr[x,y]
            
            if dist > 0.1*self.im.shape[-2]: #accept mask --> quick fix
                pos.append([x,y])
                idx +=1
        
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


    def execute(self):

        bootstrap_std = []
        for i in tqdm(range(len(self.cube))):
            self.im = self.cube[i]
            pos     = self._position()
            masks   = self._make_mask(pos)
            bootstrap_std.append(self._estimate_std(masks))
            
        slp = self._estimate_slp()

            
        return slp, np.array(bootstrap_std)