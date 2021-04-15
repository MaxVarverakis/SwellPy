import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from .particle_system import ParticleSystem


class Monodisperse(ParticleSystem):
    def __init__(self, N, boxsize=None, boxsize_x=None, boxsize_y=None, seed=None):
        """
        Args:
            N (int): The number of particles in the system
            boxsize (float): optional. Length of the sides of the box
            seed (int): optional. Seed for initial particle placement randomiztion
        """
        if  (boxsize_x==None) and (boxsize_y==None):
            boxsize_xi=boxsize
            boxsize_yi=boxsize
            super(Monodisperse, self).__init__(N, boxsize, boxsize_xi, boxsize_yi, seed=None)
        else:
            super(Monodisperse, self).__init__(N, boxsize, boxsize_x, boxsize_y, seed=None)
        self._name = "Monodisperse"
    
    def equiv_swell(self, area_frac):
        """
        Finds the particle diameter that is equivalent to some area fraction.

        Args:
            area_frac (float): the area fraction of interest
        Returns:
            (float): the equivalent diameter
        """
        af = np.array(area_frac, ndmin=1)
        if self.boxsize==None:
            return 2 * np.sqrt(af * self.boxsize_x*self.boxsize_y / (self.N * np.pi))
        else:
            return 2 * np.sqrt(af * self.boxsize**2 / (self.N * np.pi))
        
    def equiv_swell_xform(self, area_frac, scale_x, scale_y):
        """
        Finds the particle diameter that is equivalent to some area fraction, takes in account
        transformation scaling factors.
        ***SHOULD NOT NEED TO BE USED***
        Args:
            area_frac (float): the area fraction of interest
            scale_x (float): scaling in x direction
            scale_y (float): scaling in y direction
        Returns:
            (float): the equivalent diameter
        """
        af = np.array(area_frac, ndmin=1)
        xform_boxsize_x = (self.boxsize_x*scale_x/scale_y)
        xform_boxsize_y = (self.boxsize_y*scale_y/scale_x)
        swell = 2 * np.sqrt(af * xform_boxsize_x*xform_boxsize_y / (self.N * np.pi))
        return swell
    
    def equiv_area_frac(self, swell):
        """
        Finds the area fraction that is equivalent to some some swell diameter.

        Args:
            swell (float): the particle diameter of interest
        Returns:
            (float) the equivalent area fraction
        """
        d = np.array(swell, ndmin=1)
        return (d / 2)**2 * (self.N * np.pi) / self.boxsize**2

    def _tag(self, swell, xform_boxsize_x=None, xform_boxsize_y=None):
        """ 
        Get the center indices of the particles that overlap at a 
        specific swell
        
        Parameters:
            swell (float): diameter length of the particles

        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """

        # Note cKD can retun numpy arrays in query pairs
        # but there is a deallocation bug in the scipy.spatial code
        # converting from a set to an array avoids it
        if (xform_boxsize_x==None) and (xform_boxsize_y==None):
            tree = cKDTree(self.centers, boxsize = self.boxsize)
        else:
            tree = cKDTree(self.centers, boxsize = (xform_boxsize_x, xform_boxsize_y))
        pairs = tree.query_pairs(swell)
        pairs = np.array(list(pairs), dtype=np.int64)
        return pairs
    
    def find_angle(self, pairs):
        """
        Finds the angles of the kicks.
        
        Parameters:
            pairs (np.array):  An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
                
        Returns: 
            theta: list containing all kick angles
        """
        theta = []
        for i in pairs:
            x1 = self.centers[i[0]][0] # x-coordinate of first particle
            x2 = self.centers[i[1]][0] # x-coordinate of second particle
            y1 = self.centers[i[0]][1] # y-coordinate of first particle
            y2 = self.centers[i[1]][1] # y-coordinate of second particle
            angle = np.arctan2((y2-y1),(x2-x1))#*(180/np.pi) # angle in degrees
            theta.append(angle)
        return theta
    
    def tag(self, area_frac):
        """
        Finds all tagged particles at some area fraction.

        Args:
            area_frac (float): the area fraction of interest
        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """
        swell = self.equiv_swell(area_frac)
        return self._tag(swell)
    
    def repel(self, pairs, area_frac, kick):
        """
        Repels overlapping particles.

        Args:
            pairs (np.array): the pairs of overlapping particles
            area_frac (float): the area fraction of interest
            kick (float): the max kick value the particles are repelled as a percent of the
                inverse diameter
        """
        swell = self.equiv_swell(area_frac)
        self._repel(pairs, swell, kick)

    def xform_boxsize(self, scale_x, scale_y):
        xform_boxsize_x = (self.boxsize_x*scale_x/scale_y)
        xform_boxsize_y = (self.boxsize_y*scale_y/scale_x)
        xform_boxsize = np.sqrt(xform_boxsize_x*xform_boxsize_y)
        return xform_boxsize
    
    def invxform_boxsize(self, scale_x, scale_y):
        xform_boxsize_x = (self.boxsize_x*scale_y/scale_x)
        xform_boxsize_y = (self.boxsize_y*scale_x/scale_y)
        xform_boxsize = xform_boxsize_x*xform_boxsize_y
        return xform_boxsize
    
    def transform_centers(self, scale_x, scale_y):
        for i in self.centers: #Transform centers
                i[0] = i[0]*(scale_x/scale_y)
                i[1] = i[1]*(scale_y/scale_x)
                
    def inv_transform_centers(self, scale_x, scale_y):
        for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y/scale_x)
                i[1] = i[1]*(scale_x/scale_y)
    
    def train(self, area_frac, kick, cycles=np.inf, noise_type='none', noise_val=0, counter='kicks', scale_x=1, scale_y=1):
        """
        Repeatedly tags and repels overlapping particles for some number of cycles
        
        Args:
            area_frac (float or list): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles. Defaults to infinite.
            noise (float): Value for standard deviation of gaussian noise to particle 
                position in each cycle, defaults to 0
            counter (kicks or list): whether to count a cycle as one kick or 
                one run over the input list
            scale_x (float): scale system in x-direction (function keeps particle area the same, no need for double inputting to account for particle area)
            scale_y (float): scale system in y-direction (function keeps particle area the same, no need for double inputting to account for particle area)

        Returns:
            (int) the number of tagging and repelling cycles until no particles overlapped
        """
        if not (counter=='kicks' or counter=='list'):
            print('invalid counter parameter, no training performed')
            return
        
        if not type(area_frac) == list:
            area_frac = [area_frac]
        
        count = 0
        while (cycles > count):
            untagged = 0
            for frac in area_frac:
                self.pos_noise(noise_type, noise_val)
                self.wrap()
                swell = self.equiv_swell(frac)
                xform_boxsize_x = (self.boxsize_x*scale_x/scale_y)
                xform_boxsize_y = (self.boxsize_y*scale_y/scale_x)
                pairs = self._tag(swell)
                for i in self.centers: #Transform centers
                    i[0] = i[0]*(scale_x/scale_y)
                    i[1] = i[1]*(scale_y/scale_x)
                pairs = self._tag_xform(swell, xform_boxsize_x, xform_boxsize_y)
                for i in self.centers: #Transform centers back
                    i[0] = i[0]*(scale_y/scale_x)
                    i[1] = i[1]*(scale_x/scale_y)
                if len(pairs) == 0:
                    untagged += 1
                    continue
                self._repel(pairs, swell, kick)
                self.wrap()
                if counter == 'kicks':
                    count += 1
                    if count >= cycles:
                        break
            if counter == 'list':
                count += 1
            if (untagged == len(area_frac) and noise_val == 0):
                break
        return count


    def particle_plot(self, area_frac, show=True, extend = False, figsize = (7,7), filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Args:
            area_frac (float): The diameter length at which the particles are illustrated
            show (bool): default True. Display the plot after generation
            extend (bool): default False. Show wrap around the periodic boundary.
            figsize ((int,int)): default (7,7). Scales the size of the figure
            filename (string): optional. Destination to save the plot. If None, the figure is not saved. 
        """
        radius = self.equiv_swell(area_frac)/2
        boxsize = self.boxsize
        fig = plt.figure(figsize = figsize)
        plt.axis('off')
        ax = plt.gca()
        for pair in self.centers:
            ax.add_artist(Circle(xy=(pair), radius = radius))
            if (extend):
                ax.add_artist(Circle(xy=(pair) + [0, boxsize], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [boxsize, 0], radius = radius, alpha=0.5))
                ax.add_artist(Circle(xy=(pair) + [boxsize, boxsize], radius = radius, alpha=0.5))
        if (extend):
            plt.xlim(0, 2*boxsize)
            plt.ylim(0, 2*boxsize)
            plt.plot([0, boxsize*2], [boxsize, boxsize], ls = ':', color = '#333333')
            plt.plot([boxsize, boxsize], [0, boxsize*2], ls = ':', color = '#333333')

        else:
            plt.xlim(0, boxsize)
            plt.ylim(0, boxsize)
        fig.tight_layout()
        if filename != None:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def _tag_count(self, swells):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            swell (float): swollen diameter length of the particles

        Returns:
            (float): The fraction of overlapping particles
        """
        i = 0
        tagged = np.zeros(swells.size)
        while i < swells.size:
            temp = self._tag(swells[i])
            tagged[i] = np.unique(temp).size/ self.N
            i += 1
        return tagged
    
    def tag_count(self, area_frac):
        """
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            area_frac (float): area fraction of the particles

        Returns:
            (float): The fraction of overlapping particles
        """
        swells = self.equiv_swell(area_frac)
        return self._tag_count(swells)

    def _extend_domain(self, domain):
        """
        Inserts a value at the beginning of the domain equal to the separation between the first
        two values, and a value at the end of the array determined by the separation of the last
        two values

        Args:
            domain (np.array): array to extend
        Return:
            (np.array) extended domain array
        """
        first = 2 * domain[0] - domain[1]
        if (first < 0):
            first = 0
        last = 2 * domain[-1] - domain[-2]
        domain_extend = np.insert(domain, 0, first)
        domain_extend = np.append(domain_extend, last)
        return domain_extend

    
    def tag_rate(self, area_frac):
        """
        Returns the rate at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the fraction tagged at two area fractions and dividing by the 
        difference of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate tag rate at

        Returns:
            (np.array): The rate of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        tagged = self.tag_count(af_extended)
        rate = (tagged[2:] - tagged[:-2])
        return rate

    def tag_curve(self, area_frac):
        """
        Returns the curvature at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the rate at two area fractions and dividing by the difference
        of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate the tag curvature at

        Returns:
            (np.array): The curvature of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        rate = self.tag_rate(af_extended)
        curve = (rate[2:] - rate[:-2])
        return curve

    def tag_plot(self, area_frac, mode='count', show=True, filename=None):
        """
        Generates a plot of the tag count, rate, or curvature

        Args:
            area_frac (np.array): list of the area fractions to use in the plot
            mode ("count"|"rate"|"curve"): which information you want to plot. Defaults to "count".
            show (bool): default True. Whether or not to show the plot
            filename (string): default None. Filename to save the plot as. If filename=None, the plot is not saved.
        """
        if (mode == 'curve'):
            plt.ylabel('Curve')
            func = self.tag_curve
        elif (mode == 'rate'):
            plt.ylabel('Rate')
            func = self.tag_rate
        else:
            plt.ylabel('Count')
            func = self.tag_count
        data = func(area_frac) 
        plt.plot(area_frac, data)
        plt.xlabel("Area Fraction")
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def detect_memory(self, start, end, incr):
        """
        Tests the number of tagged particles over a range of area fractions, and 
        returns a list of area fractions where memories are detected. 
        
        Args:
            start (float): The first area fraction in the detection
            end (float): The last area fraction in the detection
            incr (float): The increment between test swells. Determines accuracy of the memory detection. 
        Returns:
            (np.array): list of swells where a memory is located
        """
        area_frac = np.arange(start, end, incr)
        curve = self.tag_curve(area_frac)
        zeros = np.zeros(curve.shape)
        pos = np.choose(curve < 0, [curve, zeros])
        neg = np.choose(curve > 0, [curve, zeros])
        indices = peak.indexes(pos, 0.5, incr)
        nindices = peak.indexes(-neg, 0.5, incr)
        matches = []
        for i in indices:
            for j in nindices:
                desc = True
                if (i < j):
                    for k in range(i,j):
                        if (curve[k] < curve[k+1]):
                            desc = False
                    if (desc):
                        matches.append(i)
        return area_frac[matches]
    
    def _tag_count_xform(self, swells, scale_x, scale_y):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            swell (float): swollen diameter length of the particles
        Returns:
            (float): The fraction of overlapping particles
        """
        i = 0
        tagged = np.zeros(swells.size)
        xform_boxsize_x = (self.boxsize_x*(scale_x/scale_y))
        xform_boxsize_y = (self.boxsize_y*(scale_y/scale_x))
        while i < swells.size:
            temp = self._tag_xform(swells[i], xform_boxsize_x, xform_boxsize_y)
            tagged[i] = np.unique(temp).size/ self.N
            i += 1
        return tagged
    
    def tag_count_xform(self, area_frac, scale_x, scale_y):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Returns the number of tagged pairs at a specific area fraction
        
        Args:
            area_frac (float): area fraction of the particles
        Returns:
            (float): The fraction of overlapping particles
        """
        swells = self.equiv_swell(area_frac)
        return self._tag_count_xform(swells, scale_x, scale_y)
    
    def tag_rate_xform(self, area_frac, scale_x, scale_y):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Returns the rate at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the fraction tagged at two area fractions and dividing by the 
        difference of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate tag rate at
        Returns:
            (np.array): The rate of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        tagged = self.tag_count_xform(af_extended, scale_x, scale_y)
        rate = (tagged[2:] - tagged[:-2])
        return rate

    def tag_curve_xform(self, area_frac, scale_x, scale_y):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Returns the curvature at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the rate at two area fractions and dividing by the difference
        of the area fractions. 
        
        Args:
            area_frac (np.array): array fractions to calculate the tag curvature at
        Returns:
            (np.array): The curvature of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        rate = self.tag_rate_xform(af_extended, scale_x, scale_y)
        curve = (rate[2:] - rate[:-2])
        return curve

    def tag_plot_xform(self, scale_x, scale_y, area_frac, mode='count', show=True, filename=None):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Generates a plot of the tag count, rate, or curvature
        Args:
            scale_x (float, optional):
            scale_y (float, optional):
            area_frac (np.array): list of the area fractions to use in the plot
            mode ("count"|"rate"|"curve"): which information you want to plot. Defaults to "count".
            show (bool): default True. Whether or not to show the plot
            filename (string): default None. Filename to save the plot as. If filename=None, the plot is not saved.
        """
        for i in self.centers: #Transform centers along readout axis
                i[0] = i[0]*(scale_x/scale_y)
                i[1] = i[1]*(scale_y/scale_x)
        if (mode == 'curve'):
            plt.ylabel('Curve')
            func = self.tag_curve_xform
        elif (mode == 'rate'):
            plt.ylabel('Rate')
            func = self.tag_rate_xform
        else:
            plt.ylabel('Count')
            func = self.tag_count_xform
        data = func(area_frac, scale_x, scale_y) 
        plt.plot(area_frac, data)
        plt.xlabel("Area Fraction")
        for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y/scale_x)
                i[1] = i[1]*(scale_x/scale_y)
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()
