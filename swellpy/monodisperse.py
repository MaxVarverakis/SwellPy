import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
from peakutils import peak
import crepel
from .particle_system import ParticleSystem
import pandas as pd

class Monodisperse(ParticleSystem):

    def moving_average(self,x, w):
        """
        Smoothens out dataset using rolling averages

        Args:
            x (array): dataset
            w (int): number of input elements to be averaged over
        """
        return np.convolve(x, np.ones(w), 'valid') / w

    def __init__(self, N, boxsize=None, boxsize_x=None, boxsize_y=None, seed=None):
        """
        Args:
            N (int): The number of particles in the system
            boxsize (float): optional. Length of the sides of the box
            seed (int): optional. Seed for initial particle placement randomization
        """
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
        xform_boxsize_x = (self.boxsize_x*scale_x
        #/scale_y
        )
        xform_boxsize_y = (self.boxsize_y*scale_y
        #/scale_x
        )
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
    
    def _tag_xform(self, swell, xform_boxsize_x, xform_boxsize_y):
        """ 
        Get the center indices of the particles that overlap at a 
        specific swell. Takes in account the transformation of boxsize.
        
        Parameters:
            swell (float): diameter length of the particles
            xform_boxsize_x (float): X transform boxsize with respect to scaling size
            xform_boxsize_y (float): Y transform boxsize with respect to scaling size
        Returns:
            (np.array): An array object whose elements are pairs of int values that correspond
                the the center indices of overlapping particles
        """

        # Note cKD can retun numpy arrays in query pairs
        # but there is a deallocation bug in the scipy.spatial code
        # converting from a set to an array avoids it
        tree = cKDTree(self.centers, boxsize = (xform_boxsize_x, xform_boxsize_y))
        pairs = tree.query_pairs(swell)
        pairs = np.array(list(pairs), dtype=np.int64)
        return pairs
    
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
        xform_boxsize_x = (self.boxsize_x*scale_x
        #/scale_y
        )
        xform_boxsize_y = (self.boxsize_y*scale_y
        #/scale_x
        )
        xform_boxsize = np.sqrt(xform_boxsize_x*xform_boxsize_y)
        return xform_boxsize
    
    def invxform_boxsize(self, scale_x, scale_y):
        xform_boxsize_x = (self.boxsize_x*scale_y
        #/scale_x
        )
        xform_boxsize_y = (self.boxsize_y*scale_x
        #/scale_y
        )
        xform_boxsize = xform_boxsize_x*xform_boxsize_y
        return xform_boxsize
    
    def transform_centers(self, scale_x, scale_y):
        for i in self.centers: #Transform centers
                i[0] = i[0]*(scale_x
                #/scale_y
                )
                i[1] = i[1]*(scale_y
                #/scale_x
                )
                
    def inv_transform_centers(self, scale_x, scale_y):
        for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y
                #/scale_x
                )
                i[1] = i[1]*(scale_x
                #/scale_y
                )
    
    def train(self, area_frac, kick, cycles=np.inf, noise_type='none', noise_val=0, counter='kicks', scale_x=1, scale_y=1,iso_noise = True, noise_data = False):
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
            iso_noise (bool): default True.  Whether to use isotropic noise or anisotropic noise.  Isotropic noise doesn't consider anisotropoic transformations,
             whereas anisotropic noise considers transformations.
            noise_data (bool): default False.  True: returns an array with the change in (x,y) of particles chosen for noise

        Returns:
            (int) the number of tagging and repelling cycles until no particles overlapped
        """
        if not (counter=='kicks' or counter=='list'):
            print('invalid counter parameter, no training performed')
            return
        
        if not type(area_frac) == list:
            area_frac = [area_frac]
        
        count = 0
        disp = []
        while (cycles > count):
            untagged = 0
            for frac in area_frac:
                coords = self.centers.copy()
                if iso_noise == True or (scale_x==1 and scale_y==1):
                    self.pos_noise(noise_type, noise_val)
                    self.wrap()
                    if noise_data == True:
                        for i in range(len(self.centers)):
                            if np.all(coords[i] == self.centers[i]) == False:
                                disp.append([self.centers[i][0] - coords[i][0], self.centers[i][1] - coords[i][1]])
                swell = self.equiv_swell(frac)
                xform_boxsize_x = (self.boxsize_x*scale_x)
                xform_boxsize_y = (self.boxsize_y*scale_y)
                if (scale_x==1) and (scale_y==1):
                    pairs = self._tag(swell)
                else:
                    if iso_noise == False:
                        self.anisotropic_noise(noise_type,noise_val,scale_x,scale_y)
                        self.wrap()
                        if noise_data == True:
                            for i in range(len(self.centers)):
                                if np.all(coords[i] == self.centers[i]) == False:
                                    disp.append([self.centers[i][0] - coords[i][0], self.centers[i][1] - coords[i][1]])
                    for i in self.centers: #Transform centers
                        i[0] = i[0]*(scale_x)
                        i[1] = i[1]*(scale_y)
                    pairs = self._tag_xform(swell, xform_boxsize_x, xform_boxsize_y)
                    for i in self.centers: #Transform centers back
                        i[0] = i[0]*(scale_y)
                        i[1] = i[1]*(scale_x)
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
        return [count,disp]
    
    def noise_data(self, area_frac, kick, cycles, noise_type, noise_val, counter = 'list', scale_x = 1, scale_y = 1, iso_noise = True):
        """
        Trains the system and finds the distribution of particle displacement and angle for particles chosen by noise for the last (cycles) cycles
        *** Only works for one area fraction at a time ***
        
        Args:
            area_frac (float): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles to train and to gather noise data
            noise (float): Value for standard deviation of gaussian noise to particle 
                position in each cycle
            counter (kicks or list): whether to count a cycle as one kick or 
                one run over the input list (defaults to list)
            scale_x (float): scale system in x-direction (function keeps particle area the same, no need for double inputting to account for particle area) (defaults to 1)
            scale_y (float): scale system in y-direction (function keeps particle area the same, no need for double inputting to account for particle area) (defaults to 1)
            iso_noise (bool): default True.  Whether to use isotropic noise or anisotropic noise.  Isotropic noise doesn't consider anisotropoic transformations,
             whereas anisotropic noise considers transformations.

        Returns:
            (array) Array containing four sub-arrays for the distribution and mean of particle displacement and angle for particles chosen by noise.
            The array is composed of [[displacement],[displacement mean],[displacement standard deviation],[angle],[angle mean],[angle standard deviation]].
        """
        displacement = self.train(area_frac, kick, cycles, noise_type, noise_val, counter, scale_x, scale_y, iso_noise, noise_data = True)[1]
        disp = []
        theta  = []
        for i in range(len(displacement)):
            dy = displacement[i][1]
            dx = displacement[i][0]
            disp.append(np.sqrt(dx**2+dy**2))
            theta.append(np.arctan2(dy,dx))
        dm = np.mean(disp)
        ds = np.std(disp)
        tm = np.mean(theta)
        ts = np.std(theta)
        return([[disp],[dm],[ds],[theta],[tm],[ts]])

    def noise_data_plot(self, area_frac, kick, cycles, noise_type, noise_val, counter = 'list', scale_x = 1, scale_y = 1, iso_noise = True):
        """
        Trains the system and finds the distribution of particle displacement and angle for particles chosen by noise for the last (cycles) cycles
        *** Only works for one area fraction at a time ***
        
        Args:
            area_frac (float): the area fraction to train on
            kick (float): the maximum distance particles are repelled
            cycles (int): The upper bound on the number of cycles to train and to gather noise data
            noise (float): Value for standard deviation of gaussian noise to particle 
                position in each cycle
            counter (kicks or list): whether to count a cycle as one kick or 
                one run over the input list
            scale_x (float): scale system in x-direction (function keeps particle area the same, no need for double inputting to account for particle area)
            scale_y (float): scale system in y-direction (function keeps particle area the same, no need for double inputting to account for particle area)
            iso_noise (bool): default True.  Whether to use isotropic noise or anisotropic noise.  Isotropic noise doesn't consider anisotropoic transformations,
             whereas anisotropic noise considers transformations.

        Returns:
            (plots) Histograms showing the distribution and mean of particle displacement and angle for particles chosen by noise
        """
        displacement = self.train(area_frac, kick, cycles, noise_type, noise_val, counter, scale_x, scale_y, iso_noise, noise_data = True)[1]
        disp = []
        theta  = []
        for i in range(len(displacement)):
            dy = abs(displacement[i][1])
            dx = abs(displacement[i][0])
            disp.append(np.sqrt(dx**2+dy**2))
            theta.append(np.arctan2(dy,dx))
        dm = np.mean(disp)
        ds = np.std(disp)
        tm = np.mean(theta)
        tm_pi = tm/np.pi
        ts = np.std(theta)
        ts_pi = ts/np.pi
        plt.hist(disp,bins=45)
        plt.axvline(dm,color='r',ls='--',label='Displacement Mean: %.2f' % dm)
        plt.axvline(dm+ds,color='k',ls='--',label='Standard Deviation: %.2f' % ds)
        plt.axvline(dm-ds,color='k',ls='--')
        plt.title('Noise Displacement Distribution')
        plt.xlabel('Noise Displacement')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

        plt.hist(theta,bins=45)
        plt.axvline(tm,color='r',ls='--',label='Angle Mean: %.2f$\pi$' % tm_pi)
        plt.axvline(tm+ts,color='k',ls='--',label='Standard Deviation: %.2f$\pi$' % ts_pi)
        plt.axvline(tm-ts,color='k',ls='--')
        plt.title('Noise Angle Distribution')
        plt.xlabel('Noise Angle [rad]')
        plt.ylabel('Count')
        plt.xticks([0,np.pi/4,np.pi/2],labels=['0','$\pi/4$','$\pi/2$'])
        plt.legend()
        plt.show()

    def particle_plot(self, area_frac, scale_x = 1, scale_y = 1, shape = 'circle', mode = 'none', sub1 = 0, sub2 = 0, show = True, extend = False, figsize = (7,7), filename=None):
        """
        Show plot of physical particle placement in 2-D box 
        
        Args:
            area_frac (float): The diameter length at which the particles are illustrated
            scale_x (float): scaling in x direction
            scale_y (float): scaling in y direction
            shape (str): default 'circle'.  'circle': display particles as circles; 'ellipse': display particles as ellipses corresponding to transform size
            mode (str): default 'none'. 'tag': Color the tagged particles red.  'subtract': Show the newly tagged particles within an interval of area fractions
            sub1 (float): Lower area fraction bound for subtract
            sub2 (float): Upper area fraction bound for subtract
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
        xform_boxsize_x = (self.boxsize_x*scale_x)
        xform_boxsize_y = (self.boxsize_y*scale_y)
        if shape == 'ellipse':
            for pair in self.centers:
                ax.add_artist(Ellipse(xy=(pair), width = 2*scale_y*radius, height = 2*scale_x*radius))
                if (extend):
                    ax.add_artist(Ellipse(xy=(pair) + [0, boxsize], width = 2*scale_y*radius, height = 2*scale_x*radius, alpha=0.5))
                    ax.add_artist(Ellipse(xy=(pair) + [boxsize, 0], width = 2*scale_y*radius, height = 2*scale_x*radius, alpha=0.5))
                    ax.add_artist(Ellipse(xy=(pair) + [boxsize, boxsize], width = 2*scale_y*radius, height = 2*scale_x*radius, alpha=0.5))
            if mode == 'tag':
                swell = self.equiv_swell(area_frac)
                if (scale_x==1) and (scale_y==1):
                    pairs = self._tag(swell)
                else:
                    self.transform_centers(scale_x,scale_y)
                    pairs = self._tag_xform(swell, xform_boxsize_x, xform_boxsize_y)
                    self.inv_transform_centers(scale_x, scale_y)
                for i in pairs:
                    ax.add_artist(Ellipse(xy=(self.centers[i[0]]), width = 2*scale_y*radius, height = 2*scale_x*radius, color = 'r'))
                    ax.add_artist(Ellipse(xy=(self.centers[i[1]]), width = 2*scale_y*radius, height = 2*scale_x*radius, color = 'r'))
            elif mode == 'subtract':
                swell1 = self.equiv_swell(sub1)
                swell2 = self.equiv_swell(sub2)
                if (scale_x==1) and (scale_y==1):
                    pairs1 = self._tag(swell1)
                    pairs2 = self._tag(swell2)
                else:
                    self.transform_centers(scale_x,scale_y)
                    pairs1 = self._tag_xform(swell1, xform_boxsize_x, xform_boxsize_y)
                    pairs2 = self._tag_xform(swell2, xform_boxsize_x, xform_boxsize_y)
                    self.inv_transform_centers(scale_x, scale_y)
                pairs = []
                for i in range(len(pairs2)): 
                    if pairs2[i] not in pairs1:
                        pairs.append(pairs2[i])
                for i in pairs:
                    ax.add_artist(Ellipse(xy=(self.centers[i[0]]), width = 2*scale_y*radius, height = 2*scale_x*radius, color = 'r'))
                    ax.add_artist(Ellipse(xy=(self.centers[i[1]]), width = 2*scale_y*radius, height = 2*scale_x*radius, color = 'r'))
        elif shape == 'circle':
            for pair in self.centers:
                ax.add_artist(Circle(xy=(pair), radius = radius))
                if (extend):
                    ax.add_artist(Circle(xy=(pair) + [0, boxsize], radius = radius, alpha=0.5))
                    ax.add_artist(Circle(xy=(pair) + [boxsize, 0], radius = radius, alpha=0.5))
                    ax.add_artist(Circle(xy=(pair) + [boxsize, boxsize], radius = radius, alpha=0.5))
            if mode == 'tag':
                swell = self.equiv_swell(area_frac)
                if (scale_x==1) and (scale_y==1):
                    pairs = self._tag(swell)
                else:
                    self.transform_centers(scale_x,scale_y)
                    pairs = self._tag_xform(swell, xform_boxsize_x, xform_boxsize_y)
                    self.inv_transform_centers(scale_x, scale_y)
                for i in pairs:
                    ax.add_artist(Circle(xy=(self.centers[i][0]), radius = radius, color = 'r'))
                    ax.add_artist(Circle(xy=(self.centers[i][1]), radius = radius, color = 'r'))
            elif mode == 'subtract': 
                swell1 = self.equiv_swell(sub1)
                swell2 = self.equiv_swell(sub2)
                if (scale_x==1) and (scale_y==1):
                    pairs1 = self._tag(swell1)
                    pairs2 = self._tag(swell2)
                else:
                    self.transform_centers(scale_x,scale_y)
                    pairs1 = self._tag_xform(swell1, xform_boxsize_x, xform_boxsize_y)
                    pairs2 = self._tag_xform(swell2, xform_boxsize_x, xform_boxsize_y)
                    self.inv_transform_centers(scale_x, scale_y)
                pairs = []
                for i in range(len(pairs2)): 
                    if pairs2[i] not in pairs1:
                        pairs.append(pairs2[i])
                for i in pairs:
                    ax.add_artist(Circle(xy=(self.centers[i][0]), radius = radius, color = 'r'))
                    ax.add_artist(Circle(xy=(self.centers[i][1]), radius = radius, color = 'r'))
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
    
    def detect_memory_xform(self, start, end, incr, scale_x = 1,scale_y = 1):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Tests the number of tagged particles over a range of area fractions, and 
        returns a list of area fractions where memories are detected. 
        
        Args:
            start (float): The first area fraction in the detection
            end (float): The last area fraction in the detection
            incr (float): The increment between test swells. Determines accuracy of the memory detection. 
            scale_x (float): scaling in x direction (function keeps particle area the same, no need for double inputting to account for particle area)
            scale_y (float): scaling in y direction (function keeps particle area the same, no need for double inputting to account for particle area)
        Returns:
            (np.array): list of swells where a memory is located
        """
        area_frac = np.arange(start, end, incr)
        curve = self.tag_curve_xform(area_frac, scale_x, scale_y)
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

    def detect_memory_xform_rolled(self, start, end, incr, w, scale_x = 1,scale_y = 1):
        """
        Memory Read-Out function set for reading along a given axis * adjusted with rolling mean *.
        
        Tests the number of tagged particles over a range of area fractions, and 
        returns a list of area fractions where memories are detected. * after boxcar/rolling mean is applied *
        
        Args:
            start (float): The first area fraction in the detection
            end (float): The last area fraction in the detection
            incr (float): The increment between test swells. Determines accuracy of the memory detection. 
            w (int): number of input elements to be averaged over
            scale_x (float): scaling in x direction (function keeps particle area the same, no need for double inputting to account for particle area)
            scale_y (float): scaling in y direction (function keeps particle area the same, no need for double inputting to account for particle area)
        Returns:
            (np.array): list of swells where a memory is located
        """
        area_frac = np.arange(start, end, incr)
        curve = self.tag_curve_xform_rolled(area_frac, scale_x, scale_y, w)
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
        xform_boxsize_x = (self.boxsize_x*(scale_x
        #/scale_y
        ))
        xform_boxsize_y = (self.boxsize_y*(scale_y
        #/scale_x
        ))
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
        for i in self.centers: #Transform centers along readout axis
                i[0] = i[0]*(scale_x
                #/scale_y
                )
                i[1] = i[1]*(scale_y
                #/scale_x
                )
        swells = self.equiv_swell(area_frac)
        d = self._tag_count_xform(swells, scale_x, scale_y)
        for i in self.centers: #Transform centers back
                i[0] = i[0]*(scale_y
                #/scale_x
                )
                i[1] = i[1]*(scale_x
                #/scale_y
                )
        return d
    
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

    def tag_curve_xform_rolled(self, area_frac, scale_x, scale_y, w):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Returns the curvature at which the fraction of particles overlap over a range of area fractions.
        This is the same as measuring the rate at two area fractions and dividing by the difference
        of the area fractions. * after boxcar/rolling mean is applied *
        
        Args:
            area_frac (np.array): array fractions to calculate the tag curvature at
            w (int): number of input elements to be averaged over
        Returns:
            (np.array): The curvature of the fraction of tagged particles at area fraction in the input array
        """
        af_extended = self._extend_domain(area_frac)
        rate = self.tag_rate_xform(af_extended, scale_x, scale_y)
        curve = (rate[2:] - rate[:-2])
        adj_curve  = np.ones(len(curve))
        d = self.moving_average(curve,w)
        for j in range(len(curve)):
            if j >= w - 1:
                adj_curve[j] = d[j - w]
            else:
                adj_curve[j] = curve[j]
        return adj_curve

    def tag_plot_xform(self, scale_x, scale_y, area_frac, mode='count', show=True, filename=None):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Generates a plot of the tag count, rate, or curvature
        Args:
            scale_x (float):
            scale_y (float):
            area_frac (np.array): list of the area fractions to use in the plot
            mode ("count"|"rate"|"curve"): which information you want to plot. Defaults to "count".
            show (bool): default True. Whether or not to show the plot
            filename (string): default None. Filename to save the plot as. If filename=None, the plot is not saved.
        """
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
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()

    def tag_plot_xform_rolled(self, scale_x, scale_y, area_frac, w, mode='count', show=True, filename=None):
        """
        Memory Read-Out function set for reading along a given axis.
        
        Generates a plot of the tag count, rate, or curvature * after boxcar/rolling mean is applied *

        Args:
            scale_x (float, optional):
            scale_y (float, optional):
            area_frac (np.array): list of the area fractions to use in the plot
            w (int): number of input elements to be averaged over
            mode ("count"|"rate"|"curve"): which information you want to plot. Defaults to "count".
            show (bool): default True. Whether or not to show the plot
            filename (string): default None. Filename to save the plot as. If filename=None, the plot is not saved.
        """
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
        rolled_data = np.ones(len(data))
        d = self.moving_average(data,w)
        for j in range(len(data)):
            if j >= w - 1:
                rolled_data[j] = d[j - w]
            else:
                rolled_data[j] = data[j]
            j += 1
        plt.plot(area_frac, rolled_data)
        plt.xlabel("Area Fraction")
        if filename:
            plt.savefig(filename)
        if show == True:
            plt.show()
        plt.close()
