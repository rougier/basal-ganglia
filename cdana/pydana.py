"""Pure Python version of cdana.pyx

This is meant as a more concise version of the equivalent Cython code.
The equivalence is ensured by thorough unit testing.
"""
import numpy as np

    ## Activation functions

class Identity(object):
    def call(self, x):
        return max(0.0, x)


class Clamp(object):
    def __init__(self, min=0, max=1e9):
        self.min, self.max = min, max

    def call(self, x):
        return np.clip(x, low=self.min, high=self.max)


class UniformNoise:
    def __init__(self, amount):
        self.half_amount = amount/2

    def call(self, x):
        return x + np.random.uniform(-self.half_amount, self.half_amount)


class Sigmoid(object):
    def __init__(self, Vmin=0.0, Vmax=20.0, Vh=16., Vc=3.0):
        self.Vmin, self.Vmax, self.Vh, self.Vc = Vmin, Vmax, Vh, Vc

    def call(self, V):
        return (self.Vmin + (self.Vmax - self.Vmin)/(1.0 + np.exp((self.Vh - V)/self.Vc)))


    ## Group of Artificial Neurons

class Group(object):
    """Matrix of neurons sharing the same parameters"""

    def __init__(self, shape, tau=0.01, rest=0.0, noise=0.0, activation=Identity()):
        """
        tau: Membrane time constant
        noise: Noise level
        """
        self.shape      = shape
        self.tau        = tau         # Membrane time constant
        self.rest       = rest        # Membrane resting potential
        self.noise      = noise       # Noise level
        self.activation = activation  # Activation function

        self.flush()

    def flush(self):
        """ Flush all activities and reset history index """
        self.units = np.zeros(self.shape,
                        dtype=[(   'V', float),  # Firing rate
                               (   'U', float),  # Membrane potential
                               ('Isyn', float),  # Input current from external synapses
                               ('Iext', float)]) # Input current from external sources

        self.delta = 0  # Difference of activity between the first two maximum activites
        self.history = []

    def evaluate(self, dt):
        """ Compute activities (Forward Euler method) """
        max1, max2 = float('-inf'), float('-inf')

        for unit in self.units:
            unit['U'] += dt/self.tau*(-unit['U'] + unit['Isyn'] + unit['Iext'] - self.rest)  # Update membrane potential
            noise = 1 + self.noise*np.random.uniform(low=-0.5, high=0.5)                     # Compute white noise
            unit['V'] = self.activation.call(unit['U']*noise)                                # Update firing rate

            if   unit['V'] > max1: max1, max2 = unit['V'], max1  # Here we record the max activities to store their difference
            elif unit['V'] > max2: max2 = unit['V']              # This is used later to decide if a motor decision has been made

        self.history.append(self.units['V']) # Store firing rate activity
        self.delta = max1 - max2

    def __getitem__(self, key): # quick access to the units's fields
        return self.units[key]

    def __setitem__(self, key, value):
        self.units[key] = value


    ## Connections

class Connection(object):
    def __init__(self, source, target, weights, gain):
        self.source, self.target = source, target
        self.weights, self.gain = weights, gain # weight matrix and gain (scalar)
        self.source_2D  = self.source.view().reshape(int(len(self.source)/4), 4)
        self.target_2D  = self.target.view().reshape(int(len(self.target)/4), 4)
        self.weights_2D = self.weights.view().reshape(int(len(self.weights)/4), 4)

class OneToOne(Connection):
    def propagate(self):
        self.target += self.gain * self.source * self.weights

class OneToAll(Connection):
    def propagate(self):
        self.target += self.gain * np.sum(self.source * self.weights)

class AssToMot(Connection):
    def propagate(self):
        self.target += self.gain * np.sum(self.weights * self.source_2D, axis=0) # np.dot(self.weights, self.source.reshape((4, 4)))

class AssToCog(Connection):
    def propagate(self):
        self.target += self.gain * np.sum(self.weights * self.source_2D.T, axis=0)

class MotToAss(Connection):
    def propagate(self):
        self.target_2D += self.gain * self.source * self.weights # adding on every column

class CogToAss(Connection):
    def propagate(self):
        self.target_2D += self.gain * (self.source * self.weights).reshape(4, 1) # adding on every line

class AllToAll(Connection):
    def propagate(self):
        self.target += self.gain * np.dot(self.source, self.weights_2D)
