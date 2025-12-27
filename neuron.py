# I will use NumPy for fast math like dot products
import numpy as np  

class Neuron:
    '''
    A single neuron implements:
      output = weighted sum + bias
      z = w·x + b    
      activation function for nonlinearity    
      a = activation(z)    
    '''

    def __init__(self, n_inputs: int, activation, rng=None, weight_scale: float = 0.1):
        # n_inputs = numbers this neuron expects as input 
        self.n_inputs = int(n_inputs)

        # will be using Tanh() for nonlinearity
        self.activation = activation

        # rng is a random number generator
        self.rng = rng or np.random.default_rng(0)

        # Initialize weights: one weight per input feature 
        self.w = self.rng.normal(0.0, weight_scale, size=(self.n_inputs,)).astype(np.float32)

        # Initialize bias to 0 
        self.b = np.float32(0.0)

        # Gradients for parameters 
        # dw will store dL/dw 
        self.dw = np.zeros_like(self.w)

        # db will store dL/db 
        self.db = np.float32(0.0)

    def forward(self, x):
        # Convert input to a NumPy float32 array
        x = np.asarray(x, dtype=np.float32)

        # Compute pre-activation output: 
        z = np.dot(self.w, x) + self.b

        # Apply activation function:
        a = self.activation.forward(z)

        # Cache values needed later for backprop:
        # - x is needed because dL/dw depends on x
        # - z is needed because activation derivative depends on z
        cache = {"x": x, "z": np.float32(z)}

        return np.float32(a), cache

    def backward(self, grad_a, cache):
        '''
        Backprop for this neuron.
        grad_a is dL/da: gradient of loss with respect to neuron output a (scalar).
        cache contains x and z from the matching forward() call.
        Returns: dL/dx: gradient of loss with respect to input vector x (shape: (n_inputs,))
        '''

        # Ensure scalar gradient is float32
        grad_a = np.float32(grad_a)

        x = cache["x"]  # input vector
        z = cache["z"]  # pre-activation value

        # grad_z = dL/dz = dL/da * da/dz
        grad_z = self.activation.backward(z, grad_a).astype(np.float32)

        # Accumulate gradient for weights and bias
        self.dw += grad_z * x
        self.db += grad_z

        # Gradient w.r.t inputs
        grad_x = grad_z * self.w

        # Return gradient for earlier layer / earlier timestep
        return grad_x.astype(np.float32)

    def zero_grad(self):
        # Reset accumulated gradients to zero before a new training step
        self.dw.fill(0.0)           
        self.db = np.float32(0.0)   

    def step(self, lr: float = 1e-3, clip: float = 5.0):
        # Prevent exploding gradients by clipping
        self.dw = np.clip(self.dw, -clip, clip)  
        self.db = np.clip(self.db, -clip, clip)  

        # Gradient Descent update in opposite direction of gradient:
        # w = w - lr * dL/dw
        self.w -= np.float32(lr) * self.dw
        # b = b - lr * dL/db
        self.b -= np.float32(lr) * self.db
