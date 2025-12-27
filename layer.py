import numpy as np
from neuron import Neuron

# A dense layer built from many Neuron objects.
class Layer:
    def __init__(self, n_inputs: int, n_units: int, activation, rng=None, weight_scale=0.1):
        self.n_inputs = int(n_inputs)
        self.n_units = int(n_units)
        self.activation = activation

        # rng used for reproducible initialization
        self.rng = rng or np.random.default_rng(0)

        # A list of neurons - each has its own weights/bias
        self.neurons = [
            Neuron(n_inputs=self.n_inputs, activation=self.activation, rng=self.rng, weight_scale=weight_scale)
            for _ in range(self.n_units)
        ]

    def forward(self, x):
        outs = np.zeros((self.n_units,), dtype=np.float32)
        caches = []

        for i, n in enumerate(self.neurons):
            outs[i], c = n.forward(x)
            caches.append(c)

        return outs, caches

    def backward(self, grad_out, caches):
        grad_out = np.asarray(grad_out, dtype=np.float32)
        if grad_out.shape != (self.n_units,):
            raise ValueError(f"Expected grad_out shape {(self.n_units,)}, got {grad_out.shape}")

        # All neurons receive the same input x, so total dL/dx is sum over neurons
        grad_x = np.zeros((self.n_inputs,), dtype=np.float32)

        for i, n in enumerate(self.neurons):
            grad_x += n.backward(grad_out[i], caches[i])

        return grad_x.astype(np.float32)

    # Reset gradients for every neuron in the layer
    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

    # Update each neuron with SGD
    def step(self, lr=1e-3, clip=5.0):
        for n in self.neurons:
            n.step(lr=lr, clip=clip)
