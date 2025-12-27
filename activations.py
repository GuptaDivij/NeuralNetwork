import numpy as np

class Tanh:
    def forward(self, z):
        return np.tanh(np.float32(z)).astype(np.float32)

    def backward(self, z, grad_a):
        z = np.float32(z)
        grad_a = np.float32(grad_a)
        t = np.tanh(z).astype(np.float32)
        return (grad_a * (1.0 - t * t)).astype(np.float32)

class Linear:
    def forward(self, z):
        return np.float32(z)

    def backward(self, z, grad_a):
        return np.float32(grad_a)
