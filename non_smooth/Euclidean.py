import numpy as np
class Euclidean:
    def dot(self, x, y):
        return np.dot(x.flatten(), y.flatten())

    def apply(self, x, y):
        return self.dot(x, y)

    def norm(self, x):
        return np.sqrt(self.dot(x, x))

    def dual(self, x):
        return x