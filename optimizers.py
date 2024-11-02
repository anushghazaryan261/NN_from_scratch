import numpy as np

eps = 1e-8


class Adam:
    def __init__(self, shape=(1,), beta_1=0.9, beta_2=0.999):
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.beta1_deg = beta_1
        self.beta2_deg = beta_2
        self.v = np.zeros(shape)
        self.r = np.zeros(shape)

    def __call__(self, grad, lr=0.001):
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad
        self.r = self.beta2 * self.r + (1 - self.beta2) * grad ** 2

        v_hat = self.v / (1 - self.beta1_deg)
        r_hat = self.r / (1 - self.beta2_deg)
        self.beta1_deg *= self.beta1
        self.beta2_deg *= self.beta2

        return lr * v_hat / (np.sqrt(r_hat) + eps)


class AdaGrad:
    def __init__(self, shape=(1,), beta=0.9):
        self.r = np.zeros(shape)

    def __call__(self, grad, lr=0.001):
        self.r += grad ** 2

        return lr * grad / (np.sqrt(self.r) + eps)