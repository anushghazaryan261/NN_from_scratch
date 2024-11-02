import numpy as np
from NN_from_scratch.optimizers import Adam, AdaGrad

# np.random.seed(2)
epsilon = 1e-8


class Dropout:
    def __init__(self, rate=0.1, neurons=32):
        self.rate = rate
        self.scale = 1 / (1 - rate)
        self.neurons = neurons
        self.weights = 0

    def _init_weights(self, neurons):
        self.neurons = neurons
        self.weights = np.random.uniform(size=self.neurons) > self.rate
        self.scale = np.prod(neurons) / self.weights.sum()

    def _feedforward(self, X):
        self._init_weights(X.shape[1:])
        return X * self.weights * self.scale

    def _backpropagate(self, neuron_grads):
        return neuron_grads * self.weights  # * self.scale**(-1)


class BatchNormalization:
    def __init__(self, neurons=32):
        self.neurons = neurons
        self.min = 0
        self.max = 0

    def _feedforward(self, X):
        self.min = np.min(X, axis=0)
        X -= self.min
        self.max = np.max(X, axis=0)
        X /= (self.max + 1e-9)
        return X

    def _backpropagate(self, neuron_grads):
        return neuron_grads * self.max


def relu(X):
    return (X > 0) * X


def relu_derivative(X):
    return X > 0


def sigmoid(X):
    return 1 / (1 + np.exp(-X) + epsilon)


def sigmoid_derivative(X):
    sgm = sigmoid(X)
    return sgm * (1 - sgm)


def softmax(X):
    exps = np.exp(X)
    sm = exps.sum(axis=tuple(np.arange(1, X.ndim)))
    return exps / (sm.reshape(sm.shape + tuple(1 for _ in exps.shape[1:])) + epsilon)


def softmax_derivative(X):
    sftmx = softmax(X)
    return sftmx * (1 - sftmx)


def tanh(X):
    return 1 - (2 * np.exp(-X) / (np.exp(X) + np.exp(-X)))


def tanh_derivative(X):
    return 1 - tanh(X) ** 2


def apply_activation(X, activation):
    if activation == 'relu':
        return relu(X)
    elif activation == 'sigmoid':
        return sigmoid(X)
    elif activation == 'tanh':
        return tanh(X)
    elif activation == 'softmax':
        return softmax(X)
    return X


def apply_derivative(X, activation):
    if activation == 'relu':
        return relu_derivative(X)
    elif activation == 'sigmoid':
        return sigmoid_derivative(X)
    elif activation == 'tanh':
        return tanh_derivative(X)
    elif activation == 'softmax':
        return softmax_derivative(X)
    return 1


def init_weights(weight_shape, inp_size, out_size, layer_indx, kernel_initializer='glorot_uniform'):
    if kernel_initializer == 'glorot_uniform':
        std = np.sqrt(1 / (inp_size + out_size))
        return np.random.uniform(-std, std, size=weight_shape)
    elif kernel_initializer == 'he_normal':
        std = np.sqrt(1 / np.prod(inp_size))
        return np.random.normal(0, std, size=weight_shape)
    elif kernel_initializer == 'xavier_orig':
        std = 1 / (np.prod(inp_size) ** (layer_indx - 1))
        return np.random.normal(0, std, size=weight_shape)


def create_optimizer(shape, optimizer):
    if optimizer == 'adam':
        return Adam(shape=shape)
    elif optimizer == 'adagrad':
        return AdaGrad(shape=shape)


class InpLayer:
    def __init__(self, shape):
        self.neurons = shape

    def _feedforward(self, inp):
        return inp


class RNN:
    def __init__(self, activation='tanh', neurons=(20, 20), include_bias=True, optimizer='adam',
                 k_init='glorot_uniform'):
        if type(activation) in [list, tuple]:
            self.activation = activation[0]
            self.activation_dense = activation[1]
        else:
            self.activation = activation
            self.activation_dense = 'linear'
        self.optimizer = optimizer
        self.kernel_initializer = k_init
        self.neurons = neurons
        self.include_bias = include_bias
        self.weights_inp = []
        self.weights_h = []
        self.dense = DenseLayer(neurons=neurons[1], activation=self.activation_dense, k_init='glorot_uniform')

        self.grads_inp = []
        self.grads_h = []
        self.grads_bias = []
        self.next_timestep_grad = []

        self.inp_history = []
        self.ans_history = []

    def _reset_grads(self):
        self.next_timestep_grad = np.zeros(self.neurons[0])
        self.grads_inp = np.zeros(self.weights_inp.shape)
        self.grads_h = np.zeros(self.weights_h.shape)
        if self.include_bias:
            self.grads_bias = np.zeros(self.neurons[0])

    def _init_weights(self, inp_size, layer_indx):
        if type(inp_size) is np.ndarray:
            inp_size = inp_size.prod()

        self.weights_inp = init_weights((inp_size, self.neurons[0]), inp_size, self.neurons[0], layer_indx,
                                        self.kernel_initializer)
        self.w_in_optim = create_optimizer(self.weights_inp.shape, self.optimizer)
        self.weights_h = init_weights((self.neurons[0], self.neurons[0]), self.neurons[0], self.neurons[0], layer_indx,
                                      self.kernel_initializer)
        self.w_h_optim = create_optimizer(self.weights_h.shape, self.optimizer)

        if self.include_bias:
            self.bias = np.zeros(self.neurons[0])
            self.b_optim = create_optimizer(self.bias.shape, self.optimizer)

        self._reset_grads()

        self.dense._init_weights(self.neurons[0], layer_indx)
        return self.neurons[1]

    def _feedforward(self, inp):
        """
        inp is a 1D array representing the previous layer
        """
        if inp.ndim > 2:
            inp = inp.reshape(inp.shape[0], -1)

        # print(inp.shape, self.weights_inp.shape)
        ans = inp @ self.weights_inp
        if len(self.ans_history) != 0:
            a1 = apply_activation(self.ans_history[-1], self.activation)
            ans += a1 @ self.weights_h
        if self.include_bias:
            ans += self.bias

        self.inp_history.append(inp)
        self.ans_history.append(ans)

        ans = apply_activation(ans, self.activation)

        return self.dense._feedforward(ans)

    def _backpropagate(self, neuron_grads):

        neuron_grads = self.dense._backpropagate(neuron_grads)

        neuron_grads += self.next_timestep_grad
        neuron_grads *= apply_derivative(self.ans_history[-1], self.activation)
        # saving the gradient coming from hidden state
        self.next_timestep_grad = neuron_grads @ self.weights_h.T

        g = self.inp_history[-1].reshape(self.inp_history[-1].shape + (1,)) @ \
            neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
        self.grads_inp += g.mean(axis=0)

        if len(self.ans_history) > 1:
            a1 = apply_activation(self.ans_history[-2], self.activation)
            g = a1.reshape(self.ans_history[-1].shape + (1,)) @ \
                neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
            self.grads_h += g.mean(axis=0)

        if self.include_bias:
            self.grads_bias += neuron_grads.mean(axis=0)

        self.ans_history.pop()
        self.inp_history.pop()

        return neuron_grads @ self.weights_inp.T

    def _apply_grads(self, lr):
        self.dense._apply_grads(lr)

        self.weights_inp -= self.w_in_optim(self.grads_inp, lr)
        self.weights_h -= self.w_h_optim(self.grads_h, lr)
        if self.include_bias:
            self.bias -= self.b_optim(self.grads_bias, lr)
        self._reset_grads()


class Conv2D:
    def __init__(self, activation='linear', kernel_size=(3, 3), stride=(1, 1), filters=10, include_bias=True,
                 optimizer='adam', k_init='glorot_uniform'):
        self.activation = activation
        self.optimizer = optimizer
        self.kernel_initializer = k_init
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters = filters
        self.include_bias = include_bias
        self.weights = []
        self.neurons = []  # represents the output size

        self.inp_history = []
        self.ans_history = []

    def _reset_grads(self):
        self.weight_grads = np.zeros(self.weights.shape)
        if self.include_bias:
            self.bias_grads = np.zeros(self.filters)

    def _init_weights(self, inp_size, layer_indx):
        self.height = np.arange(0, inp_size[0] - self.kernel_size[0] + 1, self.stride[0])
        self.width = np.arange(0, inp_size[1] - self.kernel_size[1] + 1, self.stride[1])
        self.neurons = np.array([len(self.height), len(self.width), self.filters])  # output size

        self.weights = init_weights(self.kernel_size + (inp_size[2], self.filters), inp_size, self.neurons.prod(),
                                    layer_indx, self.kernel_initializer)
        self.w_optim = create_optimizer(self.weights.shape, self.optimizer)

        if self.include_bias:
            self.bias = np.zeros(self.filters)
            self.b_optim = create_optimizer(self.bias.shape, self.optimizer)
        self._reset_grads()
        return self.neurons

    def _feedforward(self, inp):
        """
        inp has shape (batch, h, w, d)
        out has shape (batch, h, w, f)
        """
        if inp.ndim == 3:
            inp = inp.reshape(inp.shape + (1,))

        inp = inp[:, :, :, :, np.newaxis]
        ans = np.zeros(np.append(inp.shape[0], self.neurons))
        for h_o, h in enumerate(self.height):
            for w_o, w in enumerate(self.width):
                inp1 = inp[:, h:h + self.kernel_size[0], w:w + self.kernel_size[1], :, :]
                ans1 = (inp1 * self.weights).sum(axis=(1, 2, 3))
                if self.include_bias:
                    ans1 += self.bias
                ans[:, h_o, w_o, :] = ans1

        self.inp_history.append(inp)
        self.ans_history.append(ans)
        return apply_activation(ans, self.activation)

    def _backpropagate(self, neuron_grads):
        """
         grads shape should be (batch, h, w, f)
         return shape (batch, h, w, f)
        """
        if neuron_grads.ndim == 2:
            neuron_grads = neuron_grads.reshape(np.append(neuron_grads.shape[0], self.neurons))
        neuron_grads *= apply_derivative(self.ans_history[-1], self.activation)

        inp_grads = np.zeros(self.inp_history[-1].shape[:-1])

        if self.include_bias:
            self.bias_grads += neuron_grads.mean(axis=0).sum(axis=(0, 1,))

        for h_o, h in enumerate(self.height):
            for w_o, w in enumerate(self.width):
                self.weight_grads += (
                        neuron_grads[:, h_o:h_o + 1, w_o:w_o + 1, np.newaxis, :] * self.inp_history[-1][:,
                                                                                   h:h + self.kernel_size[0],
                                                                                   w:w + self.kernel_size[1],
                                                                                   :, :]).mean(axis=0)
                inp_grads[:, h:h + self.kernel_size[0], w:w + self.kernel_size[1], :] += (
                        neuron_grads[:, h_o:h_o + 1, w_o:w_o + 1, np.newaxis, :] * self.weights).sum(axis=-1)
        self.inp_history.pop()
        self.ans_history.pop()

        return inp_grads

    def _apply_grads(self, lr):
        self.weights -= self.w_optim(self.weight_grads, lr)
        if self.include_bias:
            self.bias -= self.b_optim(self.bias_grads, lr)
        self._reset_grads()


class DenseLayer:
    def __init__(self, activation='linear', neurons=64, include_bias=True, k_init='glorot_uniform', optimizer='adam'):
        """
        activation can be: 'relu', 'sigmoid' everything else is treated as 'linear'
        """
        self.neurons = neurons
        self.kernel_initializer = k_init
        self.optimizer = optimizer
        self.include_bias = include_bias
        self.activation = activation
        self.weights = []
        if self.include_bias:
            self.bias = []

        self.inp_history = []
        self.ans_history = []
        self.cntr = 1

    def _reset_grads(self):
        self.weight_grads = np.zeros(self.weights.shape)
        if self.include_bias:
            self.bias_grads = np.zeros(self.neurons)

    def _init_weights(self, inp_size, layer_indx):
        if type(inp_size) is np.ndarray:
            inp_size = inp_size.prod()
        self.layer_indx = layer_indx

        self.weights = init_weights((inp_size, self.neurons), inp_size, self.neurons, layer_indx,
                                    self.kernel_initializer)
        self.w_optim = create_optimizer(self.weights.shape, self.optimizer)

        if self.include_bias:
            self.bias = np.zeros(self.neurons)
            self.b_optim = create_optimizer(self.bias.shape, self.optimizer)

        self._reset_grads()
        return self.neurons

    def _feedforward(self, inp):
        """
        inp is a 1D array representing the previous layer
        """
        if inp.ndim > 2:
            inp = inp.reshape(inp.shape[0], -1)

        ans = inp @ self.weights
        if self.include_bias:
            ans += self.bias

        self.inp_history.append(inp)
        self.ans_history.append(ans)
        return apply_activation(ans, self.activation)

    def _backpropagate(self, neuron_grads):
        neuron_grads *= apply_derivative(self.ans_history[-1], self.activation)

        g = self.inp_history[-1].reshape(self.inp_history[-1].shape + (1,)) @ \
            neuron_grads.reshape(neuron_grads.shape[0], 1, -1)
        self.weight_grads += g.mean(axis=0)
        self.weight_grads += self.weight_grads.mean(axis=0)
        if self.include_bias:
            self.bias_grads += neuron_grads.mean(axis=0)

        self.inp_history.pop()
        self.ans_history.pop()

        return neuron_grads @ self.weights.T

    def _apply_grads(self, lr):
        self.weights -= self.w_optim(self.weight_grads, lr)
        if self.include_bias:
            self.bias -= self.b_optim(self.bias_grads, lr)
        self._reset_grads()
        # this is used for saving autoencoder weights
        """if self.cntr % 10000 == 0 and self.layer_indx in [0,1,2,3]:
            with open(f'weights{self.layer_indx}_{self.cntr/10000}.npy', 'wb') as f:
                np.save(f, self.weights)
            with open(f'biases{self.layer_indx}_{self.cntr/10000}.npy', 'wb') as f:
                np.save(f, self.bias)
        self.cntr += 1"""


class MultiDense:
    def __init__(self, activation='softmax', out_cnt=2, neurons=38, include_bias=True, optimizer='adam',
                 k_init='glorot_uniform'):
        self.neurons = neurons
        self.out_cnt = out_cnt
        self.denses = [DenseLayer(activation, neurons,
                                  include_bias, k_init, optimizer) for _ in range(out_cnt)]

    def _reset_grads(self):
        for d in self.denses:
            d._reset_grads()

    def _init_weights(self, inp_size, layer_indx):
        for d in self.denses:
            d._init_weights(inp_size, layer_indx)
        self._reset_grads()
        return self.neurons * self.out_cnt

    def _feedforward(self, X):
        y = self.denses[0]._feedforward(X)[:, np.newaxis, :]
        for i in range(1, self.out_cnt):
            pred = self.denses[i]._feedforward(X)[:, np.newaxis, :]
            y = np.append(y, pred, axis=1)
        return np.array(y)

    def _backpropagate(self, neuron_grads):
        n_g = 0
        for ind, d in enumerate(self.denses):
            n_g += d._backpropagate(neuron_grads[:, ind])
        return n_g

    def _apply_grads(self, lr):
        for d in self.denses:
            d._apply_grads(lr)
        self._reset_grads()
