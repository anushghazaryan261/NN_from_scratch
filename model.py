import numpy as np

from NN_from_scratch.layers import Dropout, BatchNormalization


# np.random.seed(2)

def calculate_loss(y_true, y_pred, loss='mse'):
    """
    :return: (gradients, loss)
    """
    n_g = 0
    l = 0
    epsilon = 1e-8
    y_pred += epsilon

    if loss == 'mse':
        n_g = 2 * (y_pred - y_true) / y_true.shape[1]
        l = np.mean((y_pred - y_true) ** 2)
    elif loss == 'cos':
        A = np.linalg.norm(y_true, axis=1).reshape(-1, 1)
        B = np.linalg.norm(y_pred, axis=1).reshape(-1, 1)
        l = (y_true[:, np.newaxis] @ y_pred[:, :, np.newaxis]).reshape(-1, 1) / (A * B)
        n_g = -y_true / (A * B) + y_pred * l / (A**2)
        l = 1-np.mean(l)
    elif loss == 'bce':
        n_g = (-y_true / (y_pred + epsilon) + (1 - y_true) / (1 - y_pred + epsilon)) / 2
        l = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)) / 2
    elif loss == 'cce':
        n_g = y_pred - y_true
        if y_true.ndim > 2:
            l -= np.mean(np.sum(y_true * np.log(y_pred), axis=(-2, -1)))
        else:
            l = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    return np.array(n_g), l


class Model:
    def __init__(self, layer_arr, loss='mse', optimizer='adam'):
        """
        loss: mse for regression, bce and cce for binary/categorical cross-enropy
        """
        self.loss = loss
        prev_n = layer_arr[0].neurons
        norm_cnt = 0
        for ind, layer in enumerate(layer_arr[1:]):
            layer.optimizer = optimizer
            if type(layer) in [Dropout, BatchNormalization]:
                layer.neurons = prev_n
                norm_cnt += 1
            else:
                prev_n = layer._init_weights(prev_n, ind - norm_cnt)
                # print(layer.weights.shape)
        self.layers = layer_arr

    def predict(self, X, fitting=False):
        ans = X
        if X.ndim == 1:
            ans = ans.reshape(1, -1)

        for l in self.layers[1:]:
            if type(l) is Dropout and not fitting:
                continue
            else:
                ans = l._feedforward(ans)
        return np.array(ans)

    def fit(self, X, Y, epochs=50, batch_size=32, lr=0.001):
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        scaling_factor = X.shape[0] / batch_size  # num of batches

        for _ in range(epochs):
            print('epoch', _, end=' ')
            losses = 0

            indxs = np.arange(X.shape[0])
            np.random.shuffle(indxs)
            for j in range(0, X.shape[0], batch_size):
                indx = indxs[j:j + batch_size]
                ans = self.predict(X[indx], fitting=True)

                n_g, l = calculate_loss(Y[indx], ans, self.loss)
                neuron_grads = n_g
                losses += l

                for i in range(len(self.layers) - 1, 0, -1):
                    neuron_grads = self.layers[i]._backpropagate(neuron_grads)
                    if type(self.layers[i]) not in [Dropout, BatchNormalization]:
                        self.layers[i]._apply_grads(lr)  # / scaling_factor)
            print('loss', losses / scaling_factor)

    def fit_rnn(self, X, epochs=10000, batch_size=32, sequence_len=200, lr=0.001):
        conv_sizes = np.array([i.shape[0] - sequence_len for i in X])  # num of start points in conversations
        inp_letter_cnt = self.layers[0].neurons[0]

        print(conv_sizes)
        batch_seq = np.random.choice(len(conv_sizes), (epochs, batch_size),
                                     p=conv_sizes / conv_sizes.sum())  # conversation indexes of each batch
        x = np.zeros((batch_size, sequence_len, len(X[0][0])))  # batch of data we will be training on

        for _, b_indxs in enumerate(batch_seq):
            print('epoch', _, end=' ')
            batch_grads = []
            losses = 0
            for i1, i in enumerate(b_indxs):
                ind_start = np.random.randint(0, conv_sizes[i])
                x[i1] = X[i][ind_start:ind_start + sequence_len]

            for i in range(sequence_len - inp_letter_cnt):
                ans = self.predict(x[:, i:i + inp_letter_cnt], fitting=True)
                n_g, l = calculate_loss(x[:, i + 1], ans, self.loss)
                batch_grads.append(n_g / (sequence_len - inp_letter_cnt))
                losses += l

            for neuron_grads in batch_grads[::-1]:
                for i in range(len(self.layers) - 1, 0, -1):
                    neuron_grads = self.layers[i]._backpropagate(neuron_grads)

            for l in self.layers[1:]:
                if type(l) not in [Dropout, BatchNormalization]:
                    l._apply_grads(lr)
            print('loss', losses / (sequence_len - inp_letter_cnt))
