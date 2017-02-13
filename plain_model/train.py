import os
import sys
import utils
import lstm_plain
import numpy as np
import progressbar

def binarize(x, num_words):
    out = np.zeros((num_words, 1))
    out[x] = 1
    return out

def cost_function(W, x, c_t, h_t, d, num_words):
    x_t, y_t = x[0], x[1]
    x_t = binarize(x_t, num_words)
    y_t = binarize(y_t, num_words)
    # c_t, h_t = x_t, x_t
    cost, grad, h_t, c_t = lstm_plain.cost_function(W, x_t, y_t, h_t, c_t)
    bar = progressbar.ProgressBar()
    for i in bar(xrange(0, len(x)-1)):
        x_t, y_t = x[i], x[i+1]
        x_t = binarize(x_t, num_words)
        y_t = binarize(y_t, num_words)
        t_cost, t_grad, h_t, c_t = lstm_plain.cost_function(W, x_t, y_t, h_t, c_t)
        cost += t_cost
        grad += t_grad
    return cost, grad, h_t, c_t

def gradient_descent(input_size, output_size, memory_size,
                     cost_function, inputs, alpha, n_iter, W=None):
    if W is None:
        W = utils.init_weights(input_size, memory_size, output_size)
    c_t = np.random.rand(d, 1)
    h_t = np.random.rand(d, 1)

    eps = 1e-8
    beta1 = 0.9
    beta2 = 0.999

    m = np.zeros(W.shape)
    v = np.zeros(W.shape)
    n_cases = inputs.shape[-1]
    next_case = 0
    step = 447
    try:
        for iter in xrange(0, n_iter):
            xt = inputs[next_case:next_case+step]
            cost, grad, h_t, c_t = cost_function(W, xt, c_t, h_t)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_c = m / (1 - beta1**(iter + 1))
            v_c = v / (1 - beta2**(iter + 1))
            W += - alpha * m_c / (np.sqrt(v_c) + eps)
            print 'Iteration %d | Case # %d | Cost: %g\n' % (iter + 1, next_case, cost)
            next_case = (next_case + step) % n_cases
            if iter % 10 == 0:
                np.savez('W_partial', W)
            # next_case += step
                # alpha /= 2.0
    except Exception as e:
        # print e
        np.savez('W_partial', W)
    return W

if __name__ == '__main__':
    D = np.load('dictionary.npz')
    data = D['data']
    num_words = len(D['vec_repr'].item())
    n, m = num_words, num_words
    d = 300
    J = lambda W, x, y, z: cost_function(W, x, y, z, d, num_words)
    W = gradient_descent(n, m, d, J, data, 0.001, 1000)
    np.savez('weights', W)
