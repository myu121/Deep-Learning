from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras.optimizers import *
import numpy as np
import copy
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
m = max(x_train.flatten())
x_train = x_train/m
x_test = x_test/m
#y_train = y_train/9
y_train = y_train.flatten()
#y_test = y_test/9
y_test = y_test.flatten()
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x = np.empty([len(x_train),3072])
for i in range(len(x_train)):
    x[i] = x_train[i].flatten()
x_train = x

x = np.empty([len(x_test),3072])
for i in range(len(x_test)):
    x[i] = x_test[i].flatten()
x_test = x


model = Sequential([
    Dense(1024, input_dim=3072, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=SGD_new(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


class SGD_new(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, learning_rate=0.01, momentum=0.,
                 nesterov=False, tau=300, eps_tau=0.0001, **kwargs):
        #learning_rate = kwargs.pop('lr', learning_rate)
        self.learning_rate = learning_rate
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.tau = tau
        self.eps_tau = eps_tau
        super(SGD_new, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(self.initial_decay, name='decay')
        self.nesterov = nesterov

    #@interfaces.legacy_get_updates_support
    #@K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        #lr = self.learning_rate
        #if self.initial_decay > 0:
        #    lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
        #                                              K.dtype(self.decay))))
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables()) #execute init_op
            #print the random values that we sample
            it = sess.run(self.iterations)
        #lr = self.learning_rate
        if it <= self.tau:
            lr = (self.tau-it)/self.tau*self.learning_rate + it/self.tau*self.eps_tau
        else:
            lr = self.eps_tau
        
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape, name='moment_' + str(i))
                   for (i, shape) in enumerate(shapes)]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




