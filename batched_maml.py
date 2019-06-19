import jax
import jax.experimental.stax as stax
import jax.experimental.optimizers as opt
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

###############################################
########## SETTING UP NEURAL NETWORK ##########
###############################################

net_init, net_apply = stax.serial(
    stax.Dense(40), stax.Relu,
    stax.Dense(40), stax.Relu,
    stax.Dense(1)
)

input_shape = (-1, 1,)
output_shape, net_params = net_init(input_shape)

def loss(params, inputs, targets):
    predictions = net_apply(params, inputs)
    return jax.numpy.mean((targets - predictions)**2)

xrange_inputs = jax.numpy.linspace((-5, 5, 100).reshape((100, 1)))
targets = jax.numpy.sin(xrange_inputs)
predictions = jax.vmap(partial(net_apply, net_params))(xrange_inputs)
losses = jax.vmap(partial(loss, net_params))(xrange_inputs, targets)

####################################################
########## PLOTTING UNINITIALIZED NETWORK ##########
####################################################

plt.plot(xrange_inputs, predictions, label='Prediction')
plt.plot(xrange_inputs, losses, label='Loss')
plt.plot(xrange_inputs, targets, label='Target')
plt.legend()

#################################################
########## TRAIN WITH GRADIENT DESCENT ##########
#################################################

opt_init, opt_update = opt.adam(step_size=1e-2)
opt_state = opt_init(net_params)

@jax.jit
def step(i, opt_state, x1, y1):
    p = opt.get_params(opt_state)
    g = jax.grad(loss)(p, x1, y1)
    return opt_update(i, g, opt_state)

for i in range(100):
    opt_state = step(i, opt_state, xrange_inputs, targets)
net_params = opt.get_params(opt_state)

#################################################
########## IMPLEMENTING MAML ALGORITHM ##########
#################################################

def inner_update(p, x1, y1, alpha=0.1):
    grads = jax.grad(loss)(p, x1, y1)
    inner_sgd_fn = lambda g, state: (state - alpha * g)
    return jax.tree_util.tree_multimap(inner_sgd_fn, grads, p)

def maml_loss(p, x1, y1, x2, y2):
    p2 = inner_update(p, x1, y1)
    return loss(p2, x2, y2)

opt_init, opt_update = opt.adam(step_size=1e-3)
output_shape, net_params = net_init(input_shape)
opt_state = opt_init(net_params)
@jax.jit
def step(i, opt_state, x1, y1, x2, y2):
    p = opt.get_params(opt_state)
    g = jax.grad(maml_loss)(p, x1, y1, x2, y2)
    l = maml_loss(p, x1, y1, x2, y2)
    return opt_update(i, g, opt_state), l

K = 20
np_maml_loss = []

for i in range(20000):
    A = np.random.uniform(low=0.1, high=0.5)
    phase = np.random.uniform(low=0., high=np.pi)
    x1 = np.random.uniform(low=-5.0, high=5.0, size=(K, 1))
    y1 = A * np.sin(x1 + phase)

    x2 = np.random.uniform(low=-5.0, high=5.0, size=(K, 1))
    y2 = A * np.sin(x2 + phase)
    opt_state, l = step(i, opt_state, x1, y1, x2, y2)
    np_maml_loss.append(l)
    if i % 1000 == 0:
        print(i)

net_params = opt.get_params(opt_state)
targets = jax.numpy.sin(xrange_inputs)
predictions = jax.vmap(partial(net_apply, net_params))(xrange_inputs)

plt.plot(xrange_inputs, predictions, label='Predictions')
plt.plot(xrange_inputs, targets, label="Targets")

x1 = np.random.uniform(low=-6.0, high=5.0, size=(k, 1))
y1 = 1. * jax.numpy.sin(x1, 0.)

for i in range(1, 5):
    net_params = inner_update(net_params, x1, y1)
    predictions = jax.vmap(partial(net_apply, net_params))(xrange_inputs)
    plt.plot(xrange_inputs, predictions, label='{}-shot predictions'.format(i))
plt.legend()

##########################################################
########## BATCHING MAML GRADIENTS ACROSS TASKS ##########
##########################################################

def batch_maml_loss(p, x1_b, y1_b, x2_b, y2_b):
    task_losses = jax.vmap(partial(maml_loss, p))(x1_b, y1_b, x2_b, y2_b)
    return jax.numpy.mean(task_losses)

def sample_tasks(outer_batch_size, inner_batch_size):
    As = []
    phases = []
    for _ in range(outer_batch_size):
        As.append(np.random.uniform(low=0.1, high=0.5))
        phases.append(np.random.uniform(low=0., high=np.pi))
    def get_batch():
        xs, ys = [], []
        for A, phase in zip(As, phases):
            x = np.random.uniform(low=-5., high=5., size=(inner_batch_size, 1))
            y = A * np.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return jax.numpy.stack(xs), jax.numpy.stack(ys)
    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2

opt_init, opt_update = opt.adam(step_size=1e-3)
output_shape, net_params = net_init(input_shape)
opt_state = opt_init(net_params)

np_batched_maml_loss = []
K = 20

for i in range(20000):
    x1_b, y1_b, x2_b, y2_b = sample_tasks(4, K)
    opt_state, l = step(i, opt_state, x1_b, y1_b, x2_b, y2_b)
    np_batched_maml_loss.append(l)
    if i % 1000 == 0:
        print(i)

net_params = opt.get_params(opt_state)
