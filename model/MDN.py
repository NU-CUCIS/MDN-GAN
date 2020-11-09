# from MDN_v2 import *
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
# import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Activation, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pickle
import h5py 

from sklearn.utils import shuffle

# N_HIDDEN = 15 # number of nueron in hidden layer
N_MIXES = 40 # number of mixture gaussion
OUTPUT_DIMS = 4 # output dimension
NUM_SAMPLE = 30 # number of sampled points for each input y
y_val = 0.55

def softmax(w, t=1.0):
    """Softmax function for a list or numpy array of logits. Also adjusts temperature.
    Arguments:
    w -- a list or numpy array of logits
    Keyword arguments:
    t -- the temperature for to adjust the distribution (default 1.0)
    """
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist

def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent NaN in loss."""
    return (K.elu(x) + 1 + 1e-8)

def loss_func(y_true, y_pred):
    # Reshape inputs in case this is used in a TimeDistribued layer
    num_mixes = N_MIXES
    output_dim = OUTPUT_DIMS
    y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
    y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
    # Split the inputs into paramaters
    out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                     num_mixes * output_dim,
                                                                     num_mixes],
                                         axis=-1, name='mdn_coef_split')
    # Construct the mixture models
    cat = tfd.Categorical(logits=out_pi)
    component_splits = [output_dim] * num_mixes
    mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
    sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
    coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
            in zip(mus, sigs)]
    mixture = tfd.Mixture(cat=cat, components=coll)
    loss = mixture.log_prob(y_true)
    loss = tf.negative(loss)
    loss = tf.reduce_mean(loss)
    return loss

def sample_from_output(params, output_dim, num_mixes, num_sample, temp=1.0, sigma_temp=1.0):
    """Sample from an MDN output with temperature adjustment.
    This calculation is done outside of the Keras model using
    Numpy.
    
    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    Keyword arguments:
    temp -- the temperature for sampling between mixture components (default 1.0)
    sigma_temp -- the temperature for sampling from the normal distribution (default 1.0)
    Returns:
    One sample from the the mixture model.
    """
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    # m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m*output_dim:(m+1)*output_dim]
    sig_vector = sigs[m*output_dim:(m+1)*output_dim] * sigma_temp  # adjust for temperature
    cov_matrix = np.identity(output_dim) * sig_vector
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, num_sample)
    return sample

def split_mixture_params(params, output_dim, num_mixes):
    """Splits up an array of mixture parameters into mus, sigmas, and pis
    depending on the number of mixtures and output dimension.
    Arguments:
    params -- the parameters of the mixture model
    output_dim -- the dimension of the normal models in the mixture model
    num_mixes -- the number of mixtures represented
    """
    mus = params[:num_mixes*output_dim]
    sigs = params[num_mixes*output_dim:2*num_mixes*output_dim]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits
# model = load_model('my_model.h5', custom_objects={'MDN': MDN, 'loss_func':get_mixture_loss_func(OUTPUT_DIMS,N_MIXES)})
def build_model():
    inp = Input(shape=(1,))
    x = Dense(16)(inp)
    # x = Dense(16, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    concate = concatenate([x, inp])
    x = Dense(16)(concate)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    concate = concatenate([x, concate])
    x = Dense(16)(concate)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    concate = concatenate([x, concate])
    x = Dense(16)(concate)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(16)(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(64)(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(128)(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(128)(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(256)(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(256)(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(512, activation='relu')(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # concate = concatenate([x, concate])
    # x = Dense(1024, activation='relu')(concate)
    
    # concate = concatenate([x, inp])
    # x = Dense(512)(concate)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # concate = concatenate([x, concate])
    # x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(concate)

    mdn_mus = Dense(N_MIXES*OUTPUT_DIMS)(x)
    mdn_sigmas = Dense(N_MIXES*OUTPUT_DIMS, activation=elu_plus_one_plus_epsilon)(x)  # mix*output vals exp activation
    mdn_pi = Dense(N_MIXES)(x)  # mix vals, logits
    output = concatenate([mdn_mus, mdn_sigmas, mdn_pi])
    model = tf.keras.Model(inputs=inp, outputs=output)
    model.compile(loss=loss_func, optimizer=tf.keras.optimizers.Adam(), metrics=[loss_func])
    return model


model = build_model()
model.summary()
model.load_weights('./MDN_weights.h5')
## Sample on some test data:
x_test = np.array([y_val])
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

# Make predictions from the model
y_test = model.predict(x_test)
# y_test contains parameters for distributions, not actual points on the graph.
# To find points on the graph, we need to sample from each distribution.
print (y_test.shape)

# Split up the mixture parameters (for future fun)
mus = np.apply_along_axis((lambda a: a[:N_MIXES*OUTPUT_DIMS]),1, y_test)
sigs = np.apply_along_axis((lambda a: a[N_MIXES*OUTPUT_DIMS:2*N_MIXES*OUTPUT_DIMS]),1, y_test)
pis = np.apply_along_axis((lambda a: softmax(a[-N_MIXES:])),1, y_test)

# Sample from the predicted distributions
y_samples = np.apply_along_axis(sample_from_output, 1, y_test, OUTPUT_DIMS, N_MIXES, NUM_SAMPLE)
y_samples = np.squeeze(y_samples, axis=0)
print (y_samples.shape)