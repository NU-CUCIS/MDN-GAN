# Assemble scalable generator
import tensorflow as tf
import numpy as np
import cPickle as pickle
import matplotlib
import scipy.io as sio
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import filters

## change the values for z to generate different microstructure
zval = [-0.11577563,  0.07282215,  0.61534427,  0.27211975,  0.40787307,
       -0.90973462,  0.23367849,  0.40085908,  0.78126333]

def lrelu(x, alpha=0.2, name=None):
    return tf.nn.leaky_relu(x, alpha, name=name)
def relu(x, name=None):
    return tf.nn.relu(x, name=name)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # 1st hidden layer
		conv1 = tf.layers.conv2d_transpose(x, filter_num[0], [4, 4], strides=(2, 2), padding='same', name='conv1',
										   kernel_initializer=tf.constant_initializer(G_conv1_weights),
										   bias_initializer=tf.constant_initializer(G_conv1_biases))

		bn1 = tf.layers.batch_normalization(conv1, training=isTrain, name='bn1',
										    gamma_initializer=tf.constant_initializer(G_BN1_gamma),
										    beta_initializer=tf.constant_initializer(G_BN1_beta),
										    moving_mean_initializer=tf.constant_initializer(G_BN1_moving_mean),
										    moving_variance_initializer=tf.constant_initializer(G_BN1_moving_variance))

		relu1 = relu(bn1, name='relu1')

		# 2nd hidden layer
		conv2 = tf.layers.conv2d_transpose(relu1, filter_num[1], [4, 4], strides=(2, 2), padding='same', name='conv2',
										   kernel_initializer=tf.constant_initializer(G_conv2_weights),
										   bias_initializer=tf.constant_initializer(G_conv2_biases))

		bn2 = tf.layers.batch_normalization(conv2, training=isTrain, name='bn2',
										    gamma_initializer=tf.constant_initializer(G_BN2_gamma),
										    beta_initializer=tf.constant_initializer(G_BN2_beta),
										    moving_mean_initializer=tf.constant_initializer(G_BN2_moving_mean),
										    moving_variance_initializer=tf.constant_initializer(G_BN2_moving_variance))
		relu2 = relu(bn2, name='relu2')

		# 3rd hidden layer
		conv3 = tf.layers.conv2d_transpose(relu2, filter_num[2], [4, 4], strides=(2, 2), padding='same', name='conv3',
										   kernel_initializer=tf.constant_initializer(G_conv3_weights),
										   bias_initializer=tf.constant_initializer(G_conv3_biases))
		bn3 = tf.layers.batch_normalization(conv3, training=isTrain, name='bn3',
										    gamma_initializer=tf.constant_initializer(G_BN3_gamma),
										    beta_initializer=tf.constant_initializer(G_BN3_beta),
										    moving_mean_initializer=tf.constant_initializer(G_BN3_moving_mean),
										    moving_variance_initializer=tf.constant_initializer(G_BN3_moving_variance))
		relu3 = relu(bn3, name='relu3')

		# 4th hidden layer
		conv4 = tf.layers.conv2d_transpose(relu3, filter_num[3], [4, 4], strides=(2, 2), padding='same', name='conv4',
										   kernel_initializer=tf.constant_initializer(G_conv4_weights),
										   bias_initializer=tf.constant_initializer(G_conv4_biases))
		bn4 = tf.layers.batch_normalization(conv4, training=isTrain, name='bn4',
											gamma_initializer=tf.constant_initializer(G_BN4_gamma),
											beta_initializer=tf.constant_initializer(G_BN4_beta),
											moving_mean_initializer=tf.constant_initializer(G_BN4_moving_mean),
											moving_variance_initializer=tf.constant_initializer(G_BN4_moving_variance))
		relu4 = relu(bn4, name='relu4')

		conv5 = tf.layers.conv2d_transpose(relu4, 1, [4, 4], strides=(2, 2), padding='same', name='conv5',
										   kernel_initializer=tf.constant_initializer(G_conv5_weights),
										   bias_initializer=tf.constant_initializer(G_conv5_biases))
		o = tf.nn.tanh(conv5, name='o')
		return o
# loaG the weights
weight_path = './weights.pickle'

with open(weight_path, 'rb') as f:
	weights, biases, BNs = pickle.load(f)

G_conv1_weights = weights['G_conv1_weights']
G_conv2_weights = weights['G_conv2_weights']
G_conv3_weights = weights['G_conv3_weights']
G_conv4_weights = weights['G_conv4_weights']
G_conv5_weights = weights['G_conv5_weights']

G_conv1_biases = biases['G_conv1_bias']
G_conv2_biases = biases['G_conv2_bias']
G_conv3_biases = biases['G_conv3_bias']
G_conv4_biases = biases['G_conv4_bias']
G_conv5_biases = biases['G_conv5_bias']

G_BN1_gamma = BNs['G_BN1_gamma']
G_BN1_beta = BNs['G_BN1_beta']
G_BN1_moving_mean = BNs['G_BN1_moving_mean']
G_BN1_moving_variance = BNs['G_BN1_moving_variance']

G_BN2_gamma = BNs['G_BN2_gamma']
G_BN2_beta = BNs['G_BN2_beta']
G_BN2_moving_mean = BNs['G_BN2_moving_mean']
G_BN2_moving_variance = BNs['G_BN2_moving_variance']

G_BN3_gamma = BNs['G_BN3_gamma']
G_BN3_beta = BNs['G_BN3_beta']
G_BN3_moving_mean = BNs['G_BN3_moving_mean']
G_BN3_moving_variance = BNs['G_BN3_moving_variance']

G_BN4_gamma = BNs['G_BN4_gamma']
G_BN4_beta = BNs['G_BN4_beta']
G_BN4_moving_mean = BNs['G_BN4_moving_mean']
G_BN4_moving_variance = BNs['G_BN4_moving_variance']

print "====> Weights loaded!"
z_dim = (1, 3, 3, 1)
filter_num = [128, 64, 32, 16] 
z = tf.placeholder(tf.float32, z_dim, name='z')
isTrain = tf.placeholder(dtype=tf.bool, name='isTrain')
X = generator(z, isTrain)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
z_ = np.reshape(np.array(zval), z_dim)
image = sess.run([X], feed_dict={z:z_, isTrain:False})[0]
plt.figure(0)
gray=image[0, :, :, 0]
val = filters.threshold_otsu(gray)
plt.imshow(gray>val, cmap='gray')
plt.savefig('example.jpg')
print "====> image generated!"

