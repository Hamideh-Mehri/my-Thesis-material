import tensorflow as tf
from DeepLayer import ConvTwoTranspose, DenseGan, ConvTwo_v2
import functools
from functools import reduce




def load_network(datasource, arch, latent_dim,batch_size, *initializer):
    networks = {
        
        'dcgan_mnist': lambda: DCGAN_mnist(datasource, latent_dim,batch_size, *initializer),
        'dcgan_cifar10': lambda: DCGAN_cifar10(datasource, latent_dim,batch_size, *initializer)
        
    }
    return networks[arch]()


class DCGAN_mnist(object):
   def __init__(self, datasource, latent_dim, batch_size, *initializer):
      self.datasource = datasource
      self.latentdim = latent_dim
      self.batch_size = batch_size
      self.convinitializer = initializer[0]
      self.denseinitializer = initializer[1]
   def construct_weights_gen(self, trainable=True):
       weight = {}
       if self.datasource == 'mnist' or self.datasource == 'fashion-mnist':
          DeconvInput = 7 * 7 * 128
       weight['w1_fc'] = tf.Variable(self.denseinitializer(shape=(self.latentdim, DeconvInput)),name='w1', trainable=trainable)
       weight['w2_Dconv'] = tf.Variable(self.convinitializer(shape=(5,5,128,128)), name='w2', trainable=trainable)
       weight['w3_Dconv'] = tf.Variable(self.convinitializer(shape=(5,5,64,128)), name='w3', trainable=trainable)
       weight['w4_Dconv'] = tf.Variable(self.convinitializer(shape=(5,5,32,64)), name='w4', trainable=trainable)
       weight['w5_Dconv'] = tf.Variable(self.convinitializer(shape=(5,5,1,32)), name='w5', trainable=trainable)
       return weight
   def forward_pass_gen(self, weights, inputs):
       self.batch_size = inputs.shape[0]
       inputs = tf.matmul(inputs, weights['w1_fc'])
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.reshape(inputs, [-1,7,7,128])
       inputs = tf.nn.conv2d_transpose(inputs, weights['w2_Dconv'], [self.batch_size,14,14,128], [1,2,2,1], 'SAME')
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.nn.conv2d_transpose(inputs, weights['w3_Dconv'], [self.batch_size,28,28,64], [1,2,2,1], 'SAME')
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.nn.conv2d_transpose(inputs, weights['w4_Dconv'], [self.batch_size,28,28,32], [1,1,1,1], 'SAME')
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.nn.conv2d_transpose(inputs, weights['w5_Dconv'], [self.batch_size,28,28,1], [1,1,1,1], 'SAME')
       inputs = tf.nn.tanh(inputs)
       return inputs
   def make_layers_gen(self, weights, masks):
       batch_size = self.batch_size
       inp = tf.keras.layers.Input(shape= (100,))
       fc1 = DenseGan(weights['w1_fc'], masks['w1_fc'],tf.keras.activations.linear, 'w1_fc_gen', 'w1_mask_gen')(inp)
       batchnorm1 = tf.keras.layers.BatchNormalization()(fc1)
       Rel1 = tf.keras.layers.ReLU()(batchnorm1)
       DeconvInput = tf.keras.layers.Reshape((7,7,128))(Rel1)

       Deconv2 = ConvTwoTranspose(weights['w2_Dconv'],masks['w2_Dconv'],[batch_size,14,14,128],[1,2,2,1],'SAME', 'w2_Dconv_gen', 'w2_mask_gen')(DeconvInput)
       batchnorm2 = tf.keras.layers.BatchNormalization()(Deconv2)
       Rel2 = tf.keras.layers.ReLU()(batchnorm2)

       Deconv3 = ConvTwoTranspose(weights['w3_Dconv'], masks['w3_Dconv'], [batch_size,28,28,64],[1,2,2,1],'SAME', 'w3_Dconv_gen', 'w3_mask_gen')(Rel2)
       batchnorm3 = tf.keras.layers.BatchNormalization()(Deconv3)
       Rel3 = tf.keras.layers.ReLU()(batchnorm3)

       Deconv4 = ConvTwoTranspose(weights['w4_Dconv'], masks['w4_Dconv'], [batch_size,28,28,32],[1,1,1,1],'SAME', 'w4_Dconv_gen', 'w4_mask_gen')(Rel3)
       batchnorm4 = tf.keras.layers.BatchNormalization()(Deconv4)
       Rel4 = tf.keras.layers.ReLU()(batchnorm4)

       Deconv5 = ConvTwoTranspose(weights['w5_Dconv'], masks['w5_Dconv'], [batch_size,28,28,1],[1,1,1,1],'SAME', 'w5_Dconv_gen', 'w5_mask_gen')(Rel4)
       out = tf.keras.activations.tanh(Deconv5)

       model = tf.keras.models.Model(inp, out) 
       return model

   def construct_weights_disc(self, trainable=True):
       weight = {}
       weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(5,5,1,32)), name='w1', trainable=trainable)
       weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(5,5,32,64)), name='w2', trainable=trainable)
       weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(5,5,64,128)), name='w3', trainable=trainable)
       weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(5,5,128,256)), name='w4', trainable=trainable)
       weight['w5_fc'] = tf.Variable(self.denseinitializer(shape=(4096, 1)), name='w5', trainable=trainable)
       return weight
   def forward_pass_disc(self, weights, inputs):
       inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
       inputs = tf.nn.conv2d(inputs, weights['w1_conv'], [1,2,2,1], 'SAME')
       inputs = inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
       inputs = tf.nn.conv2d(inputs, weights['w2_conv'], [1,2,2,1], 'SAME')
       inputs = inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
       inputs = tf.nn.conv2d(inputs, weights['w3_conv'], [1,2,2,1], 'SAME')
       inputs = inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
       inputs = tf.nn.conv2d(inputs, weights['w4_conv'], [1,1,1,1], 'SAME')
       inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
       inputs = tf.matmul(inputs, weights['w5_fc'])
       return inputs

       
   def make_layers_disc(self, weights, masks):
       inp = tf.keras.layers.Input(shape=(28,28,1))
       Rel1 = tf.keras.layers.LeakyReLU(alpha=0.2)(inp)

       conv1 = ConvTwo_v2(weights['w1_conv'], masks['w1_conv'], [1,2,2,1], 'SAME', 'w1_conv_disc', 'w1_mask_disc')(Rel1)
       Rel2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)

       conv2 = ConvTwo_v2(weights['w2_conv'], masks['w2_conv'], [1,2,2,1], 'SAME', 'w2_conv_disc', 'w2_mask_disc')(Rel2)
       Rel3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2)

       conv3 = ConvTwo_v2(weights['w3_conv'], masks['w3_conv'], [1,2,2,1], 'SAME', 'w3_conv_disc', 'w3_mask_disc')(Rel3)
       Rel4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3)

       conv4 = ConvTwo_v2(weights['w4_conv'],masks['w4_conv'],[1,1,1,1], 'SAME', 'w4_conv_disc', 'w4_mask_disc')(Rel4)
       
       flatten = tf.keras.layers.Flatten()(conv4)

       out = DenseGan(weights['w5_fc'], masks['w5_fc'], tf.keras.activations.linear, 'w5_fc_disc', 'w5_mask_disc')(flatten)

       model = tf.keras.models.Model(inp, out) 
       return model
