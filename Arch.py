import tensorflow as tf
from DeepLayer import ConvTwo, Dense, ConvTwo_v2
import functools
from functools import reduce



def load_network(datasource, arch, num_classes, *initializer):
    networks = {
        #'lenet300': lambda: LeNet300(),
        #'lenet5': lambda: LeNet5(),
        'alexnet-v1': lambda: AlexNet(datasource, num_classes, *initializer, k=1),
        'alexnet-v2': lambda: AlexNet(datasource, num_classes, *initializer, k=2),
        'vgg-11': lambda: VGG(datasource, num_classes, *initializer, version=11),
        'vgg-13': lambda: VGG(datasource, num_classes, *initializer, version=13),
        'vgg-16': lambda: VGG(datasource, num_classes, *initializer, version=16),
        'vgg-19': lambda: VGG(datasource, num_classes, *initializer, version=19),
        'resnet-18': lambda: ResNet18(datasource, num_classes, *initializer),
        'resnet-18-v2': lambda: ResNet18_v2(datasource, num_classes, *initializer),
        'resnet-18-v3': lambda: ResNet18_v3(datasource, num_classes, *initializer)
        
    }
    return networks[arch]()

class AlexNet(object):
   def __init__(self,datasource, num_classes, *initializer, k):
      self.num_classes = num_classes
      self.k = k
      self.datasource = datasource
      self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
      self.convinitializer = initializer[0]
      self.denseinitializer = initializer[1]


   def construct_weights(self,trainable=True, initializer1=tf.initializers.variance_scaling(), initializer2=tf.initializers.variance_scaling()):
       tf.random.set_seed(1234)
       weights={}
       weights['w1_conv'] = tf.Variable(self.convinitializer(shape=(11,11,3,96)), name='w1', trainable=trainable)
       weights['w2_conv'] = tf.Variable(self.convinitializer(shape=(5,5,96,256)), name='w2', trainable=trainable)
       weights['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,384)), name='w3', trainable=trainable)
       weights['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,384,384)),name='w4', trainable=trainable)
       weights['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,384,256)),name='w5', trainable=trainable)
       weights['w6_fc'] = tf.Variable(self.denseinitializer(shape=(256, 1024*self.k)),name='w6', trainable=trainable)
       weights['w7_fc'] = tf.Variable(self.denseinitializer(shape=(1024*self.k, 1024*self.k)),name='w7', trainable=trainable)
       weights['w8_fc'] = tf.Variable(self.denseinitializer(shape=(1024*self.k, self.num_classes)),name='w8', trainable=trainable)

       weights['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(96,)), name='b1', trainable=trainable)
       weights['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b2', trainable=trainable)
       weights['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(384,)), name='b3', trainable=trainable)
       weights['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(384,)), name='b4', trainable=trainable)
       weights['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b5', trainable=trainable)
       weights['b6_fc'] = tf.Variable(tf.zeros_initializer()(shape=(1024*self.k,)), name='b6', trainable=trainable)
       weights['b7_fc'] = tf.Variable(tf.zeros_initializer()(shape=(1024*self.k,)), name='b7', trainable=trainable)
       weights['b8_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b8', trainable=trainable)
       return weights

   def forward_pass(self,weights, inputs):
       init_stride = 4 if self.datasource == 'tiny-imagenet' else 2
       inputs = tf.nn.conv2d(inputs, weights['w1_conv'], [1,init_stride,init_stride,1], 'SAME') + weights['b1_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.nn.conv2d(inputs, weights['w2_conv'], [1, 2, 2, 1], 'SAME') + weights['b2_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.nn.conv2d(inputs, weights['w3_conv'], [1, 2, 2, 1], 'SAME') + weights['b3_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.nn.conv2d(inputs, weights['w4_conv'], [1, 2, 2, 1], 'SAME') + weights['b4_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.nn.conv2d(inputs, weights['w5_conv'], [1, 2, 2, 1], 'SAME') + weights['b5_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
       inputs = tf.matmul(inputs, weights['w6_fc']) + weights['b6_fc']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.matmul(inputs, weights['w7_fc']) + weights['b7_fc']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)
       inputs = tf.matmul(inputs, weights['w8_fc']) + weights['b8_fc'] # logits
       return inputs
    
   def make_layers(self, final_w):
       init_stride = 4 if self.datasource == 'tiny-imagenet' else 2
       shape = tuple(self.input_dims)
       inp = tf.keras.layers.Input(shape= shape)
       conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'], [1,init_stride,init_stride,1],'SAME')(inp)
       batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
       Rel1 = tf.keras.layers.ReLU()(batchnormal1)

       conv2 = ConvTwo(final_w['w2_conv'], final_w['b2_conv'], [1,2,2,1],'SAME')(Rel1)
       batchnormal2 = tf.keras.layers.BatchNormalization()(conv2)
       Rel2 = tf.keras.layers.ReLU()(batchnormal2)

       conv3 = ConvTwo(final_w['w3_conv'], final_w['b3_conv'], [1,2,2,1],'SAME')(Rel2)
       batchnormal3 = tf.keras.layers.BatchNormalization()(conv3)
       Rel3 = tf.keras.layers.ReLU()(batchnormal3)

       conv4 = ConvTwo(final_w['w4_conv'], final_w['b4_conv'], [1,2,2,1],'SAME')(Rel3)
       batchnormal4 = tf.keras.layers.BatchNormalization()(conv4)
       Rel4 = tf.keras.layers.ReLU()(batchnormal4)

       conv5 = ConvTwo(final_w['w5_conv'], final_w['b5_conv'], [1,2,2,1],'SAME')(Rel4)
       batchnormal5 = tf.keras.layers.BatchNormalization()(conv5)
       Rel5 = tf.keras.layers.ReLU()(batchnormal5)

       flatten = tf.keras.layers.Flatten()(Rel5)

       fc1 = Dense(final_w['w6_fc'], final_w['b6_fc'], tf.keras.activations.linear)(flatten)
       batchnormal_fc1 = tf.keras.layers.BatchNormalization()(fc1)
       Rel_fc1 = tf.keras.layers.ReLU()(batchnormal_fc1)

       fc2 = Dense(final_w['w7_fc'], final_w['b7_fc'], tf.keras.activations.linear)(Rel_fc1)
       batchnormal_fc2 = tf.keras.layers.BatchNormalization()(fc2)
       Rel_fc2 = tf.keras.layers.ReLU()(batchnormal_fc2)

       fc3 = Dense(final_w['w8_fc'], final_w['b8_fc'], tf.keras.activations.linear)(Rel_fc2)

       model = tf.keras.models.Model(inp, fc3)
       return model


class VGG(object):
  def __init__(self, datasource, num_classes, *initializer, version):
    self.num_classes = num_classes
    self.version = version
    self.datasource = datasource
    self.name = 'VGG-{}'.format(version)
    self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3]
    self.convinitializer = initializer[0]
    self.denseinitializer = initializer[1]
  
  def construct_weights(self, trainable=True):
    weight = {}
    if self.version == 11:
      weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(3,3,3,64)), name='w1', trainable=trainable)
      weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,128)), name='w2', trainable=trainable)
      weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,256)), name='w3', trainable=trainable)
      weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w4', trainable=trainable)
      weight['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,512)), name='w5', trainable=trainable)
      weight['w6_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w6', trainable=trainable)
      weight['w7_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w7', trainable=trainable)
      weight['w8_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w8', trainable=trainable)
      weight['w9_fc'] = tf.Variable(self.denseinitializer(shape=(512, self.num_classes)),name='w9', trainable=trainable)

      weight['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b1', trainable=trainable)
      weight['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b2', trainable=trainable)
      weight['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b3', trainable=trainable)
      weight['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b4', trainable=trainable)
      weight['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b5', trainable=trainable)
      weight['b6_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b6', trainable=trainable)
      weight['b7_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b7', trainable=trainable)
      weight['b8_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b8', trainable=trainable)
      weight['b9_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b9', trainable=trainable)
      return weight
    if self.version == 13:
      weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(3,3,3,64)), name='w1', trainable=trainable)
      weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w2', trainable=trainable)
      weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,128)), name='w3', trainable=trainable)
      weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,128)), name='w4', trainable=trainable)
      weight['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,256)), name='w5', trainable=trainable)
      weight['w6_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w6', trainable=trainable)
      weight['w7_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,512)), name='w7', trainable=trainable)
      weight['w8_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w8', trainable=trainable)
      weight['w9_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w9', trainable=trainable)
      weight['w10_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w10', trainable=trainable)
      weight['w11_fc'] = tf.Variable(self.denseinitializer(shape=(512, self.num_classes)),name='w11', trainable=trainable)

      weight['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b1', trainable=trainable)
      weight['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b2', trainable=trainable)
      weight['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b3', trainable=trainable)
      weight['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b4', trainable=trainable)
      weight['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b5', trainable=trainable)
      weight['b6_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b6', trainable=trainable)
      weight['b7_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b7', trainable=trainable)
      weight['b8_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b8', trainable=trainable)
      weight['b9_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b9', trainable=trainable)
      weight['b10_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b10', trainable=trainable)
      weight['b11_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b11', trainable=trainable)
      return weight
    if self.version == 16:
      tf.random.set_seed(1234)
      weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(3,3,3,64)), name='w1', trainable=trainable)
      weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w2', trainable=trainable)
      weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,128)), name='w3', trainable=trainable)
      weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,128)), name='w4', trainable=trainable)
      weight['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,256)), name='w5', trainable=trainable)
      weight['w6_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w6', trainable=trainable)
      weight['w7_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w7', trainable=trainable)
      weight['w8_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,512)), name='w8', trainable=trainable)
      weight['w9_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w9', trainable=trainable)
      weight['w10_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w10', trainable=trainable)
      weight['w11_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w11', trainable=trainable)
      weight['w12_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w12', trainable=trainable)
      weight['w13_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w13', trainable=trainable)
      weight['w14_fc'] = tf.Variable(self.denseinitializer(shape=(512, self.num_classes)),name='w14', trainable=trainable)

      weight['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b1', trainable=trainable)
      weight['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b2', trainable=trainable)
      weight['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b3', trainable=trainable)
      weight['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b4', trainable=trainable)
      weight['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b5', trainable=trainable)
      weight['b6_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b6', trainable=trainable)
      weight['b7_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b7', trainable=trainable)
      weight['b8_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b8', trainable=trainable)
      weight['b9_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b9', trainable=trainable)
      weight['b10_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b10', trainable=trainable)
      weight['b11_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b11', trainable=trainable)
      weight['b12_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b12', trainable=trainable)
      weight['b13_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b13', trainable=trainable)
      weight['b14_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b14', trainable=trainable)
      return weight

    if self.version == 19:
      weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(3,3,3,64)), name='w1', trainable=trainable)
      weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w2', trainable=trainable)
      weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,128)), name='w3', trainable=trainable)
      weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,128)), name='w4', trainable=trainable)
      weight['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,256)), name='w5', trainable=trainable)
      weight['w6_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w6', trainable=trainable)
      weight['w7_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w7', trainable=trainable)
      weight['w8_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w8', trainable=trainable)
      weight['w9_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,512)), name='w9', trainable=trainable)
      weight['w10_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w10', trainable=trainable)
      weight['w11_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w11', trainable=trainable)
      weight['w12_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w12', trainable=trainable)
      weight['w13_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w13', trainable=trainable)
      weight['w14_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w14', trainable=trainable)
      weight['w15_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w15', trainable=trainable)
      weight['w16_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w16', trainable=trainable)
      weight['w17_fc'] = tf.Variable(self.denseinitializer(shape=(512, self.num_classes)),name='w17', trainable=trainable)

      weight['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b1', trainable=trainable)
      weight['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b2', trainable=trainable)
      weight['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b3', trainable=trainable)
      weight['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b4', trainable=trainable)
      weight['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b5', trainable=trainable)
      weight['b6_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b6', trainable=trainable)
      weight['b7_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b7', trainable=trainable)
      weight['b8_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b8', trainable=trainable)
      weight['b9_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b9', trainable=trainable)
      weight['b10_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b10', trainable=trainable)
      weight['b11_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b11', trainable=trainable)
      weight['b12_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b12', trainable=trainable)
      weight['b13_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b13', trainable=trainable)
      weight['b14_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b14', trainable=trainable)
      weight['b15_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b15', trainable=trainable)
      weight['b16_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b16', trainable=trainable)
      weight['b17_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b17', trainable=trainable)
      return weight
  def forward_pass(self,weights, inputs):
        def _conv_block(inputs, filt, st=1):
            inputs = tf.nn.conv2d(inputs, filt['w'], [1, st, st, 1], 'SAME') + filt['b']
            inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.nn.relu(inputs)
            return inputs
        if self.version == 11:
           init_st = 2 if self.datasource == 'tiny-imagenet' else 1
           inputs = _conv_block(inputs, {'w': weights['w1_conv'], 'b': weights['b1_conv']}, init_st)
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w2_conv'], 'b': weights['b2_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w3_conv'], 'b': weights['b3_conv']})
           inputs = _conv_block(inputs, {'w': weights['w4_conv'], 'b': weights['b4_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w5_conv'], 'b': weights['b5_conv']})
           inputs = _conv_block(inputs, {'w': weights['w6_conv'], 'b': weights['b6_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w7_conv'], 'b': weights['b7_conv']})
           inputs = _conv_block(inputs, {'w': weights['w8_conv'], 'b': weights['b8_conv']})
           #inputs = tf.nn.avg_pool(inputs, ksize=[1,2,2,1], padding='VALID', strides=1)
           inputs = tf.reduce_mean(inputs, axis=[1,2])     #globalAveragePooling
           inputs = tf.matmul(inputs, weights['w9_fc']) + weights['b9_fc']
           return inputs
        if self.version == 13:
           init_st = 2 if self.datasource == 'tiny-imagenet' else 1
           inputs = _conv_block(inputs, {'w': weights['w1_conv'], 'b': weights['b1_conv']}, init_st)
           inputs = _conv_block(inputs, {'w': weights['w2_conv'], 'b': weights['b2_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w3_conv'], 'b': weights['b3_conv']})
           inputs = _conv_block(inputs, {'w': weights['w4_conv'], 'b': weights['b4_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w5_conv'], 'b': weights['b5_conv']})
           inputs = _conv_block(inputs, {'w': weights['w6_conv'], 'b': weights['b6_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w7_conv'], 'b': weights['b7_conv']})
           inputs = _conv_block(inputs, {'w': weights['w8_conv'], 'b': weights['b8_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w9_conv'], 'b': weights['b9_conv']})
           inputs = _conv_block(inputs, {'w': weights['w10_conv'], 'b': weights['b10_conv']})
           #inputs = tf.nn.avg_pool(inputs, ksize=[1,2,2,1], padding='VALID', strides=1)
           inputs = tf.reduce_mean(inputs, axis=[1,2])     #globalAveragePooling
           inputs = tf.matmul(inputs, weights['w11_fc']) + weights['b11_fc']
           return inputs
        
        
        if self.version == 16:
           init_st = 2 if self.datasource == 'tiny-imagenet' else 1
           inputs = _conv_block(inputs, {'w': weights['w1_conv'], 'b': weights['b1_conv']}, init_st)
           inputs = _conv_block(inputs, {'w': weights['w2_conv'], 'b': weights['b2_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w3_conv'], 'b': weights['b3_conv']})
           inputs = _conv_block(inputs, {'w': weights['w4_conv'], 'b': weights['b4_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w5_conv'], 'b': weights['b5_conv']})
           inputs = _conv_block(inputs, {'w': weights['w6_conv'], 'b': weights['b6_conv']})
           inputs = _conv_block(inputs, {'w': weights['w7_conv'], 'b': weights['b7_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w8_conv'], 'b': weights['b8_conv']})
           inputs = _conv_block(inputs, {'w': weights['w9_conv'], 'b': weights['b9_conv']})
           inputs = _conv_block(inputs, {'w': weights['w10_conv'], 'b': weights['b10_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w11_conv'], 'b': weights['b11_conv']})
           inputs = _conv_block(inputs, {'w': weights['w12_conv'], 'b': weights['b12_conv']})
           inputs = _conv_block(inputs, {'w': weights['w13_conv'], 'b': weights['b13_conv']})
           #inputs = tf.nn.avg_pool(inputs, ksize=[1,2,2,1], padding='VALID', strides=1)
           inputs = tf.reduce_mean(inputs, axis=[1,2])     #globalAveragePooling
           inputs = tf.matmul(inputs, weights['w14_fc']) + weights['b14_fc']
           return inputs
        if self.version == 19:
           init_st = 2 if self.datasource == 'tiny-imagenet' else 1
           inputs = _conv_block(inputs, {'w': weights['w1_conv'], 'b': weights['b1_conv']}, init_st)
           inputs = _conv_block(inputs, {'w': weights['w2_conv'], 'b': weights['b2_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w3_conv'], 'b': weights['b3_conv']})
           inputs = _conv_block(inputs, {'w': weights['w4_conv'], 'b': weights['b4_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w5_conv'], 'b': weights['b5_conv']})
           inputs = _conv_block(inputs, {'w': weights['w6_conv'], 'b': weights['b6_conv']})
           inputs = _conv_block(inputs, {'w': weights['w7_conv'], 'b': weights['b7_conv']})
           inputs = _conv_block(inputs, {'w': weights['w8_conv'], 'b': weights['b8_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w9_conv'], 'b': weights['b9_conv']})
           inputs = _conv_block(inputs, {'w': weights['w10_conv'], 'b': weights['b10_conv']})
           inputs = _conv_block(inputs, {'w': weights['w11_conv'], 'b': weights['b11_conv']})
           inputs = _conv_block(inputs, {'w': weights['w12_conv'], 'b': weights['b12_conv']})
           inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
           inputs = _conv_block(inputs, {'w': weights['w13_conv'], 'b': weights['b13_conv']})
           inputs = _conv_block(inputs, {'w': weights['w14_conv'], 'b': weights['b14_conv']})
           inputs = _conv_block(inputs, {'w': weights['w15_conv'], 'b': weights['b15_conv']})
           inputs = _conv_block(inputs, {'w': weights['w16_conv'], 'b': weights['b16_conv']})
           #inputs = tf.nn.avg_pool(inputs, ksize=[1,2,2,1], padding='VALID', strides=1)
           inputs = tf.reduce_mean(inputs, axis=[1,2])     #globalAveragePooling
           inputs = tf.matmul(inputs, weights['w17_fc']) + weights['b17_fc']
           return inputs
  def make_layers(self, final_w, final_mask):
       init_st = 2 if self.datasource == 'tiny-imagenet' else 1
       if self.version == 11:
           shape = tuple(self.input_dims)
           inp = tf.keras.layers.Input(shape= shape)

           conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'], [1,init_st,init_st,1],'SAME')(inp)
           batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
           Rel1 = tf.keras.layers.ReLU()(batchnormal1)
           pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel1)

           conv2 = ConvTwo(final_w['w2_conv'], final_w['b2_conv'], [1,1,1,1],'SAME')(pool1)
           batchnormal2 = tf.keras.layers.BatchNormalization()(conv2)
           Rel2 = tf.keras.layers.ReLU()(batchnormal2)
           pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel2)

           conv3 = ConvTwo(final_w['w3_conv'], final_w['b3_conv'], [1,1,1,1],'SAME')(pool2)
           batchnormal3 = tf.keras.layers.BatchNormalization()(conv3)
           Rel3 = tf.keras.layers.ReLU()(batchnormal3)

           conv4 = ConvTwo(final_w['w4_conv'], final_w['b4_conv'], [1,1,1,1],'SAME')(Rel3)
           batchnormal4 = tf.keras.layers.BatchNormalization()(conv4)
           Rel4 = tf.keras.layers.ReLU()(batchnormal4)
           pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel4)

           conv5 = ConvTwo(final_w['w5_conv'], final_w['b5_conv'], [1,1,1,1],'SAME')(pool3)
           batchnormal5 = tf.keras.layers.BatchNormalization()(conv5)
           Rel5 = tf.keras.layers.ReLU()(batchnormal5)

           conv6 = ConvTwo(final_w['w6_conv'], final_w['b6_conv'], [1,1,1,1],'SAME')(Rel5)
           batchnormal6 = tf.keras.layers.BatchNormalization()(conv6)
           Rel6 = tf.keras.layers.ReLU()(batchnormal6)
           pool4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel6)


           conv7 = ConvTwo(final_w['w7_conv'], final_w['b7_conv'], [1,1,1,1],'SAME')(pool4)
           batchnormal7 = tf.keras.layers.BatchNormalization()(conv7)
           Rel7 = tf.keras.layers.ReLU()(batchnormal7)

           conv8 = ConvTwo(final_w['w8_conv'], final_w['b8_conv'], [1,1,1,1],'SAME')(Rel7)
           batchnormal8 = tf.keras.layers.BatchNormalization()(conv8)
           Rel8 = tf.keras.layers.ReLU()(batchnormal8)

           #flatten = tf.keras.layers.Flatten()(Rel8)
           flatten = tf.keras.layers.GlobalAvgPool2D()(Rel8)
           fc1 = Dense(final_w['w9_fc'], final_w['b9_fc'], tf.keras.activations.linear)(flatten)
            
           model = tf.keras.models.Model(inp, fc1)
           return model
       if self.version == 13:
           shape = tuple(self.input_dims)
           inp = tf.keras.layers.Input(shape= shape)

           conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'], [1,init_st,init_st,1],'SAME')(inp)
           batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
           Rel1 = tf.keras.layers.ReLU()(batchnormal1)
           

           conv2 = ConvTwo(final_w['w2_conv'], final_w['b2_conv'], [1,1,1,1],'SAME')(Rel1)
           batchnormal2 = tf.keras.layers.BatchNormalization()(conv2)
           Rel2 = tf.keras.layers.ReLU()(batchnormal2)
           pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel2)

           conv3 = ConvTwo(final_w['w3_conv'], final_w['b3_conv'], [1,1,1,1],'SAME')(pool1)
           batchnormal3 = tf.keras.layers.BatchNormalization()(conv3)
           Rel3 = tf.keras.layers.ReLU()(batchnormal3)

           conv4 = ConvTwo(final_w['w4_conv'], final_w['b4_conv'], [1,1,1,1],'SAME')(Rel3)
           batchnormal4 = tf.keras.layers.BatchNormalization()(conv4)
           Rel4 = tf.keras.layers.ReLU()(batchnormal4)
           pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel4)

           conv5 = ConvTwo(final_w['w5_conv'], final_w['b5_conv'], [1,1,1,1],'SAME')(pool2)
           batchnormal5 = tf.keras.layers.BatchNormalization()(conv5)
           Rel5 = tf.keras.layers.ReLU()(batchnormal5)

           conv6 = ConvTwo(final_w['w6_conv'], final_w['b6_conv'], [1,1,1,1],'SAME')(Rel5)
           batchnormal6 = tf.keras.layers.BatchNormalization()(conv6)
           Rel6 = tf.keras.layers.ReLU()(batchnormal6)
           pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel6)


           conv7 = ConvTwo(final_w['w7_conv'], final_w['b7_conv'], [1,1,1,1],'SAME')(pool3)
           batchnormal7 = tf.keras.layers.BatchNormalization()(conv7)
           Rel7 = tf.keras.layers.ReLU()(batchnormal7)

           conv8 = ConvTwo(final_w['w8_conv'], final_w['b8_conv'], [1,1,1,1],'SAME')(Rel7)
           batchnormal8 = tf.keras.layers.BatchNormalization()(conv8)
           Rel8 = tf.keras.layers.ReLU()(batchnormal8)
           pool4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel8)

           conv9 = ConvTwo(final_w['w9_conv'], final_w['b9_conv'], [1,1,1,1],'SAME')(pool4)
           batchnormal9 = tf.keras.layers.BatchNormalization()(conv9)
           Rel9 = tf.keras.layers.ReLU()(batchnormal9)

           conv10 = ConvTwo(final_w['w10_conv'], final_w['b10_conv'], [1,1,1,1],'SAME')(Rel9)
           batchnormal10 = tf.keras.layers.BatchNormalization()(conv10)
           Rel10 = tf.keras.layers.ReLU()(batchnormal10)

           #flatten = tf.keras.layers.Flatten()(Rel8)
           flatten = tf.keras.layers.GlobalAvgPool2D()(Rel8)
           fc1 = Dense(final_w['w11_fc'], final_w['b11_fc'], tf.keras.activations.linear)(flatten)
           
           model = tf.keras.models.Model(inp, fc1)
           return model
       if self.version == 16:
           shape = tuple(self.input_dims)
           inp = tf.keras.layers.Input(shape= shape)

           conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'],final_mask['w1_conv'], final_mask['b1_conv'], [1,init_st,init_st,1],'SAME', 'w1_conv', 'b1_conv', 'w1_mask', 'b1_mask')(inp)
           batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
           Rel1 = tf.keras.layers.ReLU()(batchnormal1)
           

           conv2 = ConvTwo(final_w['w2_conv'], final_w['b2_conv'], final_mask['w2_conv'], final_mask['b2_conv'],[1,1,1,1],'SAME', 'w2_conv', 'b2_conv', 'w2_mask', 'b2_mask')(Rel1)
           batchnormal2 = tf.keras.layers.BatchNormalization()(conv2)
           Rel2 = tf.keras.layers.ReLU()(batchnormal2)
           pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel2)

           conv3 = ConvTwo(final_w['w3_conv'], final_w['b3_conv'], final_mask['w3_conv'], final_mask['b3_conv'],[1,1,1,1],'SAME', 'w3_conv', 'b3_conv', 'w3_mask', 'b3_mask')(pool1)
           batchnormal3 = tf.keras.layers.BatchNormalization()(conv3)
           Rel3 = tf.keras.layers.ReLU()(batchnormal3)

           conv4 = ConvTwo(final_w['w4_conv'], final_w['b4_conv'], final_mask['w4_conv'], final_mask['b4_conv'],[1,1,1,1],'SAME', 'w4_conv', 'b4_conv','w4_mask', 'b4_mask')(Rel3)
           batchnormal4 = tf.keras.layers.BatchNormalization()(conv4)
           Rel4 = tf.keras.layers.ReLU()(batchnormal4)
           pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel4)

           conv5 = ConvTwo(final_w['w5_conv'], final_w['b5_conv'], final_mask['w5_conv'], final_mask['b5_conv'], [1,1,1,1],'SAME','w5_conv', 'b5_conv', 'w5_mask', 'b5_mask')(pool2)
           batchnormal5 = tf.keras.layers.BatchNormalization()(conv5)
           Rel5 = tf.keras.layers.ReLU()(batchnormal5)

           conv6 = ConvTwo(final_w['w6_conv'], final_w['b6_conv'],final_mask['w6_conv'], final_mask['b6_conv'], [1,1,1,1],'SAME', 'w6_conv', 'b6_conv', 'w6_mask', 'b6_mask')(Rel5)
           batchnormal6 = tf.keras.layers.BatchNormalization()(conv6)
           Rel6 = tf.keras.layers.ReLU()(batchnormal6)

           conv7 = ConvTwo(final_w['w7_conv'], final_w['b7_conv'], final_mask['w7_conv'], final_mask['b7_conv'],[1,1,1,1],'SAME', 'w7_conv', 'b7_conv', 'w7_mask', 'b7_mask')(Rel6)
           batchnormal7 = tf.keras.layers.BatchNormalization()(conv7)
           Rel7 = tf.keras.layers.ReLU()(batchnormal7)
           pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel7)


           conv8 = ConvTwo(final_w['w8_conv'], final_w['b8_conv'], final_mask['w8_conv'], final_mask['b8_conv'],[1,1,1,1],'SAME', 'w8_conv', 'b8_conv', 'w8_mask', 'b8_mask')(pool3)
           batchnormal8 = tf.keras.layers.BatchNormalization()(conv8)
           Rel8 = tf.keras.layers.ReLU()(batchnormal8)

           conv9 = ConvTwo(final_w['w9_conv'], final_w['b9_conv'],final_mask['w9_conv'], final_mask['b9_conv'], [1,1,1,1],'SAME', 'w9_conv', 'b9_conv', 'w9_mask', 'b9_mask')(Rel8)
           batchnormal9 = tf.keras.layers.BatchNormalization()(conv9)
           Rel9 = tf.keras.layers.ReLU()(batchnormal9)

           conv10 = ConvTwo(final_w['w10_conv'], final_w['b10_conv'],final_mask['w10_conv'], final_mask['b10_conv'], [1,1,1,1],'SAME', 'w10_conv', 'b10_conv', 'w10_mask', 'b10_mask')(Rel9)
           batchnormal10 = tf.keras.layers.BatchNormalization()(conv10)
           Rel10 = tf.keras.layers.ReLU()(batchnormal10)
           pool4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel10)
           
           conv11 = ConvTwo(final_w['w11_conv'], final_w['b11_conv'],final_mask['w11_conv'], final_mask['b11_conv'], [1,1,1,1],'SAME', 'w11_conv', 'b11_conv', 'w11_mask', 'b11_mask')(pool4)
           batchnormal11 = tf.keras.layers.BatchNormalization()(conv11)
           Rel11 = tf.keras.layers.ReLU()(batchnormal11)


           conv12 = ConvTwo(final_w['w12_conv'], final_w['b12_conv'],final_mask['w12_conv'], final_mask['b12_conv'], [1,1,1,1],'SAME', 'w12_conv', 'b12_conv', 'w12_mask', 'b12_mask')(Rel11)
           batchnormal12 = tf.keras.layers.BatchNormalization()(conv12)
           Rel12 = tf.keras.layers.ReLU()(batchnormal12)

           conv13 = ConvTwo(final_w['w13_conv'], final_w['b13_conv'],final_mask['w13_conv'], final_mask['b13_conv'], [1,1,1,1],'SAME', 'w13_conv', 'b13_conv', 'w13_mask', 'b13_mask')(Rel12)
           batchnormal13 = tf.keras.layers.BatchNormalization()(conv13)
           Rel13 = tf.keras.layers.ReLU()(batchnormal13)
           #flatten = tf.keras.layers.Flatten()(Rel8)
           flatten = tf.keras.layers.GlobalAvgPool2D()(Rel13)
           fc1 = Dense(final_w['w14_fc'], final_w['b14_fc'],final_mask['w14_fc'], final_mask['b14_fc'], tf.keras.activations.linear,'w14_fc', 'b14_fc', 'w14_mask', 'b14_mask')(flatten)
           
           model = tf.keras.models.Model(inp, fc1)
           return model
       if self.version == 19:
           shape = tuple(self.input_dims)
           inp = tf.keras.layers.Input(shape= shape)

           conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'],final_mask['w1_conv'], final_mask['b1_conv'], [1,init_st,init_st,1],'SAME','w1_conv', 'b1_conv', 'w1_mask', 'b1_mask')(inp)
           batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
           Rel1 = tf.keras.layers.ReLU()(batchnormal1)
           

           conv2 = ConvTwo(final_w['w2_conv'], final_w['b2_conv'],final_mask['w2_conv'], final_mask['b2_conv'], [1,1,1,1],'SAME', 'w2_conv', 'b2_conv', 'w2_mask', 'b2_mask')(Rel1)
           batchnormal2 = tf.keras.layers.BatchNormalization()(conv2)
           Rel2 = tf.keras.layers.ReLU()(batchnormal2)
           pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel2)

           conv3 = ConvTwo(final_w['w3_conv'], final_w['b3_conv'],final_mask['w3_conv'], final_mask['b3_conv'], [1,1,1,1],'SAME', 'w3_conv', 'b3_conv', 'w3_mask', 'b3_mask')(pool1)
           batchnormal3 = tf.keras.layers.BatchNormalization()(conv3)
           Rel3 = tf.keras.layers.ReLU()(batchnormal3)

           conv4 = ConvTwo(final_w['w4_conv'], final_w['b4_conv'], final_mask['w4_conv'], final_mask['b4_conv'],[1,1,1,1],'SAME', 'w4_conv', 'b4_conv', 'w4_mask', 'b4_mask')(Rel3)
           batchnormal4 = tf.keras.layers.BatchNormalization()(conv4)
           Rel4 = tf.keras.layers.ReLU()(batchnormal4)
           pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel4)

           conv5 = ConvTwo(final_w['w5_conv'], final_w['b5_conv'], final_mask['w5_conv'], final_mask['b5_conv'],[1,1,1,1],'SAME', 'w5_conv', 'b5_conv', 'w5_mask', 'b5_mask')(pool2)
           batchnormal5 = tf.keras.layers.BatchNormalization()(conv5)
           Rel5 = tf.keras.layers.ReLU()(batchnormal5)

           conv6 = ConvTwo(final_w['w6_conv'], final_w['b6_conv'], final_mask['w6_conv'], final_mask['b6_conv'],[1,1,1,1],'SAME',  'w6_conv', 'b6_conv', 'w6_mask', 'b6_mask')(Rel5)
           batchnormal6 = tf.keras.layers.BatchNormalization()(conv6)
           Rel6 = tf.keras.layers.ReLU()(batchnormal6)

           conv7 = ConvTwo(final_w['w7_conv'], final_w['b7_conv'],final_mask['w7_conv'], final_mask['b7_conv'], [1,1,1,1],'SAME', 'w7_conv', 'b7_conv', 'w7_mask', 'b7_mask')(Rel6)
           batchnormal7 = tf.keras.layers.BatchNormalization()(conv7)
           Rel7 = tf.keras.layers.ReLU()(batchnormal7)
           
           conv8 = ConvTwo(final_w['w8_conv'], final_w['b8_conv'],final_mask['w8_conv'], final_mask['b8_conv'], [1,1,1,1],'SAME', 'w8_conv', 'b8_conv', 'w8_mask', 'b8_mask')(Rel7)
           batchnormal8 = tf.keras.layers.BatchNormalization()(conv8)
           Rel8 = tf.keras.layers.ReLU()(batchnormal8)
           pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel8)

           conv9 = ConvTwo(final_w['w9_conv'], final_w['b9_conv'],final_mask['w9_conv'], final_mask['b9_conv'], [1,1,1,1],'SAME', 'w9_conv', 'b9_conv', 'w9_mask', 'b9_mask')(pool3)
           batchnormal9 = tf.keras.layers.BatchNormalization()(conv9)
           Rel9 = tf.keras.layers.ReLU()(batchnormal9)

           conv10 = ConvTwo(final_w['w10_conv'], final_w['b10_conv'],final_mask['w10_conv'], final_mask['b10_conv'], [1,1,1,1],'SAME', 'w10_conv', 'b10_conv', 'w10_mask', 'b10_mask')(Rel9)
           batchnormal10 = tf.keras.layers.BatchNormalization()(conv10)
           Rel10 = tf.keras.layers.ReLU()(batchnormal10)
           
           
           conv11 = ConvTwo(final_w['w11_conv'], final_w['b11_conv'],final_mask['w11_conv'], final_mask['b11_conv'], [1,1,1,1],'SAME', 'w11_conv', 'b11_conv', 'w11_mask', 'b11_mask')(Rel10)
           batchnormal11 = tf.keras.layers.BatchNormalization()(conv11)
           Rel11 = tf.keras.layers.ReLU()(batchnormal11)


           conv12 = ConvTwo(final_w['w12_conv'], final_w['b12_conv'],final_mask['w12_conv'], final_mask['b12_conv'], [1,1,1,1],'SAME', 'w12_conv', 'b12_conv', 'w12_mask', 'b12_mask')(Rel11)
           batchnormal12 = tf.keras.layers.BatchNormalization()(conv12)
           Rel12 = tf.keras.layers.ReLU()(batchnormal12)
           pool4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')(Rel12)

           conv13 = ConvTwo(final_w['w13_conv'], final_w['b13_conv'],final_mask['w13_conv'], final_mask['b13_conv'], [1,1,1,1],'SAME', 'w13_conv', 'b13_conv', 'w13_mask', 'b13_mask')(pool4)
           batchnormal13 = tf.keras.layers.BatchNormalization()(conv13)
           Rel13 = tf.keras.layers.ReLU()(batchnormal13)

           conv14 = ConvTwo(final_w['w14_conv'], final_w['b14_conv'],final_mask['w14_conv'], final_mask['b14_conv'], [1,1,1,1],'SAME', 'w14_conv', 'b14_conv', 'w14_mask', 'b14_mask')(Rel13)
           batchnormal14 = tf.keras.layers.BatchNormalization()(conv14)
           Rel14 = tf.keras.layers.ReLU()(batchnormal14)

           conv15 = ConvTwo(final_w['w15_conv'], final_w['b15_conv'], final_mask['w15_conv'], final_mask['b15_conv'],[1,1,1,1],'SAME', 'w15_conv', 'b15_conv', 'w15_mask', 'b15_mask')(Rel14)
           batchnormal15 = tf.keras.layers.BatchNormalization()(conv15)
           Rel15 = tf.keras.layers.ReLU()(batchnormal15)

           conv16 = ConvTwo(final_w['w16_conv'], final_w['b16_conv'], final_mask['w16_conv'], final_mask['b16_conv'],[1,1,1,1],'SAME', 'w16_conv', 'b16_conv', 'w16_mask', 'b16_mask')(Rel15)
           batchnormal16 = tf.keras.layers.BatchNormalization()(conv16)
           Rel16 = tf.keras.layers.ReLU()(batchnormal16)
           #flatten = tf.keras.layers.Flatten()(Rel8)
           flatten = tf.keras.layers.GlobalAvgPool2D()(Rel16)
           fc1 = Dense(final_w['w17_fc'], final_w['b17_fc'],final_mask['w17_fc'], final_mask['b17_fc'], tf.keras.activations.linear)(flatten)
           
           model = tf.keras.models.Model(inp, fc1)
           return model

class ResNet18(object):
  def __init__(self, datasource, num_classes, *initializer):
    self.num_classes = num_classes
    self.datasource = datasource
    self.name = 'resnet-18'
    self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3]
    self.convinitializer = initializer[0]
    self.denseinitializer = initializer[1]
  
  def construct_weights(self, trainable=True):
      weight = {}
    
      weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(3,3,3,64)), name='w1', trainable=trainable)
      #residual block1
      weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w2', trainable=trainable)
      weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w3', trainable=trainable)
      #residual block2
      weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w4', trainable=trainable)
      weight['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w5', trainable=trainable)
      #residual block3
      weight['w6_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,128)), name='w6', trainable=trainable)
      weight['w7_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,128)), name='w7', trainable=trainable)
      weight['res3'] = tf.Variable(self.convinitializer(shape=(1,1,64,128)), name='res3', trainable=trainable)
      #residual block4
      weight['w8_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,128)), name='w8', trainable=trainable)
      weight['w9_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,128)), name='w9', trainable=trainable)
      #residual block5
      weight['w10_conv'] = tf.Variable(self.convinitializer(shape=(3,3,128,256)), name='w10', trainable=trainable)
      weight['w11_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w11', trainable=trainable)
      weight['res5'] = tf.Variable(self.convinitializer(shape=(1,1,128,256)), name='res5', trainable=trainable)
      #residual block6
      weight['w12_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w12', trainable=trainable)
      weight['w13_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,256)), name='w13', trainable=trainable)
      #residual block7
      weight['w14_conv'] = tf.Variable(self.convinitializer(shape=(3,3,256,512)), name='w14', trainable=trainable)
      weight['w15_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w15', trainable=trainable)
      weight['res7'] = tf.Variable(self.convinitializer(shape=(1,1,256,512)), name='res7', trainable=trainable)
      #residual block8
      weight['w16_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w16', trainable=trainable)
      weight['w17_conv'] = tf.Variable(self.convinitializer(shape=(3,3,512,512)), name='w17', trainable=trainable)

      weight['w18_fc'] = tf.Variable(self.denseinitializer(shape=(512, self.num_classes)),name='w18', trainable=trainable)

      weight['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b1', trainable=trainable)
      #residual block1
      weight['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b2', trainable=trainable)
      weight['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b3', trainable=trainable)
      #residual block2
      weight['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b4', trainable=trainable)
      weight['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b5', trainable=trainable)
      #residual block3
      weight['b6_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b6', trainable=trainable)
      weight['b7_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b7', trainable=trainable)
      #residual block4
      weight['b8_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b8', trainable=trainable)
      weight['b9_conv'] = tf.Variable(tf.zeros_initializer()(shape=(128,)), name='b9', trainable=trainable)
      #residual block5
      weight['b10_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b10', trainable=trainable)
      weight['b11_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b11', trainable=trainable)
      #residual block6
      weight['b12_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b12', trainable=trainable)
      weight['b13_conv'] = tf.Variable(tf.zeros_initializer()(shape=(256,)), name='b13', trainable=trainable)
      #residual block7
      weight['b14_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b14', trainable=trainable)
      weight['b15_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b15', trainable=trainable)
      #residual block8
      weight['b16_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b16', trainable=trainable)
      weight['b17_conv'] = tf.Variable(tf.zeros_initializer()(shape=(512,)), name='b17', trainable=trainable)

      weight['b18_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b18', trainable=trainable)
      
      return weight

  def forward_pass(self,weights, inputs):
       #residual block1
       def _residual_block(inputs, dic1, dic2, st, dic3={}):
            shortcut = inputs
            inputs = tf.nn.conv2d(inputs, dic1['w'], [1, st, st, 1], 'SAME') + dic1['b']
            inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, dic2['w'], [1, 1, 1, 1], 'SAME') + dic2['b']
            x = tf.keras.layers.BatchNormalization()(inputs)
            if st != 1:
              shortcut = tf.nn.conv2d(shortcut, dic3['w'], [1,st,st,1], 'SAME')
              shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            x = tf.keras.layers.Add()([x,shortcut])
            x = tf.nn.relu(x)

            return x

       inputs = tf.nn.conv2d(inputs, weights['w1_conv'], [1, 1, 1, 1], 'SAME') + weights['b1_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)

       inputs =_residual_block(inputs, {'w': weights['w2_conv'], 'b': weights['b2_conv']},  {'w': weights['w3_conv'], 'b': weights['b3_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w4_conv'], 'b': weights['b4_conv']}, {'w': weights['w5_conv'], 'b': weights['b5_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w6_conv'], 'b': weights['b6_conv']},  {'w': weights['w7_conv'], 'b': weights['b7_conv']},2, {'w':weights['res3']})
       inputs =_residual_block(inputs, {'w': weights['w8_conv'], 'b': weights['b8_conv']}, {'w': weights['w9_conv'], 'b': weights['b9_conv']}, 1)
       inputs =_residual_block(inputs,  {'w': weights['w10_conv'], 'b': weights['b10_conv']}, {'w': weights['w11_conv'], 'b': weights['b11_conv']},2, {'w':weights['res5']})
       inputs =_residual_block(inputs, {'w': weights['w12_conv'], 'b': weights['b12_conv']},  {'w': weights['w13_conv'], 'b': weights['b13_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w14_conv'], 'b': weights['b14_conv']},  {'w': weights['w15_conv'], 'b': weights['b15_conv']},2, {'w':weights['res7']})
       inputs =_residual_block(inputs, {'w': weights['w16_conv'], 'b': weights['b16_conv']},  {'w': weights['w17_conv'], 'b': weights['b17_conv']},1)
     
       inputs = tf.reduce_mean(inputs, axis=[1,2])     #globalAveragePooling
       inputs = tf.matmul(inputs, weights['w18_fc']) + weights['b18_fc']
       return inputs

  def make_layers(self, final_w, final_mask):
       shape = tuple(self.input_dims)
       inp = tf.keras.layers.Input(shape= shape)

       conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'],final_mask['w1_conv'], final_mask['b1_conv'], [1,1,1,1],'SAME', 'w1_conv', 'b1_conv', 'w1_mask', 'b1_mask')(inp)
       batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
       Rel1 = tf.keras.layers.ReLU()(batchnormal1)

       Res1 = Residual_Layer(final_w['w2_conv'], final_w['b2_conv'],final_mask['w2_conv'], final_mask['b2_conv'], [1,1,1,1],'SAME', 'w2_conv', 'b2_conv', 'w2_mask', 'b2_mask',
                             final_w['w3_conv'], final_w['b3_conv'],final_mask['w3_conv'], final_mask['b3_conv'], 'w3_conv', 'b3_conv', 'w3_mask', 'b3_mask')(Rel1)

       Res2 = Residual_Layer(final_w['w4_conv'], final_w['b4_conv'],final_mask['w4_conv'], final_mask['b4_conv'], [1,1,1,1],'SAME', 'w4_conv', 'b4_conv', 'w4_mask', 'b4_mask',
                             final_w['w5_conv'], final_w['b5_conv'],final_mask['w5_conv'], final_mask['b5_conv'], 'w5_conv', 'b5_conv', 'w5_mask', 'b5_mask')(Res1)
      
       Res3 = Residual_Layer(final_w['w6_conv'], final_w['b6_conv'],final_mask['w6_conv'], final_mask['b6_conv'], [1,2,2,1],'SAME', 'w6_conv', 'b6_conv', 'w6_mask', 'b6_mask',
                             final_w['w7_conv'], final_w['b7_conv'],final_mask['w7_conv'], final_mask['b7_conv'], 'w7_conv', 'b7_conv', 'w7_mask', 'b7_mask',
                             final_w['res3'], final_mask['res3'], 'res3', 'res3_mask')(Res2)

       Res4 = Residual_Layer(final_w['w8_conv'], final_w['b8_conv'],final_mask['w8_conv'], final_mask['b8_conv'], [1,1,1,1],'SAME', 'w8_conv', 'b8_conv', 'w8_mask', 'b8_mask',
                             final_w['w9_conv'], final_w['b9_conv'],final_mask['w9_conv'], final_mask['b9_conv'], 'w9_conv', 'b9_conv', 'w9_mask', 'b9_mask')(Res3)

       Res5 = Residual_Layer(final_w['w10_conv'], final_w['b10_conv'],final_mask['w10_conv'], final_mask['b10_conv'], [1,2,2,1],'SAME', 'w10_conv', 'b10_conv', 'w10_mask', 'b10_mask',
                             final_w['w11_conv'], final_w['b11_conv'],final_mask['w11_conv'], final_mask['b11_conv'], 'w11_conv', 'b11_conv', 'w11_mask', 'b11_mask',
                             final_w['res5'], final_mask['res5'], 'res5', 'res5_mask')(Res4)
       
       Res6 = Residual_Layer(final_w['w12_conv'], final_w['b12_conv'],final_mask['w12_conv'], final_mask['b12_conv'], [1,1,1,1],'SAME', 'w12_conv', 'b12_conv', 'w12_mask', 'b12_mask',
                             final_w['w13_conv'], final_w['b13_conv'],final_mask['w13_conv'], final_mask['b13_conv'], 'w13_conv', 'b13_conv', 'w13_mask', 'b13_mask')(Res5)
        
       Res7 = Residual_Layer(final_w['w14_conv'], final_w['b14_conv'],final_mask['w14_conv'], final_mask['b14_conv'], [1,2,2,1],'SAME', 'w14_conv', 'b14_conv', 'w14_mask', 'b14_mask',
                             final_w['w15_conv'], final_w['b15_conv'],final_mask['w15_conv'], final_mask['b15_conv'], 'w15_conv', 'b15_conv', 'w15_mask', 'b15_mask',
                             final_w['res7'], final_mask['res7'], 'res5', 'res7_mask')(Res6)

       Res8 = Residual_Layer(final_w['w16_conv'], final_w['b16_conv'],final_mask['w16_conv'], final_mask['b16_conv'], [1,1,1,1],'SAME', 'w16_conv', 'b16_conv', 'w16_mask', 'b16_mask',
                             final_w['w17_conv'], final_w['b17_conv'],final_mask['w17_conv'], final_mask['b17_conv'], 'w17_conv', 'b17_conv', 'w17_mask', 'b17_mask')(Res7)

       flatten = tf.keras.layers.GlobalAvgPool2D()(Res8)
       fc1 = Dense(final_w['w18_fc'], final_w['b18_fc'],final_mask['w18_fc'], final_mask['b18_fc'], tf.keras.activations.linear, 'w18_fc', 'b18_fc', 'w18_mask', 'b18_mask')(flatten)
           
       model = tf.keras.models.Model(inp, fc1)
       return model

class ResNet18_v2(object):
  def __init__(self, datasource, num_classes, *initializer):
    self.num_classes = num_classes
    self.datasource = datasource
    self.name = 'resnet-18-v2'
    self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3]
    self.convinitializer = initializer[0]
    self.denseinitializer = initializer[1]
  
  def construct_weights(self, trainable=True):
      weight = {}
    
      weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(3,3,3,16)), name='w1', trainable=trainable)
      #residual block1
      weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w2', trainable=trainable)
      weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w3', trainable=trainable)
      #residual block2
      weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w4', trainable=trainable)
      weight['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w5', trainable=trainable)
      #residual block3
      weight['w6_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w6', trainable=trainable)
      weight['w7_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w7', trainable=trainable)
     
      #residual block4
      weight['w8_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,32)), name='w8', trainable=trainable)
      weight['w9_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w9', trainable=trainable)
      weight['res4'] = tf.Variable(self.convinitializer(shape=(1,1,16,32)), name='res5', trainable=False)

      #residual block5
      weight['w10_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w10', trainable=trainable)
      weight['w11_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w11', trainable=trainable)
      weight['res5'] = tf.Variable(self.convinitializer(shape=(1,1,32,32)), name='res5', trainable=False)

      #residual block6
      weight['w12_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w12', trainable=trainable)
      weight['w13_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w13', trainable=trainable)
      weight['res6'] = tf.Variable(self.convinitializer(shape=(1,1,32,32)), name='res6', trainable=False)

      #residual block7
      weight['w14_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,64)), name='w14', trainable=trainable)
      weight['w15_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w15', trainable=trainable)
      weight['res7'] = tf.Variable(self.convinitializer(shape=(1,1,32,64)), name='res7', trainable=False)

      #residual block8
      weight['w16_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w16', trainable=trainable)
      weight['w17_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w17', trainable=trainable)
      weight['res8'] = tf.Variable(self.convinitializer(shape=(1,1,64,64)), name='res8', trainable=False)

      #residual block9
      weight['w18_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w18', trainable=trainable)
      weight['w19_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w19', trainable=trainable)
      weight['res9'] = tf.Variable(self.convinitializer(shape=(1,1,64,64)), name='res9', trainable=False)

      weight['w20_fc'] = tf.Variable(self.denseinitializer(shape=(64, self.num_classes)),name='w20', trainable=trainable)

      weight['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b1', trainable=trainable)
      #residual block1
      weight['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b2', trainable=trainable)
      weight['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b3', trainable=trainable)
      #residual block2
      weight['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b4', trainable=trainable)
      weight['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b5', trainable=trainable)
      #residual block3
      weight['b6_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b6', trainable=trainable)
      weight['b7_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b7', trainable=trainable)
      #residual block4
      weight['b8_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b8', trainable=trainable)
      weight['b9_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b9', trainable=trainable)
      #residual block5
      weight['b10_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b10', trainable=trainable)
      weight['b11_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b11', trainable=trainable)
      #residual block6
      weight['b12_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b12', trainable=trainable)
      weight['b13_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b13', trainable=trainable)
      #residual block7
      weight['b14_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b14', trainable=trainable)
      weight['b15_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b15', trainable=trainable)
      #residual block8
      weight['b16_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b16', trainable=trainable)
      weight['b17_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b17', trainable=trainable)

      #residual block9
      weight['b18_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b18', trainable=trainable)
      weight['b19_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b19', trainable=trainable)
      
      weight['b20_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b20', trainable=trainable)
      
      return weight

  def forward_pass(self,weights, inputs):
       #residual block1
       def _residual_block(inputs, dic1, dic2, st, dic3={}):
            shortcut = inputs
            inputs = tf.nn.conv2d(inputs, dic1['w'], [1, st, st, 1], 'SAME') + dic1['b']
            inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, dic2['w'], [1, 1, 1, 1], 'SAME') + dic2['b']
            x = tf.keras.layers.BatchNormalization()(inputs)
            if st != 1:
              shortcut = tf.nn.conv2d(shortcut, dic3['w'], [1,st,st,1], 'SAME')
              shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            x = tf.keras.layers.Add()([x,shortcut])
            x = tf.nn.relu(x)

            return x

       inputs = tf.nn.conv2d(inputs, weights['w1_conv'], [1, 1, 1, 1], 'SAME') + weights['b1_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)

       inputs =_residual_block(inputs, {'w': weights['w2_conv'], 'b': weights['b2_conv']},  {'w': weights['w3_conv'], 'b': weights['b3_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w4_conv'], 'b': weights['b4_conv']}, {'w': weights['w5_conv'], 'b': weights['b5_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w6_conv'], 'b': weights['b6_conv']},  {'w': weights['w7_conv'], 'b': weights['b7_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w8_conv'], 'b': weights['b8_conv']}, {'w': weights['w9_conv'], 'b': weights['b9_conv']}, 2, {'w':weights['res4']})
       inputs =_residual_block(inputs,  {'w': weights['w10_conv'], 'b': weights['b10_conv']}, {'w': weights['w11_conv'], 'b': weights['b11_conv']},2, {'w':weights['res5']})
       inputs =_residual_block(inputs, {'w': weights['w12_conv'], 'b': weights['b12_conv']},  {'w': weights['w13_conv'], 'b': weights['b13_conv']},2, {'w':weights['res6']})
       inputs =_residual_block(inputs, {'w': weights['w14_conv'], 'b': weights['b14_conv']},  {'w': weights['w15_conv'], 'b': weights['b15_conv']},2, {'w':weights['res7']})
       inputs =_residual_block(inputs, {'w': weights['w16_conv'], 'b': weights['b16_conv']},  {'w': weights['w17_conv'], 'b': weights['b17_conv']},2, {'w':weights['res8']})
       inputs =_residual_block(inputs, {'w': weights['w18_conv'], 'b': weights['b18_conv']},  {'w': weights['w19_conv'], 'b': weights['b19_conv']},2, {'w':weights['res9']})

       inputs = tf.reduce_mean(inputs, axis=[1,2])     #globalAveragePooling
       inputs = tf.matmul(inputs, weights['w20_fc']) + weights['b20_fc']
       return inputs

  def make_layers(self, final_w, final_mask):
       shape = tuple(self.input_dims)
       inp = tf.keras.layers.Input(shape= shape)

       conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'],final_mask['w1_conv'], final_mask['b1_conv'], [1,1,1,1],'SAME', 'w1_conv', 'b1_conv', 'w1_mask', 'b1_mask')(inp)
       batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
       Rel1 = tf.keras.layers.ReLU()(batchnormal1)

       Res1 = Residual_Layer(final_w['w2_conv'], final_w['b2_conv'],final_mask['w2_conv'], final_mask['b2_conv'], [1,1,1,1],'SAME', 'w2_conv', 'b2_conv', 'w2_mask', 'b2_mask',
                             final_w['w3_conv'], final_w['b3_conv'],final_mask['w3_conv'], final_mask['b3_conv'], 'w3_conv', 'b3_conv', 'w3_mask', 'b3_mask')(Rel1)

       Res2 = Residual_Layer(final_w['w4_conv'], final_w['b4_conv'],final_mask['w4_conv'], final_mask['b4_conv'], [1,1,1,1],'SAME', 'w4_conv', 'b4_conv', 'w4_mask', 'b4_mask',
                             final_w['w5_conv'], final_w['b5_conv'],final_mask['w5_conv'], final_mask['b5_conv'], 'w5_conv', 'b5_conv', 'w5_mask', 'b5_mask')(Res1)
      
       Res3 = Residual_Layer(final_w['w6_conv'], final_w['b6_conv'],final_mask['w6_conv'], final_mask['b6_conv'], [1,1,1,1],'SAME', 'w6_conv', 'b6_conv', 'w6_mask', 'b6_mask',
                             final_w['w7_conv'], final_w['b7_conv'],final_mask['w7_conv'], final_mask['b7_conv'], 'w7_conv', 'b7_conv', 'w7_mask', 'b7_mask')(Res2)

       Res4 = Residual_Layer(final_w['w8_conv'], final_w['b8_conv'],final_mask['w8_conv'], final_mask['b8_conv'], [1,1,1,1],'SAME', 'w8_conv', 'b8_conv', 'w8_mask', 'b8_mask',
                             final_w['w9_conv'], final_w['b9_conv'],final_mask['w9_conv'], final_mask['b9_conv'], 'w9_conv', 'b9_conv', 'w9_mask', 'b9_mask',
                             final_w['res4'], final_mask['res4'], 'res4', 'res4_mask')(Res3)

       Res5 = Residual_Layer(final_w['w10_conv'], final_w['b10_conv'],final_mask['w10_conv'], final_mask['b10_conv'], [1,2,2,1],'SAME', 'w10_conv', 'b10_conv', 'w10_mask', 'b10_mask',
                             final_w['w11_conv'], final_w['b11_conv'],final_mask['w11_conv'], final_mask['b11_conv'], 'w11_conv', 'b11_conv', 'w11_mask', 'b11_mask',
                             final_w['res5'], final_mask['res5'], 'res5', 'res5_mask')(Res4)
       
       Res6 = Residual_Layer(final_w['w12_conv'], final_w['b12_conv'],final_mask['w12_conv'], final_mask['b12_conv'], [1,2,2,1],'SAME', 'w12_conv', 'b12_conv', 'w12_mask', 'b12_mask',
                             final_w['w13_conv'], final_w['b13_conv'],final_mask['w13_conv'], final_mask['b13_conv'], 'w13_conv', 'b13_conv', 'w13_mask', 'b13_mask',
                             final_w['res6'], final_mask['res6'], 'res6', 'res6_mask')(Res5)
        
       Res7 = Residual_Layer(final_w['w14_conv'], final_w['b14_conv'],final_mask['w14_conv'], final_mask['b14_conv'], [1,2,2,1],'SAME', 'w14_conv', 'b14_conv', 'w14_mask', 'b14_mask',
                             final_w['w15_conv'], final_w['b15_conv'],final_mask['w15_conv'], final_mask['b15_conv'], 'w15_conv', 'b15_conv', 'w15_mask', 'b15_mask',
                             final_w['res7'], final_mask['res7'], 'res5', 'res7_mask')(Res6)

       Res8 = Residual_Layer(final_w['w16_conv'], final_w['b16_conv'],final_mask['w16_conv'], final_mask['b16_conv'], [1,2,2,1],'SAME', 'w16_conv', 'b16_conv', 'w16_mask', 'b16_mask',
                             final_w['w17_conv'], final_w['b17_conv'],final_mask['w17_conv'], final_mask['b17_conv'], 'w17_conv', 'b17_conv', 'w17_mask', 'b17_mask',
                             final_w['res8'], final_mask['res8'], 'res8', 'res8_mask')(Res7)

       Res9 = Residual_Layer(final_w['w18_conv'], final_w['b18_conv'],final_mask['w18_conv'], final_mask['b18_conv'], [1,2,2,1],'SAME', 'w18_conv', 'b18_conv', 'w18_mask', 'b18_mask',
                             final_w['w19_conv'], final_w['b19_conv'],final_mask['w19_conv'], final_mask['b19_conv'], 'w19_conv', 'b19_conv', 'w19_mask', 'b19_mask',
                             final_w['res9'], final_mask['res9'], 'res9', 'res9_mask')(Res8)

       flatten = tf.keras.layers.GlobalAvgPool2D()(Res9)
       fc1 = Dense(final_w['w20_fc'], final_w['b20_fc'],final_mask['w20_fc'], final_mask['b20_fc'], tf.keras.activations.linear, 'w20_fc', 'b20_fc', 'w20_mask', 'b20_mask')(flatten)
           
       model = tf.keras.models.Model(inp, fc1)
       return model

       

class ResNet18_v3(object):
  def __init__(self, datasource, num_classes, *initializer):
    self.num_classes = num_classes
    self.datasource = datasource
    self.name = 'resnet-18-v3'
    self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3]
    self.convinitializer = initializer[0]
    self.denseinitializer = initializer[1]
  
  def construct_weights(self, trainable=True):
      weight = {}
    
      weight['w1_conv'] = tf.Variable(self.convinitializer(shape=(3,3,3,16)), name='w1', trainable=trainable)
      #residual block1
      weight['w2_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w2', trainable=trainable)
      weight['w3_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w3', trainable=trainable)
      #residual block2
      weight['w4_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w4', trainable=trainable)
      weight['w5_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w5', trainable=trainable)
      #residual block3
      weight['w6_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w6', trainable=trainable)
      weight['w7_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,16)), name='w7', trainable=trainable)
     
      #residual block4
      weight['w8_conv'] = tf.Variable(self.convinitializer(shape=(3,3,16,32)), name='w8', trainable=trainable)
      weight['w9_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w9', trainable=trainable)
      weight['res4'] = tf.Variable(self.convinitializer(shape=(1,1,16,32)), name='res5', trainable=True)

      #residual block5
      weight['w10_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w10', trainable=trainable)
      weight['w11_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w11', trainable=trainable)
     

      #residual block6
      weight['w12_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w12', trainable=trainable)
      weight['w13_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,32)), name='w13', trainable=trainable)
      

      #residual block7
      weight['w14_conv'] = tf.Variable(self.convinitializer(shape=(3,3,32,64)), name='w14', trainable=trainable)
      weight['w15_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w15', trainable=trainable)
      weight['res7'] = tf.Variable(self.convinitializer(shape=(1,1,32,64)), name='res7', trainable=True)

      #residual block8
      weight['w16_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w16', trainable=trainable)
      weight['w17_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w17', trainable=trainable)
      

      #residual block9
      weight['w18_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w18', trainable=trainable)
      weight['w19_conv'] = tf.Variable(self.convinitializer(shape=(3,3,64,64)), name='w19', trainable=trainable)
      

      weight['w20_fc'] = tf.Variable(self.denseinitializer(shape=(64, self.num_classes)),name='w20', trainable=trainable)

      weight['b1_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b1', trainable=trainable)
      #residual block1
      weight['b2_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b2', trainable=trainable)
      weight['b3_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b3', trainable=trainable)
      #residual block2
      weight['b4_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b4', trainable=trainable)
      weight['b5_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b5', trainable=trainable)
      #residual block3
      weight['b6_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b6', trainable=trainable)
      weight['b7_conv'] = tf.Variable(tf.zeros_initializer()(shape=(16,)), name='b7', trainable=trainable)
      #residual block4
      weight['b8_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b8', trainable=trainable)
      weight['b9_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b9', trainable=trainable)
      #residual block5
      weight['b10_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b10', trainable=trainable)
      weight['b11_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b11', trainable=trainable)
      #residual block6
      weight['b12_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b12', trainable=trainable)
      weight['b13_conv'] = tf.Variable(tf.zeros_initializer()(shape=(32,)), name='b13', trainable=trainable)
      #residual block7
      weight['b14_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b14', trainable=trainable)
      weight['b15_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b15', trainable=trainable)
      #residual block8
      weight['b16_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b16', trainable=trainable)
      weight['b17_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b17', trainable=trainable)

      #residual block9
      weight['b18_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b18', trainable=trainable)
      weight['b19_conv'] = tf.Variable(tf.zeros_initializer()(shape=(64,)), name='b19', trainable=trainable)
      
      weight['b20_fc'] = tf.Variable(tf.zeros_initializer()(shape=(self.num_classes,)), name='b20', trainable=trainable)
      
      return weight

  def forward_pass(self,weights, inputs):
       #residual block1
       def _residual_block(inputs, dic1, dic2, st, dic3={}):
            shortcut = inputs
            inputs = tf.nn.conv2d(inputs, dic1['w'], [1, st, st, 1], 'SAME') + dic1['b']
            inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, dic2['w'], [1, 1, 1, 1], 'SAME') + dic2['b']
            x = tf.keras.layers.BatchNormalization()(inputs)
            if st != 1:
              shortcut = tf.nn.conv2d(shortcut, dic3['w'], [1,st,st,1], 'SAME')
              shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            x = tf.keras.layers.Add()([x,shortcut])
            x = tf.nn.relu(x)

            return x

       inputs = tf.nn.conv2d(inputs, weights['w1_conv'], [1, 1, 1, 1], 'SAME') + weights['b1_conv']
       inputs = tf.keras.layers.BatchNormalization()(inputs)
       inputs = tf.nn.relu(inputs)

       inputs =_residual_block(inputs, {'w': weights['w2_conv'], 'b': weights['b2_conv']},  {'w': weights['w3_conv'], 'b': weights['b3_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w4_conv'], 'b': weights['b4_conv']}, {'w': weights['w5_conv'], 'b': weights['b5_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w6_conv'], 'b': weights['b6_conv']},  {'w': weights['w7_conv'], 'b': weights['b7_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w8_conv'], 'b': weights['b8_conv']}, {'w': weights['w9_conv'], 'b': weights['b9_conv']}, 2, {'w':weights['res4']})
       inputs =_residual_block(inputs,  {'w': weights['w10_conv'], 'b': weights['b10_conv']}, {'w': weights['w11_conv'], 'b': weights['b11_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w12_conv'], 'b': weights['b12_conv']},  {'w': weights['w13_conv'], 'b': weights['b13_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w14_conv'], 'b': weights['b14_conv']},  {'w': weights['w15_conv'], 'b': weights['b15_conv']},2, {'w':weights['res7']})
       inputs =_residual_block(inputs, {'w': weights['w16_conv'], 'b': weights['b16_conv']},  {'w': weights['w17_conv'], 'b': weights['b17_conv']},1)
       inputs =_residual_block(inputs, {'w': weights['w18_conv'], 'b': weights['b18_conv']},  {'w': weights['w19_conv'], 'b': weights['b19_conv']},1)

       inputs = tf.reduce_mean(inputs, axis=[1,2])     #globalAveragePooling
       inputs = tf.matmul(inputs, weights['w20_fc']) + weights['b20_fc']
       return inputs

  def make_layers(self, final_w, final_mask):
       shape = tuple(self.input_dims)
       inp = tf.keras.layers.Input(shape= shape)

       conv1 = ConvTwo(final_w['w1_conv'], final_w['b1_conv'],final_mask['w1_conv'], final_mask['b1_conv'], [1,1,1,1],'SAME', 'w1_conv', 'b1_conv', 'w1_mask', 'b1_mask')(inp)
       batchnormal1 = tf.keras.layers.BatchNormalization()(conv1)
       Rel1 = tf.keras.layers.ReLU()(batchnormal1)

       #residual layer1
       shortcut = Rel1
       res1 = ConvTwo(final_w['w2_conv'], final_w['b2_conv'],final_mask['w2_conv'], final_mask['b2_conv'], [1,1,1,1],'SAME', 'w2_conv', 'b2_conv', 'w2_mask', 'b2_mask')(Rel1)
       res1 = tf.keras.layers.BatchNormalization()(res1)
       res1 = tf.keras.layers.ReLU()(res1)

       res1 = ConvTwo(final_w['w3_conv'], final_w['b3_conv'],final_mask['w3_conv'], final_mask['b3_conv'], [1,1,1,1],'SAME','w3_conv', 'b3_conv', 'w3_mask', 'b3_mask')(res1)
       res1 = tf.keras.layers.BatchNormalization()(res1)

       x = tf.keras.layers.Add()([shortcut, res1])
       x = tf.keras.layers.ReLU()(x)

       #residual layer2
       shortcut = x
       res2 = ConvTwo(final_w['w4_conv'], final_w['b4_conv'],final_mask['w4_conv'], final_mask['b4_conv'], [1,1,1,1],'SAME', 'w4_conv', 'b4_conv', 'w4_mask', 'b4_mask')(x)
       res2 = tf.keras.layers.BatchNormalization()(res2)
       res2 = tf.keras.layers.ReLU()(res2)
       
       res2 = ConvTwo(final_w['w5_conv'], final_w['b5_conv'],final_mask['w5_conv'], final_mask['b5_conv'], [1,1,1,1],'SAME','w5_conv', 'b5_conv', 'w5_mask', 'b5_mask')(res2)
       res2 = tf.keras.layers.BatchNormalization()(res2)

       x = tf.keras.layers.Add()([shortcut, res2])
       x = tf.keras.layers.ReLU()(x)
       
       #residual layer3
       shortcut = x
       res3 = ConvTwo(final_w['w6_conv'], final_w['b6_conv'],final_mask['w6_conv'], final_mask['b6_conv'], [1,1,1,1],'SAME', 'w6_conv', 'b6_conv', 'w6_mask', 'b6_mask')(x)
       res3 = tf.keras.layers.BatchNormalization()(res3)
       res3 = tf.keras.layers.ReLU()(res3)
       
       res3 = ConvTwo(final_w['w7_conv'], final_w['b7_conv'],final_mask['w7_conv'], final_mask['b7_conv'], [1,1,1,1],'SAME','w7_conv', 'b7_conv', 'w7_mask', 'b7_mask')(res3)
       res3 = tf.keras.layers.BatchNormalization()(res3)

       x = tf.keras.layers.Add()([shortcut, res3])
       x = tf.keras.layers.ReLU()(x)
       
      #residual layer4
       shortcut = x
       res4 = ConvTwo(final_w['w8_conv'], final_w['b8_conv'],final_mask['w8_conv'], final_mask['b8_conv'], [1,2,2,1],'SAME', 'w8_conv', 'b8_conv', 'w8_mask', 'b8_mask')(x)
       res4 = tf.keras.layers.BatchNormalization()(res4)
       res4 = tf.keras.layers.ReLU()(res4)
       
       res4 = ConvTwo(final_w['w9_conv'], final_w['b9_conv'],final_mask['w9_conv'], final_mask['b9_conv'], [1,1,1,1],'SAME','w9_conv', 'b9_conv', 'w9_mask', 'b9_mask')(res4)
       res4 = tf.keras.layers.BatchNormalization()(res4)

       shortcut = ConvTwo_v2(final_w['res4'], final_mask['res4'], [1,2,2,1],'SAME','res4', 'res4_mask')(shortcut)
       x = tf.keras.layers.Add()([shortcut, res4])
       x = tf.keras.layers.ReLU()(x)

      #residual layer5
       shortcut = x
       res5 = ConvTwo(final_w['w10_conv'], final_w['b10_conv'],final_mask['w10_conv'], final_mask['b10_conv'], [1,1,1,1],'SAME', 'w10_conv', 'b10_conv', 'w10_mask', 'b10_mask')(x)
       res5 = tf.keras.layers.BatchNormalization()(res5)
       res5 = tf.keras.layers.ReLU()(res5)
       
       res5 = ConvTwo(final_w['w11_conv'], final_w['b11_conv'],final_mask['w11_conv'], final_mask['b11_conv'], [1,1,1,1],'SAME','w11_conv', 'b11_conv', 'w11_mask', 'b11_mask')(res5)
       res5 = tf.keras.layers.BatchNormalization()(res5)

       x = tf.keras.layers.Add()([shortcut, res5])
       x = tf.keras.layers.ReLU()(x)

      #residual layer6
       shortcut = x
       res6 = ConvTwo(final_w['w10_conv'], final_w['b10_conv'],final_mask['w10_conv'], final_mask['b10_conv'], [1,1,1,1],'SAME', 'w10_conv', 'b10_conv', 'w10_mask', 'b10_mask')(x)
       res6 = tf.keras.layers.BatchNormalization()(res6)
       res6 = tf.keras.layers.ReLU()(res6)
       
       res6 = ConvTwo(final_w['w11_conv'], final_w['b11_conv'],final_mask['w11_conv'], final_mask['b11_conv'], [1,1,1,1],'SAME','w11_conv', 'b11_conv', 'w11_mask', 'b11_mask')(res6)
       res6 = tf.keras.layers.BatchNormalization()(res6)

       x = tf.keras.layers.Add()([shortcut, res6])
       x = tf.keras.layers.ReLU()(x)
       
      #residual layer7
       shortcut = x
       res7 = ConvTwo(final_w['w14_conv'], final_w['b14_conv'],final_mask['w14_conv'], final_mask['b14_conv'], [1,2,2,1],'SAME', 'w14_conv', 'b14_conv', 'w14_mask', 'b14_mask')(x)
       res7 = tf.keras.layers.BatchNormalization()(res7)
       res7 = tf.keras.layers.ReLU()(res7)
       
       res7 = ConvTwo(final_w['w15_conv'], final_w['b15_conv'],final_mask['w15_conv'], final_mask['b15_conv'], [1,1,1,1],'SAME',  'w15_conv', 'b15_conv', 'w15_mask', 'b15_mask')(res7)
       res7 = tf.keras.layers.BatchNormalization()(res7)

       shortcut = ConvTwo_v2(final_w['res7'], final_mask['res7'], [1,2,2,1],'SAME','res7', 'res7_mask')(shortcut)
       x = tf.keras.layers.Add()([shortcut, res7])
       x = tf.keras.layers.ReLU()(x)

      #residual layer8
       shortcut = x
       res8 = ConvTwo(final_w['w16_conv'], final_w['b16_conv'],final_mask['w16_conv'], final_mask['b16_conv'], [1,1,1,1],'SAME','w16_conv', 'b16_conv', 'w16_mask', 'b16_mask')(x)
       res8 = tf.keras.layers.BatchNormalization()(res8)
       res8 = tf.keras.layers.ReLU()(res8)
       
       res8 = ConvTwo(final_w['w17_conv'], final_w['b17_conv'],final_mask['w17_conv'], final_mask['b17_conv'],[1,1,1,1],'SAME','w17_conv', 'b17_conv', 'w17_mask', 'b17_mask')(res8)
       res8 = tf.keras.layers.BatchNormalization()(res8)

       x = tf.keras.layers.Add()([shortcut, res8])
       x = tf.keras.layers.ReLU()(x)
        
      
       #residual layer9
       shortcut = x
       res9 = ConvTwo(final_w['w18_conv'], final_w['b18_conv'],final_mask['w18_conv'], final_mask['b18_conv'], [1,1,1,1],'SAME', 'w18_conv', 'b18_conv', 'w18_mask', 'b18_mask')(x)
       res9 = tf.keras.layers.BatchNormalization()(res9)
       res9 = tf.keras.layers.ReLU()(res9)
       
       res9 = ConvTwo(final_w['w19_conv'], final_w['b19_conv'],final_mask['w19_conv'], final_mask['b19_conv'], [1,1,1,1],'SAME','w19_conv', 'b19_conv', 'w19_mask', 'b19_mask')(res9)
       res9 = tf.keras.layers.BatchNormalization()(res9)

       x = tf.keras.layers.Add()([shortcut, res9])
       x = tf.keras.layers.ReLU()(x)

       flatten = tf.keras.layers.GlobalAvgPool2D()(x)
       fc1 = Dense(final_w['w20_fc'], final_w['b20_fc'],final_mask['w20_fc'], final_mask['b20_fc'], tf.keras.activations.linear, 'w20_fc', 'b20_fc', 'w20_mask', 'b20_mask')(flatten)
           
       model = tf.keras.models.Model(inp, fc1)
       return model

       
