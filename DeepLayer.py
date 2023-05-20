import tensorflow as tf

class ConvTwo(tf.keras.layers.Layer):
  def __init__(self, weight, bias, weightmask, biasmask, stridesList, padding, weightname, biasname, W_maskname, b_maskname):
      super(ConvTwo,self).__init__()
      self.w = tf.Variable(initial_value=weight, trainable=True, name =weightname)
      self.b = tf.Variable(initial_value=bias, trainable=True, name=biasname)
      self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False, name = W_maskname)
      self.bias_mask = tf.Variable(initial_value=biasmask, trainable=False, name = b_maskname)
      self.strides = stridesList
      self.padding = padding
    
  def call(self, inputs):
      w = self.w * self.weight_mask
      b = self.b * self.bias_mask
      return tf.nn.conv2d(inputs, w, self.strides, self.padding) + b



class Dense(tf.keras.layers.Layer):
  def __init__(self, weight, bias, weightmask, biasmask, activation, weightname, biasname, W_maskname, b_maskname):
     super(Dense, self).__init__()
     self.w = tf.Variable(initial_value=weight, trainable=True, name=weightname)
     self.b = tf.Variable(initial_value=bias, trainable=True, name= biasname)
     self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False, name = W_maskname)
     self.bias_mask = tf.Variable(initial_value=biasmask, trainable=False, name = b_maskname)
     self.activation = activation
  def call(self,input):
      w = self.w * self.weight_mask
      b = self.b * self.bias_mask
      return self.activation(tf.matmul(input, w) + b)


class ConvTwoTranspose(tf.keras.layers.Layer):
    def __init__(self, weight, weightmask, outputshape, stridesList, padding, weightname, W_maskname ):
        super(ConvTwoTranspose,self).__init__()
        self.w = tf.Variable(initial_value=weight, trainable=True, name=weightname)
        self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False, name = W_maskname)
        self.outputshape = outputshape
        self.strides = stridesList
        self.padding = padding

    def call(self, inputs):
        w = self.w * self.weight_mask
        return tf.nn.conv2d_transpose(inputs, w, self.outputshape, self.strides, self.padding) 

class DenseGan(tf.keras.layers.Layer):
    def __init__(self, weight, weightmask, activation, weightname, W_maskname):
        super(DenseGan, self).__init__()
        self.w = tf.Variable(initial_value=weight, trainable=True, name=weightname)
        self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False,  name = W_maskname)
        self.activation = activation
    def call(self,input):
        w = self.w * self.weight_mask
        return self.activation(tf.matmul(input, w))

class ConvTwo_v2(tf.keras.layers.Layer):
    def __init__(self, weight,weightmask, stridesList, padding, weightname, W_maskname ):
        super(ConvTwo_v2,self).__init__()
        self.w = tf.Variable(initial_value=weight, trainable=True, name=weightname)
        self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False, name = W_maskname)
        self.strides = stridesList
        self.padding = padding
    def call(self, inputs):
        w = self.w * self.weight_mask
        return tf.nn.conv2d(inputs, w, self.strides, self.padding) 
