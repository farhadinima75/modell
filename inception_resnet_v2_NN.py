# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception Resnet V2 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import initializers, l2_regularizer

import tensorflow as tf

slim = tf.contrib.slim

padding = 'SAME'


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 16, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 16, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 16, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 24, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)
  return net


######################################################################################################################################
######################################################################################################################################
weight_decay = 1e-4
activation_fn = tf.nn.relu

def nim_bot(inputs, num_filter, scope):

  x = slim.batch_norm(inputs, epsilon=0.001, scope=scope + 'bn')
  x = tf.nn.relu(x, name= scope + 'relu')
  x = slim.conv2d(x, num_filter, 3, stride=1, padding=padding, weights_initializer=initializers.xavier_initializer(uniform = True),
                    weights_regularizer=l2_regularizer(weight_decay), activation_fn=activation_fn, scope=scope +'conv')
    
  return x


def nim_conv(inputs, num_filter, scope):

  x = slim.batch_norm(inputs, epsilon=0.001, scope=scope + 'bn1')
  x = tf.nn.relu(x, name= scope + 'relu1')
  x = slim.conv2d(x, num_filter, 1, stride=1, padding=padding, weights_initializer=initializers.xavier_initializer(uniform = True),
                    weights_regularizer=l2_regularizer(weight_decay), activation_fn=activation_fn, scope=scope +'conv1')
    
  return x


def nim_cell(input, nb_filters, name):

#         net_1 = nim_bot(input, (M-2) * cell_largness, name + 'nim_conv_1_')
#         net_2_1 = nim_conv(input, (M-1) * cell_largness,  name + '1nim_conv_2_')
     #    net_1 = nim_bot(input, cell_largness, name + 'nim_conv_1_')
         net_1 = nim_bot(input,  nb_filters,  name + 'nim_conv_1_')
         #net_2 = nim_conv(input,  nb_filters,  name + 'nim_conv_2_')
        
         sub2_1 = tf.concat(axis=3, values=[net_1, input])	
     #    sub2_1 = nim_bot(sub2_1, 10, name + 'bottle')
      
         return sub2_1


def resnet_v1_block(x, num_filter, repeat, growth, name):
    
#  x = nim_cell(x, num_filter, name)
  nb_filter =+ growth     
    
  for i in range(repeat):
    
    x = nim_cell(x, num_filter, name + str(i+1))
    nb_filter =+ growth    
  return x, nb_filter

#######################################################################################################################################
#######################################################################################################################################


def inception_resnet_v2_base(inputs,
                             final_endpoint='PreAuxLogits',
                             output_stride=16,
                             align_feature_maps=True,
                             scope=None,
                             activation_fn=tf.nn.relu):
  """Inception model from  http://arxiv.org/abs/1602.07261.
  Constructs an Inception Resnet v2 network from inputs to the given final
  endpoint. This method can construct the network up to the final inception
  block Conv2d_7b_1x1.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    output_stride: A scalar that specifies the requested ratio of input to
      output spatial resolution. Only supports 8 and 16.
    align_feature_maps: When true, changes all the VALID paddings in the network
      to SAME padding so that the feature maps are aligned.
    scope: Optional variable_scope.
    activation_fn: Activation function for block scopes.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
      or if the output_stride is not 8 or 16, or if the output_stride is 8 and
      we request an end point after 'PreAuxLogits'.
  """
  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  padding = 'SAME' if align_feature_maps else 'VALID'

  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
###################################################################
        ###############################################################
      activation_fn=tf.nn.relu

      with tf.variable_scope('Nim_block_1'):
        with tf.variable_scope('Branch_0'):
          
          nb_filter = 32
          net = inputs
        
          net = slim.conv2d(net, nb_filter, 3, stride=1, padding=padding, weights_initializer=initializers.xavier_initializer(uniform = True),
                          weights_regularizer=l2_regularizer(weight_decay), activation_fn=activation_fn, scope='conv')
          net = slim.batch_norm(net, epsilon=0.001, scope='bn')
          net = tf.nn.relu(net, name= 'relu')
		
          net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)
          net, nb_filter = resnet_v1_block(net, nb_filter, 8, 12, 'block1')    ##32
          net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)
          net, nb_filter = resnet_v1_block(net, nb_filter, 12, 16, 'block2')    ##32
          net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)
          net, nb_filter = resnet_v1_block(net, nb_filter, 72, 20, 'block3')
         # net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)   ##32
          net, nb_filter = resnet_v1_block(net, nb_filter, 8, 25, 'block4')    ##32
       #   net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)
       #   net, nb_filter = resnet_v1_block(net, nb_filter, 1, 4, 'block5')
       #   net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)
       #   net, nb_filter = resnet_v1_block(net, nb_filter, 1, 4, 'block6')
       #   net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)
       #   net, nb_filter = resnet_v1_block(net, nb_filter, 1, 4, 'block7')
        #  net = slim.avg_pool2d(net, (2,2), stride = 2, padding=padding)
          # https://stackoverflow.com/questions/38160940/ ...


						
      if add_and_check_final('PreAuxLogits', net): return net, end_points	

######################################################################################################################################
	  

def inception_resnet_v2(inputs, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2',
                        create_aux_logits=True,
                        activation_fn=tf.nn.relu):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
      Dimension batch_size may be undefined. If create_aux_logits is false,
      also height and width may be undefined.
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before  dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxilliary logits.
    activation_fn: Activation function for conv2d.

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(
      scope, 'InceptionResnetV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

      net, end_points = inception_resnet_v2_base(inputs, scope=scope,
                                                 activation_fn=activation_fn)

      if create_aux_logits and num_classes:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          #aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
          #                      scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux

      with tf.variable_scope('Logits'):
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_8x8')
        else:
          net = tf.reduce_mean(
              input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
        if not num_classes:
          return net, end_points
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')
        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points
inception_resnet_v2.default_image_size = 299


def inception_resnet_v2_arg_scope(
    weight_decay=0.00004,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    activation_fn=tf.nn.relu,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
    batch_norm_scale=False):
  """Returns the scope with the default parameters for inception_resnet_v2.

  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: Activation function for conv2d.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': batch_norm_updates_collections,
        'fused': None,  # Use fused batch norm if possible.
        'scale': batch_norm_scale,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope
