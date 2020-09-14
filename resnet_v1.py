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
"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_utils

from tensorflow.contrib.layers.python.layers import initializers, l2_regularizer


resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim


class NoOpScope(object):
  """No-op context manager."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


weight_decay = 1e-4
activation_fn = tf.nn.relu

def nim_conv(inputs, num_filter, scope):

  x = slim.batch_norm(inputs, epsilon=0.001, scope=scope + 'bn')
  x = tf.nn.relu(x, name= scope + 'relu')
  x = slim.conv2d(x, num_filter, 3, stride=1, weights_initializer=initializers.xavier_initializer(uniform = True),
                    weights_regularizer=l2_regularizer(weight_decay), activation_fn=activation_fn, scope=scope +'conv')
    
  return x

def nim_cell(input, num_filter, name):

         net_1 = nim_conv(input,  num_filter//2,  name + 'nim_conv_1_')
         net_2 = nim_conv(net_1,  num_filter//4,  name + 'nim_conv_2_')
         net_3 = nim_conv(net_2,  num_filter//6,  name + 'nim_conv_3_')
  
         sub2_1 = tf.concat(axis=3, values=[net_3, input])
      
         return sub2_1


def resnet_v1_block(x, num_filter, repeat, growth, name):
    
#  x = nim_cell(x, num_filter, name)
  nb_filter =+ growth     
    
  for i in range(repeat):
    
    x = nim_cell(x, num_filter, name + str(i+1))
    nb_filter =+ growth    
  return x, nb_filter



def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              store_non_strided_activations=False,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes a non-zero integer, net contains the
      pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with (slim.arg_scope([slim.batch_norm], is_training=is_training)
            if is_training is not None else NoOpScope()):
        nb_filter = 32
        net = inputs
        
        net = nim_conv(net, nb_filter, 'first_conv')
        nb_filter = nb_filter*2
        net, nb_filter = resnet_v1_block(net, nb_filter, 5, 16, 'block1')
        net = slim.avg_pool2d(net, (2,2), stride = 2)
        net, nb_filter = resnet_v1_block(net, nb_filter, 5, 16, 'block2')
        net = slim.avg_pool2d(net, (2,2), stride = 2)
        net, nb_filter = resnet_v1_block(net, nb_filter, 5, 16, 'block3')
        net = slim.avg_pool2d(net, (2,2), stride = 2)
        net, nb_filter = resnet_v1_block(net, nb_filter, 5, 16, 'block4')
        net = slim.avg_pool2d(net, (2,2), stride = 2)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points
resnet_v1.default_image_size = 224


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 store_non_strided_activations=False,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block(inputs, 32, 5, 16, 'block3')
   #   resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
    #  resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
   #   resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
resnet_v1_50.default_image_size = resnet_v1.default_image_size


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  store_non_strided_activations=False,
                  reuse=None,
                  scope='resnet_v1_101'):
  """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block(inputs, 32, 5, 16, 'block')
   #   resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
    #  resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
   #   resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
resnet_v1_101.default_image_size = resnet_v1.default_image_size


def resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  store_non_strided_activations=False,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_152'):
  """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block(inputs, 32, 5, 16, 'block')
   #   resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
    #  resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
   #   resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
resnet_v1_152.default_image_size = resnet_v1.default_image_size


def resnet_v1_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  store_non_strided_activations=False,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_200'):
  """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block(inputs, 32, 5, 16, 'block')
   #   resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
    #  resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
   #   resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   store_non_strided_activations=store_non_strided_activations,
                   reuse=reuse, scope=scope)
resnet_v1_200.default_image_size = resnet_v1.default_image_size
