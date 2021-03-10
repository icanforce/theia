import tensorflow as tf
import numpy as np

#-----------------------------------------------------------------------------------------------------
def weight_variable(shape, name, initial=None):
    """weight_variable generates a weight variable of a given shape."""
    if initial is None:
        initial = tf.truncated_normal(shape, stddev=0.02, seed=0)    
    return tf.get_variable(
        name=name,
        initializer=initial
    )    
#-----------------------------------------------------------------------------------------------------
def bias_variable(shape, name, initial=None):
    """bias_variable generates a bias variable of a given shape."""
    if initial is None:
        initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(
        name=name,
        initializer=initial
    ) 
#-----------------------------------------------------------------------------------------------------    
def batchnorm(x, is_training, activation=None, epsilon=1e-4, momentum=0.5, name=None):
    out = tf.layers.batch_normalization(
        inputs=x,
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        training=is_training,
        name=name
    )
    return out if activation is None else activation(out)
#-----------------------------------------------------------------------------------------------------                          
def leaky_relu(x, leak=0.0):
    return tf.maximum(x, leak*x)
# -----------------------------------------------------------------------------------------------------
# def step_unit(input, flat):
    # o = tf.ones(shape=tf.shape(input))
    # _input = tf.where(input <= -1+flat, x=(-1.+flat)*o, y=input)
    # input_ = tf.where(_input >= 1-flat, x=(1-flat)*o, y=_input)
    # return tf.sin((np.pi/2)*input_/(1-flat)) 
# -----------------------------------------------------------------------------------------------------
# def step(x, flat=0.5, leak=0.2, scale=1.0):
    # pos = step_unit(x/scale-2*tf.floor((x/scale+1)/2.), flat) + 2*tf.floor((x/scale+1)/2.)    
    # return tf.maximum(x*leak, scale*pos)    
# -----------------------------------------------------------------------------------------------------
def step(input, flat=0.):
    tf.assert_positive(0.5-flat)
    o = tf.ones(shape=tf.shape(input))
    _input = tf.where(input <= flat, x=flat*o, y=input)
    input_ = tf.where(_input >= 1-flat, x=(1-flat)*o, y=_input)
    core = lambda a, f: (tf.sin((np.pi/2)*(2*(a-f)/(1-2*f)-1)) + 1)/2.
    return core(input_, flat)
# -----------------------------------------------------------------------------------------------------
def staircase(x, flat=0., leak=.0, width=.5, height=.5):
    positive = height*(step(x/width-tf.floor(x/width), flat) + tf.floor(x/width))
    negative = x*leak
    return tf.where(x < 0, x=negative, y=positive)
#-----------------------------------------------------------------------------------------------------     
def summary(x, summary_active=False):
    if summary_active:
        tensor_name = x.op.name
        tf.summary.tensor_summary(tensor_name, x)
        tf.summary.histogram(tensor_name, x)
#-----------------------------------------------------------------------------------------------------             
