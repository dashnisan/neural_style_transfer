#!/usr/bin/env python
# coding: utf-8

####################################################################
## Packages
####################################################################

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import pprint
import marshal

s = open('nst_functions.cpython-310.pyc', 'rb')
s.seek(16)  # go past first 16 bytes
code_obj = marshal.load(s)
exec(code_obj)

####################################################################
# Print available GPU device
####################################################################
print(tf.config.list_physical_devices('GPU'))

#####################################################################
# Model: VGG
#################################################################### 

tf.random.set_seed(272) # DO NOT CHANGE content_outputTHIS VALUE
pp = pprint.PrettyPrinter(indent=4)
img_size = 500
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='../model/vgg19_weights.h5')

vgg.trainable = False
pp.pprint(vgg)

#####################################################################
# Optimization parameters:
epochs = 10001
learning_rate=0.002

#####################################################################
# IO parameters
content_im = "../images/van_gogh_portrait_1.jpg"
style_im = "../images/stones.jpg"

# create out dir if necessary:
out_desc='vg_goes_stones_10k_lr2e-3'
if not os.path.isdir(f"output/{out_desc}"):
    os.makedirs(f"output/{out_desc}")

#####################################################################
# Load content image C:
content_image = np.array(Image.open(content_im).resize((img_size, img_size)))
print(content_image.shape)
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
print(content_image.shape)
imshow(content_image[0])
plt.show()

# Load style image S:
style_image =  np.array(Image.open(style_im).resize((img_size, img_size)))
print(style_image.shape)
#imshow(style_image)
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
print(style_image.shape)
imshow(style_image[0])
plt.show()


# Initialiaze generated image G: noise + content

generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

print(generated_image.shape)
#imshow(generated_image.numpy()[0])
#plt.show()

#####################################################################
# Set style and content layers:
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.4)]

content_layer = [('block5_conv4', 1)] # analogue to STYLE_LAYERS
#####################################################################

# Get all the used activations for C and G images:
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
print(type(vgg_model_outputs))
print(vgg_model_outputs)


# Save activations for C:
content_target = vgg_model_outputs(content_image)  # Content encoder
print(type(content_target))
print(len(content_target))
print(content_target[0].shape)

# Save activations for S:
style_targets = vgg_model_outputs(style_image)     # Style encoder
print(len(style_targets))
print(style_targets[0].shape)


# C weights and output:
#print(type(content_layer))
#print(content_layer[0][0])
cw = vgg.get_layer(content_layer[0][0]).weights
co = vgg.get_layer(content_layer[0][0]).output

print('\n C WEIGHTS:')
print(type(cw))
print(len(cw))
print(cw[0].shape)
print(cw[1].shape)

print('\n C OUTPUT:')
print(type(co))
print(co.shape)

print(vgg.get_layer(content_layer[0][0]).name)

#####################################################################
new_model = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C2 = new_model(preprocessed_content)


# S weights:
#print(type(STYLE_LAYERS))
#print(STYLE_LAYERS[0][0])
#print(STYLE_LAYERS[-1][0])
sw0 = vgg.get_layer(STYLE_LAYERS[0][0]).output
#print(type(STYLE_LAYERS))
print('S layers shape {}'.format(sw0.shape))



# Assign the content image to be the input of the VGG model.  
# Set a_C to be the hidden layer activation from the selected layer:
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
print(type(preprocessed_content))
print(preprocessed_content.shape)

#a_C = vgg_model_outputs(preprocessed_content)[5] # this applies the selected activation to C
a_C = vgg_model_outputs(preprocessed_content)[4] # sthis applies the selected activation to C


print(type(a_C))
print(a_C.shape)


# Assign the input of the model to be the "style" image 
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
print(preprocessed_style.shape)

a_S = vgg_model_outputs(preprocessed_style)


#####################################################################
# OPTIMIZATION
#####################################################################
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
generated_image = tf.Variable(generated_image)

@tf.function()
def train_step(generated_image):
    print(type(generated_image))
    print(generated_image.shape)
    with tf.GradientTape() as tape:
        # Compute a_G as the vgg_model_outputs for the current generated image
        #(1 line)
        a_G = vgg_model_outputs(generated_image)
        print(type(a_G))
        print(len(a_G))
        
        # Compute the costs:
        J_content = compute_content_cost(a_C, a_G)
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)
        J = total_cost(J_content, J_style, alpha = 10, beta = 40)
                
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J

# Training
# Display training progress:
for i in range(epochs):
    train_step(generated_image)
    if i % 1000 == 0:
        print(f"Epoch {i} ")
    if i % 1000 == 0:
        image = tensor_to_image(generated_image)
        #imshow(image)
        image.save(f"output/{out_desc}/image_{i}.jpg")
        #plt.show() 


# Show C S G in a row
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text('CONTENT IMAGE')
ax = fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text('STYLE IMAGE')
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('NEURAL TRANSFER STYLE IMAGE')
#plt.show()
plt.savefig(f"output/{out_desc}/all.jpg")


