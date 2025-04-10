# In[1]:


import tensorflow as tf

import numpy as np
import os
import pathlib
from PIL import Image

from matplotlib import pyplot as plt
from IPython import display
from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())


# In[2]:

# Check if TensorFlow can detect the GPU
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# # Optionally, check detailed GPU info
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         print(f"GPU Name: {gpu.name}")


# In[3]:


# PATH = pathlib.Path(r'C:\Users\home\Landscapes')


# In[5]:


# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


# In[6]:


def my_load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image, channels = 3)
  # image = tf.image.decode_image(image, channels=3) 
  w = tf.shape(image)[1]

  input_image = image[:, :w, :]

  # Convert image to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  

  return input_image


# In[7]:

# Ensure its 256x256
# creating these to initialize the model before hand
inp = my_load(r'D:\Image\dummy\1.png')
# Casting to int for matplotlib to display the images
plt.figure()
plt.imshow(inp / 255.0)


# In[8]:


OUTPUT_CHANNELS = 3


# In[9]:


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


# In[10]:


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)


# In[11]:


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# In[12]:


up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)


# In[13]:


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


# In[14]:


generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)


# In[15]:


gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])


# In[16]:


def my_resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image


# In[17]:


# Normalizing the images to [-1, 1]
def my_normalize(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image


# In[18]:


import numpy as numpy
def my_load_image_test(image_file):
  input_image = my_load(image_file)
  input_image = my_resize(input_image,
                                    IMG_HEIGHT, IMG_WIDTH)
  input_image = my_normalize(input_image)

  return input_image


# In[19]:


tf.keras.backend.clear_session()


# ### USING SAVED MODEL FOR IMAGE GENERATION

# In[20]:


checkpoint_dir = './training_checkpoints_landscape'  
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


# In[21]:


# Create a checkpoint object for restoring
checkpoint = tf.train.Checkpoint(generator=generator)

# Restore the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()
    print(f"Checkpoint restored from {latest_checkpoint}")
else:
    print("No checkpoint found in the directory.")

# In[23]:


def generate_images(model, test_input):
    prediction = model(test_input, training=True)

    # Convert prediction to numpy array
    generated_image = prediction[0].numpy()  # Convert to numpy array if necessary

    # Rescale from [-1, 1] to [0, 1]
    generated_image = (generated_image + 1) * 0.5  # Rescale to [0, 1]

    # Now scale to [0, 255]
    generated_image = (generated_image * 255).astype(np.uint8)  # Ensure correct range and dtype

    # Create a PIL image directly from prediction[0]
    generated_image_pil = Image.fromarray(generated_image)

    # Display the input and predicted images
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    plt.figure(figsize=(15, 15))
    
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # Optional: scaling for display purposes
        plt.axis('off')
    plt.show()

    return generated_image_pil


# In[26]:


def get_image():
    # print("here")
    test_dataset1 = tf.data.Dataset.list_files(r'D:\Image\predict\*.png')
    
    print(test_dataset1.cardinality())
    if(test_dataset1.cardinality() == 0) :
      return
    
    test_dataset1 = test_dataset1.map(my_load_image_test)
    
    test_dataset1 = test_dataset1.take(1) 
    test_dataset1 = test_dataset1.batch(1)
    

    for inp in test_dataset1.take(1):
        # print("here")
        generated_image = generate_images(generator, inp)
        
        generated_image_path = r'D:\Image\output\generated_image.png'
        
        generated_image.save(generated_image_path)
        print(f"Generated image saved to {generated_image_path}")
    
# In[27]:
# if __name__ == "__main__":
#     # Only execute when running this file directly, not on import
#     get_image()
