# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:21:46 2021

@author: sense
"""

from gen_text_vectors import *
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import expand_dims
from transformer_layer import Encoder
from model_transformer import Discriminator,Generator
from random import randint, choice
import os
import time
from IPython.display import clear_output
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as pyplot

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def random_flip(image):
  image = tf.image.flip_left_right(image)
  return image.numpy()

def random_jitter(image):
  image = expand_dims(image, 0) #add additional dimension necessary for zooming
  image = image_augmentation_generator.flow(image, batch_size=1)
  result = image[0].reshape(image[0].shape[1:]) #remove additional dimension (1, 64, 64, 3) to (64, 64, 3)
  return result

image_augmentation_generator = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.8, 1.0])

df = pd.read_csv('final.csv')

data = pickle.load(open("image_vectors.p", "rb"))


''' '''
image_names = df['images'].values
cleaned_captions = clean_and_tokenize_comments_for_image(df['captions'].values)

n = 227
image_embeddings = []
captions = []
labels = []
for i, k in enumerate(data.keys()):
    if i % n == 0:
        image_embeddings.append(random_jitter(data[k]))
        labels.append(k)
    else:
        image_embeddings.append(data[k])
        labels.append(k)
        
top_k = 10000
caption_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
caption_tokenizer.fit_on_texts(cleaned_captions)
train_caption_seqs = caption_tokenizer.texts_to_sequences(cleaned_captions)

#new edit
print(caption_tokenizer.word_index)
caption_vocab = caption_tokenizer.word_index

caption_tokenizer.word_index['<pad>'] = 0
caption_tokenizer.index_word[0] = '<pad>'

train_caption_seqs = caption_tokenizer.texts_to_sequences(cleaned_captions)
caption_vector = tf.keras.preprocessing.sequence.pad_sequences(train_caption_seqs, padding='post')
max_length = calc_max_length(train_caption_seqs)
print(max_length)

max_length_captions = max_length


''' '''
def load_data():
    return np.asarray(image_embeddings), np.asarray(caption_vector).astype('float32')

images, lbs = load_data()

BUFFER_SIZE = images.shape[0]
BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices((images,lbs)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=len(caption_vocab),
#                maximum_position_encoding=10000)

discriminator = Discriminator(caption_vocab)#define_discriminator(len(caption_vocab),mask=None,training=False)
generator = Generator(caption_vocab)

# i = 0
# loss = 0
# for (b,(img,cap)) in enumerate(train_dataset):
#   if i != 1:
      
#         # img1,img2 = tf.split(img,2,0)
#         # cap1,cap2 = tf.split(cap,2,0)
#         # disc_output = discriminator(img1,cap1,training=False,mask=None)
#         # print(disc_output.shape)
        
#         latent_dim = 100
#         n_batch = 64
#         noise_1 = tf.random.normal([32, latent_dim])
#         noise_2 = tf.random.normal([32, latent_dim])
#         noise = tf.concat([noise_1, noise_2], 0)
        
#         generated_images = generator(noise,cap,False,mask=None)
#         print(generated_images.shape)
#         i += 1
        
#         padding_mask1 = create_masks(cap1)
#         img1,img2 = tf.split(generated_images,2,0)
#         cap1,cap2 = tf.split(cap,2,0)
#         disc_output = discriminator(img1,cap1,training=True,mask=padding_mask1)
#         print(disc_output.shape)
        
        
#   else:
#       break


'''Training '''
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
ms_loss_weight = 1.0
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.000035, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.000035, beta_1 = 0.5)

def smooth_positive_labels(y):
	return y - 0.3 + (np.random.random(y.shape) * 0.5)
 
def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.3

# Randomly flip some labels. Credits to https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
def noisy_labels(y, p_flip):
    n_select = int(p_flip * int(y.shape[0]))
    flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)

    op_list = []
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1.0, y[i]))
        else:
            op_list.append(y[i])

    outputs = tf.stack(op_list)
    return outputs

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
  
    return enc_padding_mask

def generate_random_vectors(n_samples):  
  vectorized_random_captions = []

  for n in range(n_samples):
    rnd = randint(8, 25)
    result_array = np.empty((0, 50))
    for i in range(rnd):
      result_array = np.append(result_array, [choice(caption_vector)], axis=0)
    vectorized_random_captions.append(np.mean(result_array, axis=0).astype('float32'))

  return np.array(vectorized_random_captions)

def discriminator_loss(r_real_output_real_text, f_fake_output_real_text_1, f_real_output_fake_text):
   alpha = 0.5
   real_output_noise = smooth_positive_labels(noisy_labels(tf.ones_like(r_real_output_real_text), 0.10))
   fake_output_real_text_noise_1 = smooth_negative_labels(tf.zeros_like(f_fake_output_real_text_1))
   real_output_fake_text_noise = smooth_negative_labels(tf.zeros_like(f_real_output_fake_text))

   real_loss = tf.reduce_mean(binary_cross_entropy(real_output_noise, r_real_output_real_text))
   fake_loss_ms_1 = tf.reduce_mean(binary_cross_entropy(fake_output_real_text_noise_1, f_fake_output_real_text_1))
   fake_loss_2 = tf.reduce_mean(binary_cross_entropy(real_output_fake_text_noise, f_real_output_fake_text))

   total_loss = real_loss + alpha * fake_loss_2 + (1-alpha) * fake_loss_ms_1 
   return total_loss

def generator_loss(f_fake_output_real_text):
   return tf.reduce_mean(binary_cross_entropy(tf.ones_like(f_fake_output_real_text), f_fake_output_real_text))

@tf.function
def train_step(images, epoch, image_batch_shape):

  #define half_batch
  latent_dim = 100
  n_batch = 64

  noise_1 = tf.random.normal([int(image_batch_shape/2), latent_dim])#32
  noise_2 = tf.random.normal([int(image_batch_shape/2), latent_dim])#32
  real_captions = images[1]
  real_images = images[0]

  random_captions = generate_random_vectors(image_batch_shape)#n_batch
  random_captions_1, random_captions_2  = tf.split(random_captions, 2, 0)
  real_captions_1, real_captions_2  = tf.split(real_captions, 2 ,0)
  real_images_1, real_images_2 = tf.split(real_images, 2, 0)
  
  real_padding_mask = create_masks(real_captions)
  real_padding_mask1 = create_masks(real_captions_1)
  real_padding_mask2 = create_masks(real_captions_2)
  random_padding_mask1 = create_masks(random_captions_1)
  random_padding_mask2 = create_masks(random_captions_2)

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    
    noise = tf.concat([noise_1, noise_2], 0)

    generated_images = generator(noise, real_captions, training=True, mask=real_padding_mask)

    fake_1, fake_2 = tf.split(generated_images, 2, 0)
    # print("Shape of fake", fake_1.shape)
    # print("shape of real_captions_1", real_captions_1)
    f_fake_output_real_text_1 = discriminator(fake_1, real_captions_1, training=True, mask=real_padding_mask1)
    f_fake_output_real_text_2 = discriminator(fake_2, real_captions_2, training=True, mask=real_padding_mask2)

    r_real_output_real_text_1 = discriminator(real_images_1, real_captions_1, training=True, mask=real_padding_mask1)
    r_real_output_real_text_2 = discriminator(real_images_2, real_captions_2, training=True, mask=real_padding_mask2)

    f_real_output_fake_text_1 = discriminator(real_images_1, random_captions_1, training=True, mask=random_padding_mask1)
    f_real_output_fake_text_2 = discriminator(real_images_2, random_captions_2, training=True, mask=random_padding_mask2)

    #### Calculating losses ####

    gen_loss = generator_loss(f_fake_output_real_text_1) + generator_loss(f_fake_output_real_text_2) 
    # mode seeking loss
    
    lz = tf.math.reduce_mean(tf.math.abs(fake_2-fake_1)) / tf.math.reduce_mean(tf.math.abs(noise_2-noise_1))
    eps = 1 * 1e-5
    loss_lz = 1 / (eps+lz) * ms_loss_weight
    total_gen_loss = gen_loss + loss_lz

    tf.print('G_loss', [total_gen_loss])

    disc_loss_1 = discriminator_loss(r_real_output_real_text_1, f_fake_output_real_text_1, f_real_output_fake_text_1)
    disc_loss_2 = discriminator_loss(r_real_output_real_text_2, f_fake_output_real_text_2, f_real_output_fake_text_2)
    
    total_disc_loss = disc_loss_1 + disc_loss_2

    tf.print('D_loss', [total_disc_loss])

    #### Done calculating losses ####

  gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)  

  gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)    

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def train(dataset, epochs = 100):

  checkpoint_dir = 'checkpoints'
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)
  
  ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
  if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint)  #ckpt_manager.checkpoints[3]
    print ('Latest checkpoint restored!!')

  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch, epoch, image_batch[1].shape[0]) 


    if (epoch + 1) % 40 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))

    if (epoch +1) % 60 == 0:
    
      clear_output(wait=True)
      generator.save('/content/drive/My Drive/46stage_new_gan_animal_model_%03d.h5' % (epoch + 1))     

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
train(train_dataset)
