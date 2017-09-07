#By @Kevin Xu
#kevin28520@gmail.com

# 深度学习QQ群, 1群满): 153032765
# 2群：462661267
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.


# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%
import tensorflow as tf
import numpy as np
import os
import math
#%%

# you need to change this to your data directory
train_dir = '/python/cat_emotion/data/train/'

def get_files(file_dir,ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    happycat = []
    label_happycat = []
    angrycat = []
    label_angrycat = []
    curiouscat = []
    label_curiouscat = []
    relaxedcat = []
    label_relaxedcat = []
    negativecat = []
    label_negativecat = []
    
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='happycat':
            happycat.append(file_dir + file)
            label_happycat.append(1)
        elif name[0]=='angrycat':
            angrycat.append(file_dir + file)
            label_angrycat.append(2)
        elif name[0]=='curiouscat':
            curiouscat.append(file_dir + file)
            label_curiouscat.append(3)
        elif name[0]=='relaxedcat':
            relaxedcat.append(file_dir + file)
            label_relaxedcat.append(4)
        elif name[0]=='negativecat':
            negativecat.append(file_dir + file)
            label_negativecat.append(0)
        
    print('There are %d happycats\nThere are %d angrycats\nThere are %d curiouscats\nThere are %d relaxedcats\nThere are %d negativecats' 
          %(len(happycat), len(angrycat), len(curiouscat), len(relaxedcat), len(negativecat)))
    
    image_list = np.hstack((happycat, angrycat, curiouscat, relaxedcat, negativecat))
    label_list = np.hstack((label_happycat, label_angrycat, label_curiouscat, label_relaxedcat, label_negativecat))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples
    
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    
    
    return tra_images,tra_labels,val_images,val_labels


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        #原教程为rgb,3 channels;现要用gray,所以是1 channel？
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

      
    #data argumentation，remain to be tested
#    image = tf.random_crop(image, [156,156, 3])# randomly crop the image size to 156 x 156
#    image = tf.image.random_flip_left_right(image)
#    image = tf.image.random_brightness(image, max_delta=63)
#    image = tf.image.random_contrast(image,lower=0.2,upper=1.8)  
    # if you want to test the generated batches of images, you might want to comment the following line.
    # ！！注意：如果想看到正常的图片，请注释掉（标准化）和 （image_batch = tf.cast(image_batch, tf.float32)）
    # 但是训练时不要注释掉！！！！！
    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch


#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes
#    
#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 2
##capacity有什么讲究吗？
#CAPACITY = 256
#IMG_W = 208
#IMG_H = 208
#
#train_dir = '/python/cat_emotion/data/train/'
#ratio=0.2
#tra_images, tra_labels, val_images, val_labels = get_files(train_dir, ratio)
#tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([tra_image_batch, tra_label_batch])
#            
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%