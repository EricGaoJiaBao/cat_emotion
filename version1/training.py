#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
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
import os
import numpy as np
import tensorflow as tf
import input_data
import model

#%%

N_CLASSES = 5
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 32 #原教程有25000张图，不能超过64，对电脑性能要求高
CAPACITY = 500 #2000 
MAX_STEP = 1000 # 7500也够了其实;with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.001 # with current parameters, it is suggested to use learning rate<0.0001


#%%
def run_training():
    
    # you need to change the directories to yours.
    train_dir = '/python/cat_emotion/data/train/'
    logs_train_dir = '/python/cat_emotion/logs3/train/'
    
    train, train_label = input_data.get_files(train_dir)
    
    train_batch, train_label_batch = input_data.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE, 
                                                  CAPACITY)
         
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)        
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
       
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
            if step % 25 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 200 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
    
#%%Evaluation function
#import math
#test_dir = '/python/cat_emotion/data/test/'
#train_dir = '/python/cat_emotion/data/train/'
#n_test = 255
#n_train = 1024
#
#def evaluate(data_dir,n_picture):
#    name = data_dir.split(sep='/')
#    name = name[-2]
#    print('On the %sing data:' %name)
#    with tf.Graph().as_default():        
#        log_dir = '/python/cat_emotion/logs3/train/'   
#        N_CLASSES = 5
#        IMG_W = 208
#        IMG_H = 208
#        BATCH_SIZE = 32
#        CAPACITY = 500
#        # reading test data
#        image_list, label_list = input_data.get_files(data_dir)
#        images, labels = input_data.get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#        logits = model.inference(images, BATCH_SIZE, N_CLASSES)
#        top_k_op = tf.nn.in_top_k(logits, labels, 1)
#        saver = tf.train.Saver(tf.global_variables())
#        
#        with tf.Session() as sess:
#            
#            print("Reading checkpoints...")
#            ckpt = tf.train.get_checkpoint_state(log_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
#            else:
#                print('No checkpoint file found')
#                return
#        
#            coord = tf.train.Coordinator()
#            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
#            
#            try:
#                num_iter = int(math.ceil(n_picture / BATCH_SIZE))
#                true_count = 0
#                total_sample_count = num_iter * BATCH_SIZE
#                step = 0
#
#                while step < num_iter and not coord.should_stop():
#                    predictions = sess.run([top_k_op])
#                    true_count += np.sum(predictions)
#                    step += 1
#                    precision = true_count / total_sample_count 
#                print('%d correctly predicted in total %d pictures' %(true_count,total_sample_count))
#                print('%.3f%% accuracy on %sing data' % (precision*100,name))
#            except Exception as e:
#                coord.request_stop(e)
#            finally:
#                coord.request_stop()
#                coord.join(threads)
#    
#evaluate(test_dir,n_test)
#print('\n\n')
#evaluate(train_dir,n_train)

#%% Evaluate one image
# when training, comment the following codes.

#
#from PIL import Image
#import matplotlib.pyplot as plt
#
#def get_one_image(test):
#    '''Randomly pick one image from testing data(depend on the dir)
#    Return: ndarray
#    '''
#    n = len(test)
#    ind = np.random.randint(0, n)
#    img_dir = test[ind]
#
#    image = Image.open(img_dir)
#    plt.imshow(image)
#    image = np.array(image)
#    return image
#
#def evaluate_one_image():
#    '''Test one image against the saved models and parameters
#    '''
#    
#    # you need to change the directories to yours.
#    test_dir = '/python/cat_emotion/data/test/'#here we use training data
#    test, test_label = input_data.get_files(test_dir)
#    image_array = get_one_image(test)
#    
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        N_CLASSES = 5
#        
#        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image, [1, 208, 208, 3])
#        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#        
#        logit = tf.nn.softmax(logit)
#        
#        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#        
#        # you need to change the directories to yours.
#        logs_train_dir = '/python/cat_emotion/logs2/train/' 
#                       
#        saver = tf.train.Saver()
#        
#        with tf.Session() as sess:
#
#            print("Reading checkpoints...")
#            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
#            else:
#                print('No checkpoint file found')
#            
#            prediction = sess.run(logit, feed_dict={x: image_array})
#            max_index = np.argmax(prediction)
#            if max_index==0:
#                print('This is a negativecat with possibility %.6f' %prediction[:, 0])
#            elif max_index==1:
#                print('This is a happycat with possibility %.6f' %prediction[:, 1])       
#            elif max_index==2:
#                print('This is a angrycat with possibility %.6f' %prediction[:, 2])
#            elif max_index==3:
#                print('This is a curiouscat with possibility %.6f' %prediction[:, 3])
#            elif max_index==4:
#                print('This is a relaxedcat with possibility %.6f' %prediction[:, 4])
#

            
#%%Evaluate test images,刚开始写的腊鸡版本，写的很烂，没用session，循环打开、关闭session，耗时很长。但是不舍得删掉hhhh
#
#
#from PIL import Image
##
#def evaluate_test_images():
#    '''Test images against the saved models and parameters
#    '''
#    #Parameters
#    i=0
#    #下方两条信息只print一次,同时记录测试集图片数量
#    j=0    
#    test_dir = '/python/cat_emotion/data/test/'
#    for file in os.listdir(test_dir):
#        name = file.split(sep='.')
#        label_map = {'negativecat':0, 'happycat':1, 'angrycat':2, 'curiouscat':3, 'relaxedcat':4}
#        label = label_map[name[0]]
#        image = Image.open(test_dir + file)
#        image_array = np.array(image)
#
##       检测有没有灰度图这种毒瘤.....        
##        if image_array.shape!=(208,208,3):
##            print(file)
##            print(image_array.shape)
#        
#        with tf.Graph().as_default():
#            BATCH_SIZE = 1
#            N_CLASSES = 5
#            
#            image = tf.cast(image_array, tf.float32)
#            image = tf.image.per_image_standardization(image)
#            image = tf.reshape(image, [1, 208, 208, 3])
#            logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#            logit = tf.nn.softmax(logit)
#            x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#                    
#            logs_train_dir = '/python/cat_emotion/logs3/train/' 
#            saver = tf.train.Saver()
#            
#            with tf.Session() as sess:
#                if j==0:
#                    print("Reading checkpoints...")
#                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#                if ckpt and ckpt.model_checkpoint_path:
#                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                    saver.restore(sess, ckpt.model_checkpoint_path)
#                    if j==0:
#                        print('Loading success, global_step is %s' % global_step)
#                else: 
#                    if j==0:
#                        print('No checkpoint file found')
#                j=j+1             
#                prediction = sess.run(logit, feed_dict={x: image_array})
#                max_index = np.argmax(prediction)
#                if max_index==label:
#                    i=i+1
#                
#    test_accuracy = i/j*100
#    
#    print('There are totally %d testing pictures, and %d of them are correctly predicted' %(j,i))
#    print('Evaluation result: The test accuracy is %.2f%%' %test_accuracy)
    
