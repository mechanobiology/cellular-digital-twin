from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import numpy as np
import math
from ops import *
import time
from data_reader import data_reader
from progressbar import ETA, Bar, Percentage, ProgressBar
from skimage import io
from data_reader import data_reader

class GAN(object):

    def __init__(self, sess, flag):
        self.conf = flag
        self.sess = sess
        self.chan_out_r = 1
        self.chan_out_r_s =2
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        if not os.path.exists(self.conf.sampledir):
            os.makedirs(self.conf.sampledir)
        print("Start building network=================")
        self.configure_networks()
        print("Finishing building network=================")
    
    def configure_networks(self):
        self.global_step  = tf.Variable(0, trainable=False)

        self.build_network()
        variables = tf.trainable_variables()

        self.var_gen = [var for var in variables if var.name.startswith('Generator')]
        self.var_disc = [var for var in variables if var.name.startswith('Discriminator')]
        
        self.train_disc = tf.contrib.layers.optimize_loss(self.dis_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate_dis, optimizer='Adam', variables=self.var_disc, update_ops=[])
        self.train_gen = tf.contrib.layers.optimize_loss(self.gen_loss, global_step = self.global_step, 
            learning_rate=self.conf.learning_rate_gen, optimizer='Adam', variables=self.var_gen, update_ops=[])
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary()


    def build_network(self):

        self.sampled_z_s = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
        self.input_y = tf.placeholder(tf.int32,[None,self.conf.n_class])    
        self.input_x = tf.placeholder(tf.float32,[None, self.conf.slices, self.conf.height, self.conf.width, 3]) #mod
        self.input_x_r = tf.placeholder(tf.float32,[None, self.conf.slices, self.conf.height, self.conf.width, 1]) #mod
        self.input_y= tf.cast(self.input_y, tf.float32)
        print("Start building the generator of the ConGAN========================")
        #build the conditional auto encoder
        with tf.variable_scope('Generator') as scope:
            self.output_r, self.down_outputs = encode_img(self.input_x_r, self.conf.hidden_size)
            print(self.output_r.get_shape())
            self.X_rec_s = generator(self.down_outputs, self.sampled_z_s, self.output_r, self.input_y, self.conf.batch_size) # only s channel
        print("=========================Now split and insert")
        self.ch1, self.ch2_, self.ch3 = tf.split(self.input_x, num_or_size_splits=3, axis= 4) #mod
        print(self.ch1.get_shape())
        print(self.X_rec_s.get_shape())
        self.X_rec = tf.concat([self.ch1, self.X_rec_s], axis= 4) #mod
        if int(self.conf.mode) == 1:
            self.X_real = tf.concat([self.ch1, self.ch2_], axis= 4) #mod
        elif int(self.conf.mode) == 2:
            self.X_real = tf.concat([self.ch1, self.ch3], axis=4)  # mod
        print('X_rec shape: {}'.format(self.X_rec.get_shape()))
        print('X_real shape: {}'.format(self.X_real.get_shape()))

        with tf.variable_scope('Discriminator') as scope:
            self.out_real = discriminator(self.X_real, self.input_y, self.conf.batch_size)
            scope.reuse_variables()
            self.out_fake = discriminator(self.X_rec,  self.input_y, self.conf.batch_size)


        # the loss for the conditional auto encoder
        self.d_loss_real = self.get_bce_loss(self.out_real, tf.ones_like(self.out_real))
        self.d_loss_fake = self.get_bce_loss(self.out_fake, tf.zeros_like(self.out_fake))
        self.g_loss = self.get_bce_loss(self.out_fake, tf.ones_like(self.out_fake))
        self.rec_loss = self.get_mse_loss(self.X_rec_s, self.ch2_)
        self.X_rec_s_flat = tf.reshape(self.X_rec_s, [self.conf.slices*self.conf.height*self.conf.width])
        self.ch2_flat = tf.reshape(self.ch2_, [self.conf.slices*self.conf.height*self.conf.width])
        self.ch3_flat = tf.reshape(self.ch3, [self.conf.slices * self.conf.height * self.conf.width])
        if int(self.conf.mode) == 1:
            self.pearson_loss = self.correlationMetric(self.X_rec_s_flat, self.ch2_flat)
        elif int(self.conf.mode) == 2:
            self.pearson_loss = self.correlationMetric(self.X_rec_s_flat, self.ch3_flat)

        # build the model for the final conditional generation
        
        self.dis_loss= self.d_loss_fake+self.d_loss_real
        self.gen_loss= self.pearson_loss + self.g_loss*self.conf.gamma_gen
        self.test_x_r = tf.placeholder(tf.float32,[None, self.conf.slices, self.conf.height, self.conf.width, 1]) #mod
        self.test_y = tf.placeholder(tf.int32,[None,self.conf.n_class])
        self.random_s_test= tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        fix_s_test = tf.zeros_like(self.random_s_test)
        self.test_y = tf.cast(self.test_y, tf.float32)
        with tf.variable_scope('Generator', reuse= True) as scope:
            inter_r, test_downs = encode_img(self.test_x_r, self.conf.hidden_size)
            self.test_out = generator(test_downs, self.random_s_test, inter_r, self.test_y, self.conf.batch_size)
        with tf.variable_scope('Generator', reuse= True) as scope:
            self.test_out2 = generator(test_downs, fix_s_test, inter_r, self.test_y, self.conf.batch_size)
        
        print("==================FINAL shape is ")
        print(self.test_out.get_shape())

        
       

    def config_summary(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/Rec_loss', self.rec_loss))
        summarys.append(tf.summary.scalar('/Global_step', self.global_step))
        summarys.append(tf.summary.scalar('/d_loss_real', self.d_loss_real))
        summarys.append(tf.summary.scalar('/d_loss_fake', self.d_loss_fake))
        summarys.append(tf.summary.scalar('/d_loss', self.dis_loss))
        summarys.append(tf.summary.scalar('/g_loss', self.g_loss)) 
        summarys.append(tf.summary.scalar('/generator_loss', self.gen_loss))
        summarys.append(tf.summary.image('input_X', self.input_x, max_outputs = 10))
        summarys.append(tf.summary.image('input_s', self.ch2_, max_outputs = 10))
    #    summarys.append(tf.summary.image('input_r', self.input_x, max_outputs = 10))
        summarys.append(tf.summary.image('rec_r', self.X_rec_s, max_outputs = 10))
        summarys.append(tf.summary.image('recon_X', self.X_rec, max_outputs = 10))        
        summary = tf.summary.merge(summarys)
        return summary

    
    def get_coefficient(self, iter_number):
        boundaries= [50000,150000]
        values = [0.0, 0.5, 1.0]
        rate = tf.train.piecewise_constant(iter_number, boundaries, values)
        return rate
        

    def correlationMetric(self, x, y):
        """Metric returning the Pearson correlation coefficient of two tensors over some axis."""
        n = tf.cast(tf.shape(x), x.dtype)
        xsum = tf.reduce_sum(x)
        ysum = tf.reduce_sum(y)
        xmean = xsum / n
        ymean = ysum / n
        xvar = tf.reduce_sum( tf.squared_difference(x, xmean))
        yvar = tf.reduce_sum( tf.squared_difference(y, ymean))
        cov = tf.reduce_sum( (x - xmean) * (y - ymean))
        corr = cov / tf.sqrt(xvar * yvar)
        return tf.constant(1.0, dtype=x.dtype) - corr

    def correlationLoss(self, x, y):
        """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
        while trying to have the same mean and variance"""
        #x = tf.convert_to_tensor(x)
        #y = math_ops.cast(y, x.dtype)
        n = tf.cast(tf.shape(x), x.dtype)
        xsum = tf.reduce_sum(x)
        ysum = tf.reduce_sum(y)
        xmean = xsum / n
        ymean = ysum / n
        xsqsum = tf.reduce_sum( tf.squared_difference(x, xmean))
        ysqsum = tf.reduce_sum( tf.squared_difference(y, ymean))
        cov = tf.reduce_sum( (x - xmean) * (y - ymean))
        corr = cov / tf.sqrt(xsqsum * ysqsum)
        # absdif = tmean(tf.abs(x - y), axis=axis) / tf.sqrt(yvar)
        sqdif = tf.reduce_sum(tf.squared_difference(x, y)) / n / tf.sqrt(ysqsum / n)
        # meandif = tf.abs(xmean - ymean) / tf.abs(ymean)
        # vardif = tf.abs(xvar - yvar) / yvar
        # return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (meandif * 0.01) + (vardif * 0.01)) , dtype=tf.float32 )
        return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (0.01 * sqdif)) , dtype=tf.float32 )


    def get_bce_loss(self, output_tensor, target_tensor, epsilon=1e-10):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= output_tensor, labels = target_tensor))
   #     return tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -(1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

    def get_log_softmax(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= x, labels = y))

    def get_mse_loss(self, x, y):
        return tf.losses.mean_squared_error(predictions= x, labels= y)

    def get_l1_loss(self,x, y):
        return tf.losses.absolute_difference(x, y, scope='l1_loss')

    def get_l2_loss(self,t):
        return tf.nn.l2_loss(t)

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def save_summary(self, summary, step):
         print('---->summarizing', step)
         self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.checkpoint >0:
            print('=======Now load the model===============')
            self.reload(self.conf.checkpoint)
        data = data_reader(self.conf)
        iterations = 1
        max_epoch = int (self.conf.max_epoch - self.conf.checkpoint)

        for epoch in range(max_epoch):
            pbar = ProgressBar()
            for i in pbar(range(self.conf.updates_per_epoch)):
                inputs, labels, _ = data.next_batch(self.conf.batch_size)
                inputs_r = data.extract(inputs)
                sampled_zs = np.random.normal(size= (self.conf.batch_size,self.conf.hidden_size))
                feed_dict = {self.sampled_z_s: sampled_zs, self.input_y: labels, self.input_x_r:inputs_r, self.input_x: inputs}
                _ , d_loss = self.sess.run([self.train_disc,self.dis_loss], feed_dict= feed_dict)
                _ , g_loss = self.sess.run([self.train_gen, self.gen_loss], feed_dict = feed_dict) #mod
                if iterations %self.conf.save_step == 0:
                    self.save(iterations+self.conf.checkpoint)
                iterations = iterations +1
            print("g_loss is ===================", g_loss, "d_loss is =================", d_loss)
            test_x, test_y, _ = data.next_test_batch(self.conf.batch_size)
            test_x_r = data.extract(test_x)
            test_out, test_out_2 = self.sess.run([self.test_out, self.test_out2], feed_dict= {self.test_x_r: test_x_r,  self.test_y: test_y})
            self.save_image(test_out, test_out_2, test_x, epoch+int(self.conf.checkpoint))
            self.evaluate(data, epoch+int(self.conf.checkpoint))

    def save_image(self, imgs, imgs2, inputs, epoch):
        imgs_test_folder = os.path.join(self.conf.working_directory, 'validation')
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for k in range(self.conf.batch_size):
            temp_test_dir= os.path.join(imgs_test_folder, 'epoch_%d_#img_%d.tiff'%(epoch,k)) #mod
            res = np.zeros((self.conf.slices, self.conf.height, self.conf.width*6+10, 3), dtype=np.int8) #mod
            res[:,:,0:self.conf.width,:]= inputs[k,:,:,:,:] * 255 #mod
            res[:,:,self.conf.width+2:self.conf.width*2+2,0]=inputs[k,:,:,:,0] * 255 #mod
            res[:,:,self.conf.width+2:self.conf.width*2+2,int(self.conf.mode)]=inputs[k,:,:,:,int(self.conf.mode)] * 255 #mod
            res[:,:,self.conf.width*2+4:self.conf.width*3+4, int(self.conf.mode)]= inputs[k,:,:,:,int(self.conf.mode)] * 255 #mod
            res[:,:,self.conf.width*3+6:self.conf.width*4+6, int(self.conf.mode)]= imgs[k,:,:,:,0] * 255 #mod
            res[:,:,self.conf.width*4+8:self.conf.width*5+8, 0]= inputs[k,:,:,:,0] * 255 #mod
            #res[:,:,self.conf.width*4+8:self.conf.width*5+8, 2]= inputs[k,:,:,:,2] * 255 #mod
            res[:,:,self.conf.width*4+8:self.conf.width*5+8, int(self.conf.mode)]= imgs[k,:,:,:,0] * 255 #mod
            res[:,:,self.conf.width*5+10:self.conf.width*6+10, int(self.conf.mode)]= imgs2[k,:,:,:,0] * 255 #mod
            io.imsave(temp_test_dir, res)
        print("Evaluation images generated!==============================") 


    

    def generate_con_image(self):
        
        for i in range(self.conf.n_class):
            sampled_y = np.zeros((self.conf.batch_size, self.conf.n_class), dtype=np.float32)
            sampled_y[:,i]=1
            imgs = self.sess.run(self.generate_con_out, {self.generated_y: sampled_y})
            for k in range(imgs.shape[0]):
                imgs_folder = os.path.join(self.conf.working_directory, 'imgs_con_parallel', str(i))
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)   
                imsave(os.path.join(imgs_folder,'%d.png') % k,
                    imgs[k,:,:,:])
        print("conditional generated imgs saved!!!!==========================")               
    
    def evaluate(self, data, epoch):        
        print("Now start Testing set evaluation ==============================")
        pbar = ProgressBar()
        imgs_original_folder = os.path.join(self.conf.working_directory, 'evaluation_test_epoch%d'%(epoch))
        if not os.path.exists(imgs_original_folder):
            os.makedirs(imgs_original_folder)
        imgs_test_folder = os.path.join(self.conf.working_directory, 'evaluation_test_epoch%d'%(epoch))
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for i in pbar(range(self.conf.max_test_epoch)):
            x, y, r = data.next_test_batch(self.conf.batch_size)
            x_extracted = data.extract(x)
            y_label  = np.argmax(y, axis= 1)
            for j in range (self.conf.max_generated_imgs):
                output_test = self.sess.run(self.test_out, feed_dict={self.test_x_r: x_extracted,  self.test_y: y})
                print(output_test.shape)
                for k in range(output_test.shape[0]):
                    class_dir = os.path.join(imgs_test_folder, 'cls%d' % (y_label[k]))
                    temp_test_dir = os.path.join(class_dir, 'epoch_%d_#img_%d_cls_%d' % (i, k, y_label[k]))
                    if not os.path.exists(temp_test_dir):
                        os.makedirs(temp_test_dir)
                    res_x = np.zeros((self.conf.slices, self.conf.height, self.conf.width, 3), dtype=np.int8)  # mod
                    res_x[:,:,:,:] = x[k,:,:,:,:]*255
                    io.imsave(os.path.join(temp_test_dir, 'ground_truth.tiff'),
                           res_x) #mod
                    res = np.zeros((self.conf.slices, self.conf.height, self.conf.width, 3), dtype=np.int8) #mod
                    res[:,:,:,int(self.conf.mode)] = output_test[k,:,:,:,0] * 255 #mod
                    print(res.shape)
                    io.imsave(os.path.join(temp_test_dir, 'prediction.tiff'),
                           res) #mod
                    res2 = np.zeros((self.conf.slices, self.conf.height, self.conf.width*2+2, 3), dtype=np.int8)
                    res2[:, :, 0:self.conf.width, int(self.conf.mode)] = x[k, :, :, :, int(self.conf.mode)] * 255
                    res2[:, :, self.conf.width+2:self.conf.width*2+2, int(self.conf.mode)] = output_test[k,:,:,:,0] * 255
                    io.imsave(os.path.join(temp_test_dir, 'target_error_panel.tiff'),
                           res2)  # mod
        print("Evaluation images generated!==============================")


    def independent_test(self):
        data = data_reader(self.conf)
        self.reload(self.conf.checkpoint)
        print("Now start Testing set evaluation ==============================")
        pbar = ProgressBar()
        imgs_original_folder = os.path.join(self.conf.working_directory, 'independent_test')
        if not os.path.exists(imgs_original_folder):
            os.makedirs(imgs_original_folder)
        imgs_test_folder = os.path.join(self.conf.working_directory, 'independent_test')
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for i in pbar(range(self.conf.max_test_epoch)):
            x, y, r = data.next_test_batch(self.conf.batch_size)
            x_extracted = data.extract(x)
            y_label  = np.argmax(y, axis= 1)
            for j in range (self.conf.max_generated_imgs):
                output_test = self.sess.run(self.test_out, feed_dict={self.test_x_r: x_extracted,  self.test_y: y})
                print(output_test.shape)
                for k in range(output_test.shape[0]):
                    class_dir = os.path.join(imgs_test_folder, 'cls%d' % (y_label[k]))
                    temp_test_dir = os.path.join(class_dir, 'epoch_%d_#img_%d_cls_%d' % (i, k, y_label[k]))
                    if not os.path.exists(temp_test_dir):
                        os.makedirs(temp_test_dir)
                    res_x = np.zeros((self.conf.slices, self.conf.height, self.conf.width, 3), dtype=np.int8)  # mod
                    res_x[:,:,:,:] = x[k,:,:,:,:]*255
                    io.imsave(os.path.join(temp_test_dir, 'ground_truth.tiff'),
                           res_x) #mod
                    res = np.zeros((self.conf.slices, self.conf.height, self.conf.width, 3), dtype=np.int8) #mod
                    res[:,:,:,int(self.conf.mode)] = output_test[k,:,:,:,0] * 255 #mod
                    print(res.shape)
                    io.imsave(os.path.join(temp_test_dir, 'prediction.tiff'),
                           res) #mod
                    res2 = np.zeros((self.conf.slices, self.conf.height, self.conf.width*2+2, 3), dtype=np.int8)
                    res2[:, :, 0:self.conf.width, int(self.conf.mode)] = x[k, :, :, :, int(self.conf.mode)] * 255
                    res2[:, :, self.conf.width+2:self.conf.width*2+2, int(self.conf.mode)] = output_test[k,:,:,:,0] * 255
                    io.imsave(os.path.join(temp_test_dir, 'target_error_panel.tiff'),
                           res2)  # mod
        print("Evaluation images generated!==============================")


    def reload(self, epoch):
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        model_path = checkpoint_path + '-' + str(epoch*self.conf.updates_per_epoch)
        if not os.path.exists(model_path + '.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
        print("model load successfully===================")

