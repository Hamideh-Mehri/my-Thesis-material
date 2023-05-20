import tensorflow as tf
import time
from Data import generate_tensordata
from augment import augment
import numpy as np
import pandas as pd
import wandb

class Train(object):
    def __init__(self,optimizer, lr_decay_type, learning_rate, boundaries, values, epochs, batchsize):
     
        self.optimizer = optimizer
        self.values = values
        self.batchsize = batchsize
            
        self.lr_decay_type = lr_decay_type
        self.learning_rate = learning_rate
        self.boundaries = boundaries
        
        #self.model = model
        self.epoch = epochs
        
       

# Instantiate an optimizer
    def get_optimizer(self, lr):
        if self.optimizer =='Adam':
              optim = tf.keras.optimizers.Adam(lr)
        elif self.optimizer == 'sgd':
              optim = tf.keras.optimizers.SGD(lr)
        elif self.optimizer == 'momentum':
              optim = tf.keras.optimizers.SGD(lr, momentum=0.9)
        else: 
             raise NotImplementedError
        return optim

    def prepare_learningrate(self):
        global_step = tf.Variable(0, trainable=False)
        if self.lr_decay_type == 'constant':
              learnrate = tf.constant(self.learning_rate)
        elif self.lr_decay_type == 'piecewise':
              learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.boundaries, self.values)
              learnrate = learning_rate_fn(global_step)
        else:
            raise NotImplementedError
        return learnrate

    def loss_function(self):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return loss_fn

    #def regularization(self, net):
        #reg = 0.00025 * tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in net.trainable_weights])
        #return reg
    
    
    
    

    def train(self,net, tensordata_train, tensordata_val, tensordata_test, loss_fn, optimizer, train_sample, ckpt, manager, savedmodelname, wandb_object):
    
        start_time = time.time()
        epoch = 0

        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        train_loss_avg = tf.keras.metrics.Mean()
        val_loss_avg = tf.keras.metrics.Mean()

        plot_train_loss = []
        plot_train_acc = []
        plot_val_loss = []
        plot_val_acc = []

        

        #checkpoint
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
           print("Restored from {}".format(manager.latest_checkpoint))
        else:
           print("Initializing from scratch.")

        
        datagen = augment()
        x_train, y_train = next(iter(tensordata_train.batch(train_sample)))
        datagen.fit(x_train)

        steps_per_epoch = max(train_sample// self.batchsize, 1)
        itr = 0
        
        for i in range(self.epoch):
            for _id in range(steps_per_epoch):
                
                x, y = next(iter(datagen.flow(x_train, y_train, batch_size=self.batchsize)))
                x_batch_train, y_batch_train = tf.convert_to_tensor(x), tf.convert_to_tensor(y)

                x_batch_val, y_batch_val = next(iter(tensordata_val.batch(self.batchsize)))
  
                with tf.GradientTape() as tape:
                    logits = net(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits) 
            
                grads = tape.gradient(loss_value, net.trainable_weights)
                optimizer.apply_gradients(zip(grads, net.trainable_weights))

                logits_val = net(x_batch_val)
                loss_val = loss_fn(y_batch_val, logits_val)
                loss_train = loss_fn(y_batch_train, logits)
            
                
                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)
                val_acc_metric.update_state(y_batch_val, logits_val)
                train_loss_avg.update_state(loss_train)
                val_loss_avg.update_state(loss_val)


                if (itr + 1) % 100 == 0:
                    
                    train_acc = train_acc_metric.result()
                    val_acc = val_acc_metric.result()
                    
                    trainloss = train_loss_avg.result()
                    valloss = val_loss_avg.result()

                    plot_train_loss.append(trainloss)
                    plot_val_loss.append(valloss)

                    plot_train_acc.append(train_acc)
                    plot_val_acc.append(val_acc)

                    print("Training loss/validation loss at iteration %d: %.4f/ %.4f" % (itr+1, float(trainloss), float(valloss)))
                    print(savedmodelname.split('*')[-1], end='*')
                    print('train acc/validation acc at iteration %d: %.4f/%.4f ' % (itr+1, float(train_acc), float(val_acc)))
                    print("Time taken: %.2fs" % (time.time() - start_time))
                    print("Seen so far: %s samples" % ((itr + 1) * self.batchsize))

                    if itr > 199:
                       wandb_object.log({'iteration': itr+1 , 'trainloss': trainloss, 'validationloss': valloss, 'trainaccuracy': train_acc, 'validationaccuracy':val_acc})

                    train_acc_metric.reset_states()
                    val_acc_metric.reset_states()
                    train_loss_avg.reset_states()
                    val_loss_avg.reset_states()
                    start_time = time.time()
                    print("")

                itr +=1

        save_path = manager.save()
        
            
        test_accuracy = tf.keras.metrics.Accuracy()
        i = 0
        accuracy = 0
        for x, y in tensordata_test.batch(100):
            logits = net(x, training=False)
            prediction = tf.math.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int64)
            i += 1
            accuracy += test_accuracy(prediction, y)
        test_acc = accuracy/i
        
        print("Test set accuracy: {:.3%} ,number of epochs: {}".format(test_acc, self.epoch))
        
        
        
        #net.save(savedmodelname)
        return test_acc

    def train_for_sweep(self,net, tensordata_train, tensordata_val, tensordata_test, loss_fn, optimizer, train_sample,savedmodelname,wandb_object):
    
        start_time = time.time()
        epoch = 0

        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        train_loss_avg = tf.keras.metrics.Mean()
        val_loss_avg = tf.keras.metrics.Mean()

        plot_train_loss = []
        plot_train_acc = []
        plot_val_loss = []
        plot_val_acc = []

        
        datagen = augment()
        x_train, y_train = next(iter(tensordata_train.batch(train_sample)))
        datagen.fit(x_train)

        steps_per_epoch = max(train_sample// self.batchsize, 1)
        itr = 0
        
        for i in range(self.epoch):
            for _id in range(steps_per_epoch):
                
                x, y = next(iter(datagen.flow(x_train, y_train, batch_size=self.batchsize)))
                x_batch_train, y_batch_train = tf.convert_to_tensor(x), tf.convert_to_tensor(y)

                x_batch_val, y_batch_val = next(iter(tensordata_val.batch(self.batchsize)))
  
                with tf.GradientTape() as tape:
                    logits = net(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits) 
            
                grads = tape.gradient(loss_value, net.trainable_weights)
                optimizer.apply_gradients(zip(grads, net.trainable_weights))

                logits_val = net(x_batch_val)
                loss_val = loss_fn(y_batch_val, logits_val)
                loss_train = loss_fn(y_batch_train, logits)
            
                
                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)
                val_acc_metric.update_state(y_batch_val, logits_val)
                train_loss_avg.update_state(loss_train)
                val_loss_avg.update_state(loss_val)


                if (itr + 1) % 100 == 0:
                    
                    train_acc = train_acc_metric.result()
                    val_acc = val_acc_metric.result()
                    
                    trainloss = train_loss_avg.result()
                    valloss = val_loss_avg.result()

                    plot_train_loss.append(trainloss)
                    plot_val_loss.append(valloss)

                    plot_train_acc.append(train_acc)
                    plot_val_acc.append(val_acc)

                    print("Training loss/validation loss at iteration %d: %.4f/ %.4f" % (itr+1, float(trainloss), float(valloss)))
                    print(savedmodelname.split('_')[-1], end='*')
                    print('train acc/validation acc at iteration %d: %.4f/%.4f ' % (itr+1, float(train_acc), float(val_acc)))
                    print("Time taken: %.2fs" % (time.time() - start_time))
                    print("Seen so far: %s samples" % ((itr + 1) * self.batchsize))

                    # if itr != 99:
                    #     wandb_object.log({'iteration': itr+1 , 'trainloss': trainloss, 'validationloss': valloss, 'trainaccuracy': train_acc, 'validationaccuracy':val_acc})
                    wandb_object.log({'validationaccuracy':val_acc})
                    
                    train_acc_metric.reset_states()
                    val_acc_metric.reset_states()
                    train_loss_avg.reset_states()
                    val_loss_avg.reset_states()
                    start_time = time.time()
                    print("")

                itr +=1

            
        test_accuracy = tf.keras.metrics.Accuracy()
        i = 0
        accuracy = 0
        for x, y in tensordata_test.batch(100):
            logits = net(x, training=False)
            prediction = tf.math.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int64)
            i += 1
            accuracy += test_accuracy(prediction, y)
        test_acc = accuracy/i
        
        print("Test set accuracy: {:.3%} ,number of epochs: {}".format(test_acc, self.epoch))
        
        
        
        #net.save(savedmodelname)
        return test_acc



    def train_original(self,net, tensordata_train, tensordata_val, tensordata_test, loss_fn, optimizer, train_sample, ckpt, manager, savedmodelname, wandb_object):
    
        start_time = time.time()
        epoch = 0

        train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        train_loss_avg = tf.keras.metrics.Mean()
        val_loss_avg = tf.keras.metrics.Mean()

        plot_train_loss = []
        plot_train_acc = []
        plot_val_loss = []
        plot_val_acc = []

        

        #checkpoint
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
           print("Restored from {}".format(manager.latest_checkpoint))
        else:
           print("Initializing from scratch.")

        
        datagen = augment()
        x_train, y_train = next(iter(tensordata_train.batch(train_sample)))
        datagen.fit(x_train)

        steps_per_epoch = max(train_sample// self.batchsize, 1)
        itr = 0
        
        for i in range(self.epoch):
            for _id in range(steps_per_epoch):
        
            
                #x_batch_train, y_batch_train = next(iter(tensordata_train.batch(self.batchsize)))
                x, y = next(iter(datagen.flow(x_train, y_train, batch_size=self.batchsize)))
                x_batch_train, y_batch_train = tf.convert_to_tensor(x), tf.convert_to_tensor(y)
                
                with tf.GradientTape() as tape:
                    logits = net(x_batch_train, training=True)
                    #reg_term = 0.0005 * tf.reduce_sum([tf.reduce_sum(tf.abs(v)) for v in net.trainable_weights])
                    #loss_value = loss_fn(y_batch_train, logits) + reg_term
                    loss_value = loss_fn(y_batch_train, logits) 
            
                grads = tape.gradient(loss_value, net.trainable_weights)
                

                x_batch_val, y_batch_val = next(iter(tensordata_val.batch(self.batchsize)))
                logits_val = net(x_batch_val)
                loss_val = loss_fn(y_batch_val, logits_val)

                loss_train = loss_fn(y_batch_train, logits)
            
                optimizer.apply_gradients(zip(grads, net.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)
                val_acc_metric.update_state(y_batch_val, logits_val)
                train_loss_avg.update_state(loss_train)
                val_loss_avg.update_state(loss_val)

                

                if (itr + 1) % 100 == 0:
                    trainloss = train_loss_avg.result()
                    valloss = val_loss_avg.result()
                    plot_train_loss.append(trainloss)
                    plot_val_loss.append(valloss)
                    print("Training loss/validation loss at iteration %d: %.4f/ %.4f" % (itr+1, float(trainloss), float(valloss)))
                    train_acc = train_acc_metric.result()
                    val_acc = val_acc_metric.result()
                    plot_train_acc.append(train_acc)
                    plot_val_acc.append(val_acc)
                    print(savedmodelname.split('_')[-1], end='*')
                    print('train acc/validation acc at iteration %d: %.4f/%.4f ' % (itr+1, float(train_acc), float(val_acc)))
                    print("Time taken: %.2fs" % (time.time() - start_time))
                    print("Seen so far: %s samples" % ((itr + 1) * self.batchsize))
                    
                    wandb_object.log({'iteration': itr+1 , 'trainloss': trainloss, 'validationloss': valloss, 'trainaccuracy': train_acc, 'validationaccuracy':val_acc})

                    train_acc_metric.reset_states()
                    val_acc_metric.reset_states()
                    train_loss_avg.reset_states()
                    val_loss_avg.reset_states()
                    start_time = time.time()
                    print("")

                itr +=1

        save_path = manager.save()
        
            
        test_accuracy = tf.keras.metrics.Accuracy()
        i = 0
        accuracy = 0
        for x, y in tensordata_test.batch(100):
            logits = net(x, training=False)
            prediction = tf.math.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int64)
            i += 1
            accuracy += test_accuracy(prediction, y)
        test_acc = accuracy/i
        
        print("Test set accuracy: {:.3%} ,number of epochs: {}".format(test_acc, self.epoch))
        
        
        
        #net.save(savedmodelname)
        return test_acc


    