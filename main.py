from utils import get_config_from_json
from compute import compute_test_accuracy, visual_sparsity_layer
import tensorflow as tf
import keras
import os
import pandas as pd
import numpy as np
from compute_GAN import compute






def main():
   with tf.device('/gpu:0'):
      datasource = 'fashion-mnist'    #datasource which is used for pruning
      transfer_datasource ='mnist'
      arch = 'dcgan_mnist'
      pruning_method = 'SNIP_GAN2'
      train_iteration = 250
      batchsize = 256
      savedmodelname = 'pruned_dcgan_fashion_mnist_trained_on_mnist'
      target_sparsity = 0.9
      compute(datasource, arch, pruning_method, batchsize, train_iteration, savedmodelname, target_sparsity, transfer_datasource)

      #config_resnet, _ = get_config_from_json('./config/ResNet/config1.json')
      # config_vgg, _ = get_config_from_json('./config/VGG/config1.json')

      #original_accuracy = compute_test_accuracy(config_resnet, 'without_pruning', 'config5')
      # mask_accuracy_grasp_v1 = compute_test_accuracy(config_vgg, 'grasp_v1', 'grasp_v1')
      # mask_accuracy_grasp_v2 = compute_test_accuracy(config_vgg, 'grasp_v2', 'grasp_v2')
      # mask_accuracy_grasp_v3 = compute_test_accuracy(config_vgg, 'grasp_v3', 'grasp_v3')
      # mask_accuracy_grasp_v1_iter = compute_test_accuracy(config_vgg, 'grasp_v1_iter', 'grasp_v1_iter')
      # mask_accuracy_grasp_v2_iter = compute_test_accuracy(config_vgg, 'grasp_v2_iter', 'grasp_v2_iter')
      # mask_accuracy_grasp_v3_iter = compute_test_accuracy(config_vgg, 'grasp_v3_iter', 'grasp_v3_iter')
      # mask_accuracy_synflow = compute_test_accuracy(config_vgg, 'synflow', 'synflow')
      # mask_accuracy_snip = compute_test_accuracy(config_vgg, 'snip', 'snip')
      # random_accuracy =  compute_test_accuracy(config_vgg, 'random', 'random')
      # original_accuracy = compute_test_accuracy(config_vgg, 'without_pruning', 'original')
      # mask_accuracy_grasp_v1_iter = compute_test_accuracy(config_vgg, 'grasp_v1_iter', 'grasp_v1_iter')

      # mask_accuracy_grasp_v2_iter = compute_test_accuracy(config_vgg, 'grasp_v2_iter', 'grasp_v2_iter')
      # mask_accuracy_grasp_v3_iter = compute_test_accuracy(config_vgg, 'grasp_v3_iter', 'grasp_v3_iter')
      # mask_accuracy_grasp_v1 = compute_test_accuracy(config_vgg, 'grasp_v1', 'grasp_v1')
      # mask_accuracy_grasp_v2 = compute_test_accuracy(config_vgg, 'grasp_v2', 'grasp_v2')
      # mask_accuracy_grasp_v3 = compute_test_accuracy(config_vgg, 'grasp_v3', 'grasp_v3')


      # random_accuracy =  compute_test_accuracy(config_vgg, 'random')
      # mask_accuracy_snip2 = compute_test_accuracy(config_vgg, 'snip2')
      #mask_accuracy_grasp2 = compute_test_accuracy(config_vgg, 'grasp2', 'grasp2')
      #mask_accuracy_grasp_new = compute_test_accuracy(config_vgg, 'grasp_new')
      
      # mask_accuracy_snip = compute_test_accuracy(config_vgg, 'snip')
     
      # print("mask_accuracy_synflow:{} mask_accuracy_snip:{} mask_accuracy_grasp:{} original_accuracy:{} accuracy_snip2:{} accuracy_grasp2: {}, random_accuracy: {}"
      #  .format(mask_accuracy_synflow, mask_accuracy_snip, mask_accuracy_grasp, original_accuracy, mask_accuracy_snip2, mask_accuracy_grasp2, random_accuracy))

      #config, _ = get_config_from_json('./config/ResNet/config1.json')
      #config_vgg, _ = get_config_from_json('./config/VGG/config5.json')

      #original_accuracy = compute_test_accuracy_sweep(config, 'without_pruning', 'original')
      #original_accuracy = compute_test_accuracy(config, 'without_pruning', 'original')
      #random_accuracy =  compute_test_accuracy(config, 'random', 'random')
      #mask_accuracy_grasp = compute_test_accuracy(config_vgg, 'grasp_new', 'grasp_new')
      # mask_accuracy_synflow = compute_test_accuracy(config, 'synflow', 'synflow')
      # mask_accuracy_snip = compute_test_accuracy(config, 'snip', 'snip')
        
      # print("grasp_v1_iter:{} grasp_v2_iter:{} grasp_v3_iter:{} grasp_v1:{} grasp_v2:{} grasp_v3:{}"
      #        .format(mask_accuracy_grasp_v1_iter, mask_accuracy_grasp_v2_iter, mask_accuracy_grasp_v3_iter, mask_accuracy_grasp_v1, mask_accuracy_grasp_v2, 
      #        mask_accuracy_grasp_v3))

      #mask_accuracy_snip = compute_test_accuracy(config, 'snip')
#with tf.device('/gpu:1'):
   #config_vgg, _ = get_config_from_json('./config/VGG/config5.json')
   # visual_sparsity_layer(config_vgg, 'random')
   # visual_sparsity_layer(config_vgg, 'snip2')
   # visual_sparsity_layer(config_vgg, 'grasp2')
   # visual_sparsity_layer(config_vgg, 'snip')
   # visual_sparsity_layer(config_vgg, 'grasp')
   #visual_sparsity_layer(config_vgg, 'synflow')

   #visual_sparsity_layer(config, 'snip')
   
   #print(mask_accuracy_grasp2)
   #print("mask_accuracy_snip:{} mask_accuracy_snip2:{}".format(mask_accuracy_snip, mask_accuracy_snip2))
       
       

if __name__ == "__main__":
    main()
