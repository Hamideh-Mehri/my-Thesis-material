import tensorflow as tf
from ArchGan import load_network
from train_GAN import train_gan
from SNIP_GAN import SNIP_GAN2, SNIP_GAN
import numpy as np
import copy
import matplotlib.pyplot as plt 
import functools



def compute(datasource, arch, pruning_method, batchsize, train_iteration, savedmodelname,target_sparsity, transfer_datasource):

    if transfer_datasource == 'mnist':

        #load and preprocess data
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images-127.5)/127.5

        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        test_images = (test_images-127.5)/127.5

        BUFFER_SIZE = 70000
        BATCH_SIZE = batchsize

        # Concatenate
        all_images = np.concatenate([train_images, test_images], axis=0)

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(all_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    elif transfer_datasource == 'fashion-mnist':
        #load and preprocess data
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images-127.5)/127.5

        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        test_images = (test_images-127.5)/127.5

        BUFFER_SIZE = 70000
        BATCH_SIZE = batchsize

        # Concatenate
        all_images = np.concatenate([train_images, test_images], axis=0)

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(all_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #load model
    latent_dim = 100
    initializer1 = tf.initializers.variance_scaling()
    initializer2 = tf.initializers.variance_scaling()
    model = load_network(datasource, arch, latent_dim,BATCH_SIZE, initializer1, initializer2)
    #load pruned weights
    batchtype = 'equal'
    
    if pruning_method == 'without_pruning':
        final_w_disc = model.construct_weights_disc(True)
        keys_disc = []
        for key in final_w_disc.keys():
            keys_disc.append(key)
        mask_disc = {k:tf.Variable(tf.ones(final_w_disc[k].shape), trainable=False, dtype=tf.float32) for k in keys_disc}

        final_w_gen = model.construct_weights_gen(True)
        keys_gen = []
        for key in final_w_gen.keys():
            keys_gen.append(key)
        mask_gen = {k:tf.Variable(tf.ones(final_w_gen[k].shape), trainable=False, dtype=tf.float32) for k in keys_gen}

        #build_generator
        generator = model.make_layers_gen(final_w_gen, mask_gen)
        #build discriminator
        discriminator = model.make_layers_disc(final_w_disc, mask_disc)

    elif pruning_method == 'SNIP_GAN':
        final_w_disc, mask_disc, final_w_gen, mask_gen = SNIP_GAN(model, target_sparsity, BATCH_SIZE, batchtype)
        #build_generator
        generator = model.make_layers_gen(final_w_gen, mask_gen)
        #build discriminator
        discriminator = model.make_layers_disc(final_w_disc, mask_disc)

    elif pruning_method == 'SNIP_GAN2':
        final_w_disc, mask_disc, final_w_gen, mask_gen = SNIP_GAN2(model, target_sparsity, BATCH_SIZE, batchtype)
        #build_generator
        generator = model.make_layers_gen(final_w_gen, mask_gen)
        #build discriminator
        discriminator = model.make_layers_disc(final_w_disc, mask_disc)

    step_per_epochs = all_images.shape[0]//BATCH_SIZE
    train_gan(discriminator, generator, batchsize, train_iteration, step_per_epochs, train_dataset,savedmodelname)



def compute_mask(datasource, arch, pruning_method, batchsize, target_sparsity):
    if datasource == 'mnist':

        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images-127.5)/127.5

        BUFFER_SIZE = 60000
        BATCH_SIZE = batchsize
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #load model
    # datasource = 'mnist'
    # arch = 'dcgan_mnist'
    latent_dim = 100
    initializer1 = tf.initializers.variance_scaling()
    initializer2 = tf.initializers.variance_scaling()
    model = load_network(datasource, arch, latent_dim,BATCH_SIZE, initializer1, initializer2)
    #load pruned weights
    target_sparsity = 0.9
    batchtype = 'equal'
    
    if pruning_method == 'without_pruning':
        final_w_disc = model.construct_weights_disc(True)
        keys_disc = []
        for key in final_w_disc.keys():
            keys_disc.append(key)
        mask_disc = {k:tf.Variable(tf.ones(final_w_disc[k].shape), trainable=False, dtype=tf.float32) for k in keys_disc}

        final_w_gen = model.construct_weights_gen(True)
        keys_gen = []
        for key in final_w_gen.keys():
            keys_gen.append(key)
        mask_gen = {k:tf.Variable(tf.ones(final_w_gen[k].shape), trainable=False, dtype=tf.float32) for k in keys_gen}


    elif pruning_method == 'SNIP_GAN':
        final_w_disc, mask_disc, final_w_gen, mask_gen = SNIP_GAN(model, target_sparsity, BATCH_SIZE, batchtype)

    elif pruning_method == 'SNIP_GAN2':
        final_w_disc, mask_disc, final_w_gen, mask_gen = SNIP_GAN2(model, target_sparsity, BATCH_SIZE, batchtype)
    
    return final_w_disc, mask_disc, final_w_gen, mask_gen


def get_sparsity_of_layers(weight):
    keylist = weight.keys()
    sparsity_layer_dic = {}
    for key in keylist:
        num_all_parameter = functools.reduce(lambda x, y: x*y, weight[key].shape)
        nonzero_parameter = tf.math.count_nonzero(weight[key]).numpy()
        zero_parameter = num_all_parameter - nonzero_parameter
        percent_of_sparsity = round((zero_parameter/num_all_parameter) * 100, 1)
        sparsity_layer_dic[key] = percent_of_sparsity
    return sparsity_layer_dic

def get_labels(sparsitydic, net):
    keylist = list(sparsitydic.keys())
    labels =[]
    for key in keylist:
       labels.append(key + '_' + net)
    return labels

def visual_sparsity_layer(datasource, arch, pruning_method, batchsize, target_sparsity):

    final_w_disc, mask_disc, final_w_gen, mask_gen = compute_mask(datasource, arch, pruning_method, batchsize, target_sparsity)

    sparsitydic_disc = get_sparsity_of_layers(final_w_disc)
    sparsitydic_gen = get_sparsity_of_layers(final_w_gen)


    sparsity_percent_values_disc = list(sparsitydic_disc.values())
    sparsity_percent_values_gen = list(sparsitydic_gen.values())
    sparsity_percent_values = sparsity_percent_values_disc + sparsity_percent_values_gen
    keylist_disc = list(sparsitydic_disc.keys())
    keylist_gen = list(sparsitydic_gen.keys())
    keylist = keylist_disc + keylist_gen
    num_layerparameters_disc = len(keylist_disc)
    num_layerparameters_gen = len(keylist_gen)
    num_layerparameters = num_layerparameters_disc + num_layerparameters_gen
    x = np.arange(1, num_layerparameters+1)
    y = sparsity_percent_values
    sizes = [i*5 for i in y]
    labels_disc  = get_labels(sparsitydic_disc, 'disc')
    labels_gen  = get_labels(sparsitydic_gen, 'gen')
    labels = labels_disc + labels_gen
    sizes_copy = copy.deepcopy(sizes)
    sizes_copy.reverse()
    plt.figure(figsize=(16, 10))
    plt.scatter(x, y, s=sizes, cmap = 'Purples', c = sizes, edgecolors='black')
    plt.xticks(x, labels, fontsize=20, rotation=90)
    plt.yticks(np.arange(10, 110, 10))
    title = arch + '*'+ str(target_sparsity)+ ' ' + 'sparsity'+ '*' + 'snipgan'+ ' ' + 'method'
    plt.title(title)
    plt.show()
    plt.savefig(title + '.png')


