from Data import generate_tensordata, Dataset
from train import Train
from Arch import load_network
from RANDOM import RANDOM
from SNIP import SNIP
from GRASP_v3 import GRASP_v3
from GRASP_v2 import GRASP_v2
from GRASP_v1 import GRASP_v1
from GRASP_v3_iter import GRASP_v3_iter
from GRASP_v2_iter import GRASP_v2_iter
from GRASP_v1_iter import GRASP_v1_iter
from SNIP2 import SNIP2
from SynFlow import SynFlow
import tensorflow as tf
import re
import matplotlib.pyplot as plt
import functools
import numpy as np
import copy
import pandas as pd
import wandb

def compute_test_accuracy(configs, pruning_method, run_id):
    
    initializerdic = {'varianceScaling': tf.initializers.variance_scaling(), 'HeNormal': tf.keras.initializers.HeNormal(),
                       'zero': tf.keras.initializers.Zeros() , 'randomnormal':  tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                       'randomuniform': tf.keras.initializers.RandomUniform(minval=0., maxval=1.), 'glorotnormal': tf.keras.initializers.GlorotNormal()}
    batch_size = configs.batch_size
    batch_size_for_prune = configs.batch_size_for_prune
    datasource = configs.datasource
    optimizer = configs.optimizer
    lr_decay_type = configs.lr_decay_type
    learning_rate = configs.learning_rate
    boundaries = configs.boundaries
    values = configs.values
    train_iterations = configs.train_iterations
    arch = configs.arch
    initializer1= initializerdic[configs.initializer_conv]
    initializer2= initializerdic[configs.initializer_dense]
 
    batchtype = configs.batchtype
    target_sparsity = configs.target_sparsity
    prune_epoch = configs.pruning_epoch
    experiment = configs.experiment


    tensordata_train = generate_tensordata(datasource, 'train')
    tensordata_val = generate_tensordata(datasource, 'val')
    tensordata_test = generate_tensordata(datasource, 'test')

    dat = Dataset(datasource)
    num_classes = dat.num_classes
    
    model = load_network(datasource, arch, num_classes, initializer1, initializer2)
    weights = model.construct_weights(True)

    if pruning_method == 'snip':
       final_mask, final_weight = SNIP(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)
    elif pruning_method == 'synflow':
       final_mask, final_weight = SynFlow(model, tensordata_train, weights, prune_epoch)
       net = model.make_layers(final_weight, final_mask)
    elif pruning_method == 'without_pruning':
       final_weight = model.construct_weights(True)
       weightkeys = []
       for key in final_weight.keys():
           weightkeys.append(key)
       final_mask = {k:tf.Variable(tf.ones(final_weight[k].shape), trainable=False, dtype=tf.float32) for k in weightkeys}
       net = model.make_layers(final_weight, final_mask)
    elif pruning_method == 'snip2':
       final_mask, final_weight = SNIP2(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)
    elif pruning_method == 'grasp_v2':
       final_mask, final_weight = GRASP_v2(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)
    elif pruning_method == 'random':
       final_mask, final_weight = RANDOM(model, target_sparsity)
       net = model.make_layers(final_weight, final_mask)
    elif pruning_method == 'grasp_v1':
       final_mask, final_weight = GRASP_v1(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)

    elif pruning_method == 'grasp_v3':
       final_mask, final_weight = GRASP_v3(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)

    elif pruning_method == 'grasp_v1_iter':
       final_mask, final_weight = GRASP_v1_iter(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)

    elif pruning_method == 'grasp_v2_iter':
       final_mask, final_weight = GRASP_v2_iter(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)

    elif pruning_method == 'grasp_v3_iter':
       final_mask, final_weight = GRASP_v3_iter(model, target_sparsity, batch_size_for_prune, batchtype)
       net = model.make_layers(final_weight, final_mask)

    trainobject = Train(optimizer, lr_decay_type, learning_rate, boundaries, values, train_iterations, batch_size)

    loss_fn = trainobject.loss_function()
    #reg = trainobject.regularization(net)
    lr = trainobject.prepare_learningrate()
    optim = trainobject.get_optimizer(lr)

    train_sample = dat.num_example['train']

   #checkpoint
    ckpt = tf.train.Checkpoint(step = tf.Variable(1), net = net)
    checkpoint_dir = './tf_ckpts' +'/' + configs.experiment + '/'+ pruning_method
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    
    savedmodelname = configs.experiment + '*' + pruning_method
    projectname = configs.experiment 
    
    wandb.init(project=projectname, name=run_id)

    accuracy = trainobject.train(net, tensordata_train, tensordata_val,tensordata_test, loss_fn, optim,  train_sample,ckpt, manager, savedmodelname, wandb)

    wandb.finish()

    #write to dataframe
    unpickeled_df = pd.read_pickle('experimentResults.pkl')
    row = [experiment, arch, datasource, pruning_method, target_sparsity, round(accuracy.numpy(),3), train_iterations]
    unpickeled_df.loc[len(unpickeled_df)] = row
    unpickeled_df.to_pickle('experimentResults.pkl')

    return accuracy
    manager.latest_checkpoint = None


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

def get_labels(sparsitydic):
  keylist = list(sparsitydic.keys())
  labels =[]
  for key in keylist:
    if 'conv' in key and 'w' in key:
      num = re.findall(r'\d+', key) 
      labels.append('conv' + num[0])
    elif 'fc' in key and 'w' in key:
       num = re.findall(r'\d+', key) 
       labels.append('fc' + num[0])
    elif 'conv' in key and 'b' in key:
       num = re.findall(r'\d+', key) 
       labels.append('conv' + num[0] + '_bias')
    elif 'fc' in key and 'b' in key:
       num = re.findall(r'\d+', key) 
       labels.append('fc' + num[0] + '_bias')
  return labels

def visual_sparsity_layer(configs, pruning_method):

    target_sparsity = configs.target_sparsity
    batchtype = configs.batchtype
    batch_size = configs.batch_size
    datasource = configs.datasource
    arch = configs.arch
    dat = Dataset(datasource)
    num_classes = dat.num_classes
    prune_epoch = configs.pruning_epoch
    initializerdic = {'varianceScaling': tf.initializers.variance_scaling(), 'HeNormal': tf.keras.initializers.HeNormal()}
    initializer1= initializerdic[configs.initializer_conv]
    initializer2= initializerdic[configs.initializer_dense]


    tensordata = generate_tensordata(datasource, 'train')

    model = load_network(datasource, arch, num_classes, initializer1, initializer2)
    weights = model.construct_weights(True)

    if pruning_method == 'snip':
       final_mask, final_weight = SNIP(model, target_sparsity, batch_size, batchtype)
    elif pruning_method == 'grasp':
       final_mask, final_weight = GRASP(model, target_sparsity, batch_size, batchtype)
    elif pruning_method == 'synflow':
       final_mask, final_weight = SynFlow(model, tensordata, weights, prune_epoch)
    elif pruning_method == 'snip2':
       final_mask, final_weight = SNIP2(model, target_sparsity, batch_size, batchtype)
    elif pruning_method == 'grasp2':
       final_mask, final_weight = GRASP2(model, target_sparsity, batch_size, batchtype)
    elif pruning_method == 'random':
       final_mask, final_weight = RANDOM(model, target_sparsity)
       net = model.make_layers(final_weight)

    sparsitydic = get_sparsity_of_layers(final_weight)
    sparsity_percent_values = list(sparsitydic.values())
    keylist = list(sparsitydic.keys())
    num_layerparameters = len(keylist)
    x = np.arange(1, num_layerparameters+1)
    y = sparsity_percent_values
    sizes = [i*5 for i in y]
    labels = get_labels(sparsitydic)
    sizes_copy = copy.deepcopy(sizes)
    sizes_copy.reverse()
    plt.figure(figsize=(16, 10))
    plt.scatter(x, y, s=sizes, cmap = 'Purples', c = sizes, edgecolors='black')
    plt.xticks(x, labels, fontsize=20, rotation=90)
    plt.yticks(np.arange(10, 110, 10))
    title = arch + datasource + str(target_sparsity) + 'percent sparsity' + pruning_method + 'method'
    plt.title(title)
    plt.show()
    plt.tight_layout()
    filename = pruning_method + '_' + 'sparsity'+ '_' + configs.datasource + '_' + configs.arch
    #plt.savefig(filename)

    
    
    
    

    



