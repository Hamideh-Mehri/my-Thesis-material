import tensorflow as tf
import functools
from Data import preparebatch
import numpy as np


def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse

def compute_loss(labels, logits):
    num_classes = logits.shape.as_list()[-1]
    labels = tf.one_hot(labels, num_classes, dtype=tf.float32)
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def SNIP2(model, target_sparsity, batch_size, batchtype):
    weights = model.construct_weights(False)
    prn_keys = []
    for key in weights.keys():
       if tf.math.count_nonzero(weights[key]).numpy() != 0:
           prn_keys.append(key)
    mask_init = {k:tf.Variable(tf.ones(weights[k].shape), trainable=False, dtype=tf.float32) for k in prn_keys}
    iteration = 10
    for itr in range(iteration):
        print('iteration {}'.format(itr))
        inputs, targets = preparebatch(model.datasource, batch_size, batchtype)
        #compute gradients
        with tf.GradientTape() as tape:
            tape.watch(mask_init)
            w_mask = apply_mask(weights, mask_init)
            logits = model.forward_pass(w_mask, inputs)
            loss = tf.reduce_mean(compute_loss(targets, logits))
        grads = tape.gradient(loss, [mask_init[k] for k in prn_keys])

        gradients = dict(zip(prn_keys, grads))
        gradients_abs = {k: tf.abs(v) for k, v in gradients.items()}
        grad_key = gradients_abs.keys()

        grad_shape = {k: gradients_abs[k].shape.as_list() for k in grad_key}
        #grad_shape={'w1': [11, 11, 3, 96],'w2': [5, 5, 96, 256],'w3': [3, 3, 256, 384],'w4': [3, 3, 384, 384],'w5': [3, 3, 384, 256],'w6': [256, 1024],'w7': [1024, 1024],'w8': [1024, 10],
        #'b1': [96],'b2': [256],'b3': [384],'b4': [384],'b5': [256],'b6': [1024],'b7': [1024],'b8': [10]}
        split_sizes = []
        for key in grad_key:
            split_sizes.append(functools.reduce(lambda x, y: x*y, grad_shape[key]))   
        grad_v = tf.concat([tf.reshape(gradients_abs[k], [-1]) for k in grad_key], axis=0) 
        normalgrad_v = tf.divide(grad_v, tf.reduce_sum(grad_v))
        num_params = normalgrad_v.shape.as_list()[0]
        kappa = int(round(num_params * (1. - target_sparsity)))   
        topk, ind = tf.nn.top_k(normalgrad_v, k=kappa, sorted=True)
        sp_mask = tf.SparseTensor(dense_shape=normalgrad_v.shape.as_list(), values=tf.ones_like(ind, dtype=tf.float32).numpy(), indices=np.expand_dims(ind.numpy(), 1))
        mask_v = tf.sparse.to_dense(sp_mask, validate_indices=False)
        #restore mask_v as dictionary of weights
        v_splits = tf.split(mask_v, num_or_size_splits=split_sizes)
        mask_restore = {}
        for i, key in enumerate(grad_key):
            mask_restore.update({key: tf.reshape(v_splits[i], grad_shape[key])})
        final_mask = mask_restore
        final_w = apply_mask(weights, final_mask)
        mask_init = final_mask
    #set the weights of mask the same as weights
    all_keys = final_w.keys()
    mask_keys = final_mask.keys()
    remain_keys = list(set(all_keys) - set(mask_keys))
    final_mask.update({k: tf.ones(shape=final_w[k].shape) for k in remain_keys})
    return final_mask, final_w