import tensorflow as tf
import functools
import numpy as np
import random


def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse

def RANDOM(model, target_sparsity):
    weights = model.construct_weights(True)

    prn_keys = []
    for key in weights.keys():
       if tf.math.count_nonzero(weights[key]).numpy() != 0:
           prn_keys.append(key)

    weight_shape = {k: weights[k].shape.as_list() for k in prn_keys}

    split_sizes = []
    for key in prn_keys:
        split_sizes.append(functools.reduce(lambda x, y: x*y, weight_shape[key]))
    weight_v = tf.concat([tf.reshape(weights[k], [-1]) for k in prn_keys], axis=0) 
    num_params = weight_v.shape.as_list()[0]
    kappa = int(round(num_params * (1. - target_sparsity)))   
    ind = random.sample(range(num_params), kappa)
    sp_mask = tf.SparseTensor(dense_shape=weight_v.shape.as_list(), values=tf.ones_like(ind, dtype=tf.float32).numpy(), indices=np.expand_dims(np.array(ind), 1))
    mask_v = tf.sparse.to_dense(sp_mask, validate_indices=False)
    #restore mask_v as dictionary of weights
    v_splits = tf.split(mask_v, num_or_size_splits=split_sizes)
    mask_restore = {}
    for i, key in enumerate(prn_keys):
        mask_restore.update({key: tf.reshape(v_splits[i], weight_shape[key])})
    final_mask = mask_restore
    final_w = apply_mask(weights, final_mask)
    #set the weights of mask the same as weights
    all_keys = final_w.keys()
    mask_keys = final_mask.keys()
    remain_keys = list(set(all_keys) - set(mask_keys))
    final_mask.update({k: tf.ones(shape=final_w[k].shape) for k in remain_keys})
    return final_mask, final_w