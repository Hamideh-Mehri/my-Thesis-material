import tensorflow as tf
import functools
import numpy as np
import copy

def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse



def score(model, tensordata, weights):
   datasource = model.datasource
   inp, _ = next(iter(tensordata.batch(1)))
   input_one = tf.ones(inp.shape)
   weights_copy = copy.deepcopy(weights)
   #prn_keys = weights_copy.keys()
   all_keys = weights_copy.keys()
   prn_keys = []
   for key in weights_copy.keys():
      if tf.math.count_nonzero(weights_copy[key]).numpy() != 0:
         prn_keys.append(key)
   #weights_abs = {k:tf.abs(weights_copy[k]) for k in all_keys}
   with tf.GradientTape() as tape:
      #tape.watch(weights_abs)
      tape.watch(weights_copy)
      weights_abs = {k:tf.abs(weights_copy[k]) for k in all_keys}
      logits = model.forward_pass(weights_abs, input_one)
      loss = tf.reduce_sum(logits)
   grads = tape.gradient(loss, [weights_copy[k] for k in prn_keys])    #or weights_abs???
   gradients = dict(zip(prn_keys, grads))
   scores = {}
   for k in prn_keys:
      scores[k] = tf.abs(weights[k] * gradients[k])
   return scores



def SynFlow(model, tensordata, weights, prune_epoch = 100):
   final_weights = copy.deepcopy(weights)
   epochs = prune_epoch
   for epoch in range(epochs):
       S = score(model, tensordata, final_weights)
       S_shape = {k:S[k].shape.as_list() for k in S.keys()}
       split_sizes = []
       for k in S.keys():
          split_sizes.append(functools.reduce(lambda x,y: x*y, S_shape[k]))
       S_v = tf.concat([tf.reshape(S[k], [-1]) for k in S.keys()], axis=0)
       num_params = S_v.shape.as_list()[0]
       n =epochs
       k = epoch + 1
       power = -k/n
       compression_ratio = 10
       sparsity = compression_ratio ** power
       kappa = int(round(num_params * (sparsity)))
       topk, ind = tf.nn.top_k(S_v, k=kappa, sorted=True)
       sp_mask = tf.SparseTensor(dense_shape=S_v.shape.as_list(), values=tf.ones_like(ind, dtype=tf.float32).numpy(), indices=np.expand_dims(ind.numpy(), 1))
       mask_v = tf.sparse.to_dense(sp_mask, validate_indices=False)
       v_splits = tf.split(mask_v, num_or_size_splits=split_sizes)
       mask_restore = {}
       for i, key in enumerate(S.keys()):
          mask_restore.update({key: tf.reshape(v_splits[i], S_shape[key])})
       final_mask = mask_restore
       final_weights = apply_mask(final_weights, final_mask)
   #set the weights of mask the same as weights
   all_keys = final_weights.keys()
   mask_keys = final_mask.keys()
   remain_keys = list(set(all_keys) - set(mask_keys))
   final_mask.update({k: tf.ones(shape=final_weights[k].shape) for k in remain_keys})
   return final_mask, final_weights

