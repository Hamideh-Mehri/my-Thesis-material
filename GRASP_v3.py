import copy
import tensorflow as tf
import functools
from Data import preparebatch
import numpy as np


def compute_loss(labels, logits):
    num_classes = logits.shape.as_list()[-1]
    labels = tf.one_hot(labels, num_classes, dtype=tf.float32)
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse

def GRASP_v3(model, target_sparsity, batch_size, batchtype):
   weights = model.construct_weights(True)
   inputs1, targets1 = preparebatch(model.datasource, batch_size , batchtype)
   inputs2, targets2 = preparebatch(model.datasource, batch_size , batchtype)
   inputs = tf.concat([inputs1, inputs2], 0)
   targets = tf.concat([targets1, targets2], 0)
   N = batch_size

   inp1 = copy.deepcopy(inputs1)
   targ1 = copy.deepcopy(targets1)

   inp2 = copy.deepcopy(inputs1)
   targ2 = copy.deepcopy(targets2)

   prn_keys = []
   for key in weights.keys():
       if tf.math.count_nonzero(weights[key]).numpy() != 0:
           prn_keys.append(key)
   z = 0
   with tf.GradientTape() as tape3:
      with tf.GradientTape() as tape0, tf.GradientTape() as tape1, tf.GradientTape() as tape2:
         tape0.watch(weights)
         logits1 = model.forward_pass(weights, inp1)
         loss1 = tf.reduce_mean(compute_loss(targ1, logits1))

         tape1.watch(weights)
         logits2 = model.forward_pass(weights, inp2)
         loss2 = tf.reduce_mean(compute_loss(targ2, logits2))

         tape2.watch(weights)
         logits3 = model.forward_pass(weights, inputs)
         loss3 = tf.reduce_mean(compute_loss(targets, logits3))
      grads_w_p = tape0.gradient(loss1, [weights[k] for k in prn_keys])
      grads_w = grads_w_p
      grads_w_p = tape1.gradient(loss2, [weights[k] for k in prn_keys])
      grads_f = tape2.gradient(loss3, [weights[k] for k in prn_keys] )

      for idx in range(len(grads_w)):
          grads_w[idx] += grads_w_p[idx]

      tape3.watch(weights)
      for idx in range(len(grads_w)):
          z += tf.reduce_sum(grads_w[idx] * grads_f[idx])
   grads = tape3.gradient(z, [weights[k] for k in prn_keys])

   gradients = dict(zip(prn_keys, grads))
   scores = {}
   for k in prn_keys:
       scores[k] = (-weights[k]) * gradients[k]

   grad_key = scores.keys()
   grad_shape = {k: scores[k].shape.as_list() for k in grad_key}
   split_sizes = []
   for key in grad_key:
       split_sizes.append(functools.reduce(lambda x, y: x*y, grad_shape[key]))   
   grad_v = tf.concat([tf.reshape(scores[k], [-1]) for k in grad_key], axis=0) 
   normalgrad_v = tf.divide(grad_v, tf.abs(tf.reduce_sum(grad_v)))
   num_params = normalgrad_v.shape.as_list()[0]
   kappa = int(round(num_params * (target_sparsity)))     #num_params to remove
   topk, ind = tf.nn.top_k(normalgrad_v, k=kappa, sorted=True)
   removeIndices = ind.numpy()
   allIndices = np.arange(0,normalgrad_v.shape[0])
   remainIndices = np.nonzero(~np.isin(allIndices, removeIndices))
   num_remain = remainIndices[0].shape[0]
   sp_mask = tf.SparseTensor(dense_shape=normalgrad_v.shape.as_list(), values=np.ones(num_remain, dtype='float32'), indices=np.expand_dims(remainIndices[0], 1))
   mask_v = tf.sparse.to_dense(sp_mask, validate_indices=False)
    #restore mask_v as dictionary of weights
   v_splits = tf.split(mask_v, num_or_size_splits=split_sizes)
   mask_restore = {}
   for i, key in enumerate(grad_key):
       mask_restore.update({key: tf.reshape(v_splits[i], grad_shape[key])})
   final_mask = mask_restore
   final_w = apply_mask(weights, final_mask)
   #set the weights of mask the same as weights
   all_keys = final_w.keys()
   mask_keys = final_mask.keys()
   remain_keys = list(set(all_keys) - set(mask_keys))
   final_mask.update({k: tf.ones(shape=final_w[k].shape) for k in remain_keys})
   return final_mask, final_w