import tensorflow as tf
import functools
from Data import preparebatch
import numpy as np
from ArchGan import load_network

def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse

def compute_loss(logits,flag):
    if flag == 'one':
       return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(logits), logits)
    elif flag == 'zero':
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(logits), logits)

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


def SNIP_GAN(model, target_sparsity, batch_size, batchtype):
    
    weights_gen = model.construct_weights_gen(False)
    tf.random.set_seed(5)
    latent_dim = model.latentdim
    noise = tf.random.normal([batch_size, latent_dim])
    fake_images = model.forward_pass_gen(weights_gen, noise)

    weights_disc = model.construct_weights_disc(True)
    
    real_images,_ = preparebatch(model.datasource, batch_size, batchtype)

    prn_keys_disc = []
    for key in weights_disc.keys():
       prn_keys_disc.append(key)

    prn_keys_gen = []
    for key in weights_gen.keys():
       prn_keys_gen.append(key)

    mask_init_disc = {k:tf.Variable(tf.ones(weights_disc[k].shape), trainable=False, dtype=tf.float32) for k in prn_keys_disc}
    mask_init_gen = {k:tf.Variable(tf.ones(weights_gen[k].shape), trainable=False, dtype=tf.float32) for k in prn_keys_gen}

    discriminator = model.make_layers_disc(weights_disc, mask_init_disc)

    lr = 2e-4
    decay = 6e-8
    discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=lr * 0.5, decay=decay * 0.5)

    with tf.GradientTape() as tape_disc:
        tape_disc.watch(mask_init_disc)
        w_mask_disc = apply_mask(weights_disc, mask_init_disc)
        logits_real = model.forward_pass_disc(w_mask_disc, real_images)
        logits_fake = model.forward_pass_disc(w_mask_disc, fake_images)
        real_loss = tf.reduce_mean(compute_loss(logits_real, flag='one'))
        fake_loss = tf.reduce_mean(compute_loss(logits_fake, flag='zero'))
        disc_loss = real_loss + fake_loss
    grads_disc = tape_disc.gradient( disc_loss ,[mask_init_disc[k] for k in prn_keys_disc])

    #discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))
    with tf.GradientTape() as tape_gen:
        tape_gen.watch(mask_init_gen)
        w_mask_gen = apply_mask(weights_gen, mask_init_gen)
        generated_images = model.forward_pass_gen(w_mask_gen, noise)
        discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))
        logits_gen = discriminator(generated_images, training=False)
        gen_loss = tf.reduce_mean(compute_loss(logits_gen, flag='one'))
    grads_gen = tape_gen.gradient(gen_loss, [mask_init_gen[k] for k in prn_keys_gen])


    gradients_disc = dict(zip(prn_keys_disc, grads_disc))

    gradients_abs_disc = {k: tf.abs(v) for k, v in gradients_disc.items()}
    grad_key_disc = gradients_abs_disc.keys()

    grad_shape_disc = {k: gradients_abs_disc[k].shape.as_list() for k in grad_key_disc}
    #grad_shape={'w1': [11, 11, 3, 96],'w2': [5, 5, 96, 256],'w3': [3, 3, 256, 384],'w4': [3, 3, 384, 384],'w5': [3, 3, 384, 256],'w6': [256, 1024],'w7': [1024, 1024],'w8': [1024, 10],
    #'b1': [96],'b2': [256],'b3': [384],'b4': [384],'b5': [256],'b6': [1024],'b7': [1024],'b8': [10]}
    split_sizes_disc = []
    for key in grad_key_disc:
       split_sizes_disc.append(functools.reduce(lambda x, y: x*y, grad_shape_disc[key]))   
    grad_v_disc = tf.concat([tf.reshape(gradients_abs_disc[k], [-1]) for k in grad_key_disc], axis=0) 
    normalgrad_v_disc = tf.divide(grad_v_disc, tf.reduce_sum(grad_v_disc))
    num_params_disc = normalgrad_v_disc.shape.as_list()[0]
    kappa_disc = int(round(num_params_disc * (1. - target_sparsity)))   
    topk_disc, ind_disc = tf.nn.top_k(normalgrad_v_disc, k=kappa_disc, sorted=True)
    sp_mask_disc = tf.SparseTensor(dense_shape=normalgrad_v_disc.shape.as_list(), values=tf.ones_like(ind_disc, dtype=tf.float32).numpy(), indices=np.expand_dims(ind_disc.numpy(), 1))
    mask_v_disc = tf.sparse.to_dense(sp_mask_disc, validate_indices=False)
    #restore mask_v as dictionary of weights
    v_splits_disc = tf.split(mask_v_disc, num_or_size_splits=split_sizes_disc)
    mask_restore_disc = {}
    for i, key in enumerate(grad_key_disc):
       mask_restore_disc.update({key: tf.reshape(v_splits_disc[i], grad_shape_disc[key])})
    final_mask_disc = mask_restore_disc
    final_w_disc = apply_mask(weights_disc, final_mask_disc)


    gradients_gen = dict(zip(prn_keys_gen, grads_gen))

    gradients_abs_gen = {k: tf.abs(v) for k, v in gradients_gen.items()}
    grad_key_gen = gradients_abs_gen.keys()

    grad_shape_gen = {k: gradients_abs_gen[k].shape.as_list() for k in grad_key_gen}
    #grad_shape={'w1': [11, 11, 3, 96],'w2': [5, 5, 96, 256],'w3': [3, 3, 256, 384],'w4': [3, 3, 384, 384],'w5': [3, 3, 384, 256],'w6': [256, 1024],'w7': [1024, 1024],'w8': [1024, 10],
    #'b1': [96],'b2': [256],'b3': [384],'b4': [384],'b5': [256],'b6': [1024],'b7': [1024],'b8': [10]}
    split_sizes_gen = []
    for key in grad_key_gen:
       split_sizes_gen.append(functools.reduce(lambda x, y: x*y, grad_shape_gen[key]))   
    grad_v_gen = tf.concat([tf.reshape(gradients_abs_gen[k], [-1]) for k in grad_key_gen], axis=0) 
    normalgrad_v_gen = tf.divide(grad_v_gen, tf.reduce_sum(grad_v_gen))
    num_params_gen = normalgrad_v_gen.shape.as_list()[0]
    kappa_gen = int(round(num_params_gen * (1. - target_sparsity)))   
    topk_gen, ind_gen = tf.nn.top_k(normalgrad_v_gen, k=kappa_gen, sorted=True)
    sp_mask_gen = tf.SparseTensor(dense_shape=normalgrad_v_gen.shape.as_list(), values=tf.ones_like(ind_gen, dtype=tf.float32).numpy(), indices=np.expand_dims(ind_gen.numpy(), 1))
    mask_v_gen = tf.sparse.to_dense(sp_mask_gen, validate_indices=False)
    #restore mask_v as dictionary of weights
    v_splits_gen = tf.split(mask_v_gen, num_or_size_splits=split_sizes_gen)
    mask_restore_gen = {}
    for i, key in enumerate(grad_key_gen):
       mask_restore_gen.update({key: tf.reshape(v_splits_gen[i], grad_shape_gen[key])})
    final_mask_gen = mask_restore_gen
    final_w_gen = apply_mask(weights_gen, final_mask_gen)

    return final_w_disc, final_mask_disc, final_w_gen, final_mask_gen


def SNIP_GAN2(model, target_sparsity, batch_size, batchtype):
   weights_gen = model.construct_weights_gen(False)
   tf.random.set_seed(5)
   latent_dim = model.latentdim
   noise = tf.random.normal([batch_size, latent_dim])
   fake_images = model.forward_pass_gen(weights_gen, noise)

   weights_disc = model.construct_weights_disc(True)

   real_images,_ = preparebatch(model.datasource, batch_size, batchtype)

   prn_keys_disc = []
   for key in weights_disc.keys():
      prn_keys_disc.append(key)

   prn_keys_gen = []
   for key in weights_gen.keys():
      prn_keys_gen.append(key)

   mask_init_disc = {k:tf.Variable(tf.ones(weights_disc[k].shape), trainable=False, dtype=tf.float32) for k in prn_keys_disc}
   mask_init_gen = {k:tf.Variable(tf.ones(weights_gen[k].shape), trainable=False, dtype=tf.float32) for k in prn_keys_gen}

   discriminator = model.make_layers_disc(weights_disc, mask_init_disc)

   lr = 2e-4
   decay = 6e-8
   discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=lr * 0.5, decay=decay * 0.5)

   with tf.GradientTape() as tape_disc:
      tape_disc.watch(mask_init_disc)
      w_mask_disc = apply_mask(weights_disc, mask_init_disc)
      logits_real = model.forward_pass_disc(w_mask_disc, real_images)
      logits_fake = model.forward_pass_disc(w_mask_disc, fake_images)
      real_loss = tf.reduce_mean(compute_loss(logits_real, flag='one'))
      fake_loss = tf.reduce_mean(compute_loss(logits_fake, flag='zero'))
      disc_loss = real_loss + fake_loss
   grads_disc = tape_disc.gradient( disc_loss ,[mask_init_disc[k] for k in prn_keys_disc])

   #discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))
   with tf.GradientTape() as tape_gen:
      tape_gen.watch(mask_init_gen)
      w_mask_gen = apply_mask(weights_gen, mask_init_gen)
      generated_images = model.forward_pass_gen(w_mask_gen, noise)
      discriminator_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_weights))
      logits_gen = discriminator(generated_images, training=False)
      gen_loss = tf.reduce_mean(compute_loss(logits_gen, flag='one'))
   grads_gen = tape_gen.gradient(gen_loss, [mask_init_gen[k] for k in prn_keys_gen])


   gradients_disc = dict(zip(prn_keys_disc, grads_disc))

   gradients_abs_disc = {k: tf.abs(v) for k, v in gradients_disc.items()}
   grad_key_disc = gradients_abs_disc.keys()

   grad_shape_disc = {k: gradients_abs_disc[k].shape.as_list() for k in grad_key_disc}
   #grad_shape={'w1': [11, 11, 3, 96],'w2': [5, 5, 96, 256],'w3': [3, 3, 256, 384],'w4': [3, 3, 384, 384],'w5': [3, 3, 384, 256],'w6': [256, 1024],'w7': [1024, 1024],'w8': [1024, 10],
   #'b1': [96],'b2': [256],'b3': [384],'b4': [384],'b5': [256],'b6': [1024],'b7': [1024],'b8': [10]}
   split_sizes_disc = []
   for key in grad_key_disc:
      split_sizes_disc.append(functools.reduce(lambda x, y: x*y, grad_shape_disc[key]))   
   grad_v_disc = tf.concat([tf.reshape(gradients_abs_disc[k], [-1]) for k in grad_key_disc], axis=0) 
   normalgrad_v_disc = tf.divide(grad_v_disc, tf.reduce_sum(grad_v_disc))
   num_params_disc = normalgrad_v_disc.shape.as_list()[0]
   kappa_disc = int(round(num_params_disc * (1. - target_sparsity)))   
   topk_disc, ind_disc = tf.nn.top_k(normalgrad_v_disc, k=kappa_disc, sorted=True)
   sp_mask_disc = tf.SparseTensor(dense_shape=normalgrad_v_disc.shape.as_list(), values=tf.ones_like(ind_disc, dtype=tf.float32).numpy(), indices=np.expand_dims(ind_disc.numpy(), 1))
   mask_v_disc = tf.sparse.to_dense(sp_mask_disc, validate_indices=False)
   #restore mask_v as dictionary of weights
   v_splits_disc = tf.split(mask_v_disc, num_or_size_splits=split_sizes_disc)
   mask_restore_disc = {}
   for i, key in enumerate(grad_key_disc):
      mask_restore_disc.update({key: tf.reshape(v_splits_disc[i], grad_shape_disc[key])})
   final_mask_disc = mask_restore_disc
   final_w_disc = apply_mask(weights_disc, final_mask_disc)

   sparsitydic_disc = get_sparsity_of_layers(final_mask_disc)
   sparsitydic_gen = {}
   sparsitydic_gen['w1_fc'] = sparsitydic_disc['w5_fc']
   sparsitydic_gen['w2_Dconv'] = sparsitydic_disc['w4_conv']
   sparsitydic_gen['w3_Dconv'] = sparsitydic_disc['w3_conv']
   sparsitydic_gen['w4_Dconv'] = sparsitydic_disc['w2_conv']
   sparsitydic_gen['w5_Dconv'] = sparsitydic_disc['w1_conv']

   gradients_gen = dict(zip(prn_keys_gen, grads_gen))

   gradients_abs_gen = {k: tf.abs(v) for k, v in gradients_gen.items()}
   grad_key_gen = gradients_abs_gen.keys()
   grad_shape_gen = {k: gradients_abs_gen[k].shape.as_list() for k in grad_key_gen}
   
   grad_v_gen= {}
   for key in grad_key_gen:
      grad_v_gen[key] = tf.reshape(gradients_abs_gen[key], [-1])

   normalgrad_v_gen= {}
   for key in grad_key_gen:
      normalgrad_v_gen[key] = tf.divide(grad_v_gen[key], tf.reduce_sum(grad_v_gen[key]))
   
   num_params_gen = {}
   for key in grad_key_gen:
      num_params_gen[key] = grad_v_gen[key].shape[0]

   kappa_gen = {}
   for key in grad_key_gen:
      kappa_gen[key] = int(round(num_params_gen[key] * (1 - (sparsitydic_gen[key]/100)))) 

   topk_gen ={}
   ind_gen = {}
   for key in grad_key_gen:
      topk_gen[key], ind_gen[key] = tf.nn.top_k(normalgrad_v_gen[key], k=kappa_gen[key], sorted=True)

   sp_mask_gen = {}
   mask_v_gen = {}
   for key in grad_key_gen:
      sp_mask_gen[key] = tf.SparseTensor(dense_shape=normalgrad_v_gen[key].shape.as_list(), values=tf.ones_like(ind_gen[key], dtype=tf.float32).numpy(), indices=np.expand_dims(ind_gen[key].numpy(), 1))
      mask_v_gen[key] = tf.sparse.to_dense(sp_mask_gen[key], validate_indices=False)

   final_mask_gen = {key: tf.reshape(mask_v_gen[key], grad_shape_gen[key]) for key in grad_key_gen}
   final_w_gen = apply_mask(weights_gen, final_mask_gen)

   return final_w_disc, final_mask_disc, final_w_gen, final_mask_gen
