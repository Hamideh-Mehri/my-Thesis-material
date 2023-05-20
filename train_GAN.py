import tensorflow as tf

def train_gan(discriminator, generator, batch_size, train_iteration,step_per_epochs, train_dataset,generator_name):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    lr = 2e-4
    decay = 6e-8

    generator_optimizer = tf.keras.optimizers.RMSprop(lr=lr, decay=decay)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=lr * 0.5, decay=decay * 0.5)

    #train_iteration=250
    BATCH_SIZE = batch_size
    noise_dim = 100
    real_label = tf.ones(shape=(BATCH_SIZE,1))
    fake_label = tf.zeros(shape=(BATCH_SIZE,1))
  
    
    for itr in range(train_iteration):
        for step in range(step_per_epochs):
  
            image_batch = next(iter(train_dataset))
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            with tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)

                real_loss = cross_entropy(tf.ones_like(real_output), real_output)
                fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
                disc_loss = real_loss + fake_loss
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))

            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            with tf.GradientTape() as gen_tape:
                generated_images = generator(noise, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_weights)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))
        print(f"\tCurrently on epoch  {itr} ")
            
            
            
            

    #save generator
    generator.save(generator_name)












# def train_gan(discriminator, generator, batch_size, train_iteration, train_dataset, generator_name):
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     lr = 2e-4
#     decay = 6e-8
#     if train_dataset == 'mnist':
#        generator_optimizer = tf.keras.optimizers.RMSprop(lr=lr, decay=decay)
#        discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=lr * 0.5, decay=decay * 0.5)
#     elif train_dataset == 'fashion-mnist':
#        generator_optimizer = tf.keras.optimizers.Adam()
#        discriminator_optimizer = tf.keras.optimizers.Adam()

#     train_iteration=25000
#     BATCH_SIZE = batch_size
#     real_label = tf.ones(shape=(BATCH_SIZE,1))
#     fake_label = tf.zeros(shape=(BATCH_SIZE,1))

#     loss_metric_disc = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
#     loss_metric_gen = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
    
#     acc_metric_disc = tf.keras.metrics.BinaryAccuracy()
#     acc_metric_gen = tf.keras.metrics.BinaryAccuracy()
    

#     plot_loss_disc = []
#     plot_acc_disc = []
#     plot_loss_gen = []
#     plot_acc_gen = []

#     for itr in range(train_iteration):
#         image_batch = next(iter(train_dataset))
#         noise_dim = 100
#         noise = tf.random.normal([BATCH_SIZE, noise_dim])
#         with tf.GradientTape() as disc_tape:
#             generated_images = generator(noise, training=True)
#             real_output = discriminator(image_batch, training=True)
#             fake_output = discriminator(generated_images, training=True)

#             real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#             fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#             disc_loss = real_loss + fake_loss
#         gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
#         discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))

#         noise = tf.random.normal([BATCH_SIZE, noise_dim])
#         with tf.GradientTape() as gen_tape:
#             generated_images = generator(noise, training=True)
#             fake_output = discriminator(generated_images, training=True)

#             gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
#         gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_weights)
#         generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))
        
#         # image_batch_val = next(iter(train_dataset))
#         # image_batch_val = tf.cast(image_batch_val, dtype=tf.dtypes.float32)
#         # noise_val = tf.random.normal([BATCH_SIZE, noise_dim])
#         # generated_images_val = generator(noise_val, training=True)
#         # generated_images_val = tf.cast(generated_images_val, dtype=tf.dtypes.float32)
#         # label = tf.concat((real_label, fake_label), axis=0)
#         # data = tf.concat((image_batch_val, generated_images_val), axis=0)
#         # logits_disc = discriminator(data, training=False)
#         # logits_gen = discriminator(generated_images_val, training=False)

#         # Update training metric.
#         # loss_metric_disc.update_state(label, logits_disc)
#         # loss_metric_gen.update_state(real_label, logits_gen)
#         # acc_metric_disc.update_state(label, logits_disc)
#         # acc_metric_gen.update_state(real_label, logits_gen)
        
#         if (itr + 1) % 100 == 0:
#             loss_disc = loss_metric_disc.result()
#             loss_gen = loss_metric_gen.result()
            
#             acc_disc = acc_metric_disc.result()
#             acc_gen = acc_metric_gen.result()
            

#             plot_loss_disc.append(loss_disc)
#             plot_loss_gen.append(loss_gen)

#             plot_acc_disc.append(acc_disc)
#             plot_acc_gen.append(acc_gen)
            
#             print('train BinaryCrossentropyloss discriminator/generator acc at iteration %d: %.4f/%.4f ' % (itr+1, float(loss_disc), float(loss_gen)))
#             print("Training accuracy discriminator/generator at iteration %d: %.4f/ %.4f" % (itr+1, float(acc_disc), float(acc_gen)))
#             acc_metric_disc.reset_states()
#             acc_metric_gen.reset_states()

#             loss_metric_disc.reset_states()
#             loss_metric_gen.reset_states()
    

#     #save generator
#     generator.save(generator_name)