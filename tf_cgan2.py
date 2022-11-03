import tensorflow as tf
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow_docs.vis import embed
#import matplotlib.pyplot as plt
#import imageio
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
GPUS = [0, 1, 2, 3]
devices = ["GPU:" + str(gpu_id) for gpu_id in GPUS]
# If you have *different* GPUs in your system, you probably have to set up cross_device_ops like this
strategy = tf.distribute.MirroredStrategy(devices=devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

# We'll use all the available examples from both the training and test
# sets.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = keras.utils.to_categorical(all_labels, 10)

# Batch the input data
BUFFER_SIZE = len(all_digits)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
#.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")

# Create Distributed Datasets from the datasets
train_dist_dataset = strategy.experimental_distribute_dataset(dataset)
#test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

# Create the model architecture
def create_disc():
	discriminator = keras.Sequential(
	    [
	        keras.layers.InputLayer((28, 28, discriminator_in_channels)),
	        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
	        layers.LeakyReLU(alpha=0.2),
	        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
	        layers.LeakyReLU(alpha=0.2),
	        layers.GlobalMaxPooling2D(),
	        layers.Dense(1),
	    ],
	    name="discriminator",
	)
	return discriminator
def create_gen():
	generator = keras.Sequential(
	    [
	        keras.layers.InputLayer((generator_in_channels,)),
	        # We want to generate 128 + num_classes coefficients to reshape into a
	        # 7x7x(128 + num_classes) map.
	        layers.Dense(7 * 7 * generator_in_channels),
	        layers.LeakyReLU(alpha=0.2),
	        layers.Reshape((7, 7, generator_in_channels)),
	        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
	        layers.LeakyReLU(alpha=0.2),
	        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
	        layers.LeakyReLU(alpha=0.2),
	        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
	    ],
	    name="generator",
	)
	return generator
with strategy.scope():
	# We will use sparse categorical crossentropy as always. But, instead of having the loss function
    # manage the map reduce across GPUs for us, we'll do it ourselves with a simple algorithm.
    # Remember -- the map reduce is how the losses get aggregated
    # Set reduction to `none` so we can do the reduction afterwards and divide byglobal batch size.
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    #compute_loss
    def loss_fn(labels, predictions):
        # Compute Loss uses the loss object to compute the loss
        # Notice that per_example_loss will have an entry per GPU
        # so in this case there'll be 2 -- i.e. the loss for each replica
        per_example_loss = loss_object(labels, predictions)
        # You can print it to see it -- you'll get output like this:
        # Tensor("sparse_categorical_crossentropy/weighted_loss/Mul:0", shape=(48,), dtype=float32, device=/job:localhost/replica:0/task:0/device:GPU:0)
        # Tensor("replica_1/sparse_categorical_crossentropy/weighted_loss/Mul:0", shape=(48,), dtype=float32, device=/job:localhost/replica:0/task:0/device:GPU:1)
        # Note in particular that replica_0 isn't named in the weighted_loss -- the first is unnamed, the second is replica_1 etc
        print(per_example_loss)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    # Accuracy on train and test will be SparseCategoricalAccuracy
    #train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    # Optimizer will be Adam
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    # Create the model within the scope
    discriminator = create_disc()
    generator = create_gen()

    gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
    disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
"""
class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
"""
def train_step(data):
    # Unpack the data.
    real_images, one_hot_labels = data

    # Add dummy dimensions to the labels so that they can be concatenated with
    # the images. This is for the discriminator.
    image_one_hot_labels = one_hot_labels[:, :, None, None]
    image_one_hot_labels = tf.repeat(
        image_one_hot_labels, repeats=[image_size * image_size]
    )
    image_one_hot_labels = tf.reshape(
        image_one_hot_labels, (-1, image_size, image_size, num_classes)
    )

    # Sample random points in the latent space and concatenate the labels.
    # This is for the generator.
    batch_size = tf.shape(real_images)[0]
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    random_vector_labels = tf.concat(
        [random_latent_vectors, one_hot_labels], axis=1
    )

    # Decode the noise (guided by labels) to fake images.
    generated_images = generator(random_vector_labels)

    # Combine them with real images. Note that we are concatenating the labels
    # with these images here.
    fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
    real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
    combined_images = tf.concat(
        [fake_image_and_labels, real_image_and_labels], axis=0
    )

    # Assemble labels discriminating real from fake images.
    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
    )

    # Train the discriminator.
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(
        zip(grads, discriminator.trainable_weights)
    )

    # Sample random points in the latent space.
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    random_vector_labels = tf.concat(
        [random_latent_vectors, one_hot_labels], axis=1
    )

    # Assemble labels that say "all real images".
    misleading_labels = tf.zeros((batch_size, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        fake_images = generator(random_vector_labels)
        fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
        predictions = discriminator(fake_image_and_labels)
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    #
    #gen_loss_tracker.update_state(g_loss)
    return [d_loss,g_loss]
"""
    # Monitor loss.
    gen_loss_tracker.update_state(g_loss)
    disc_loss_tracker.update_state(d_loss)
    return {
        "g_loss": gen_loss_tracker.result(),
        "d_loss": disc_loss_tracker.result(),
    }
"""
# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  (per_replica_d_losses,per_replica_g_losses) = strategy.run(train_step, args=(dataset_inputs,))
  #tf.print(per_replica_losses.values)
  reduce_d_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_d_losses, axis=None)
  reduce_g_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_g_losses, axis=None)
  return reduce_d_loss,reduce_g_loss
"""
def train_step(inputs):
  images, labels = inputs
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = compute_loss(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_accuracy.update_state(labels, predictions)
  return loss
"""
EPOCHS = 20
for epoch in range(EPOCHS):
  # Do Training
  (total_d_loss,total_g_loss) = (0.0,0.0)
  num_batches = 0
  for batch in train_dist_dataset:
  	(reduce_d_loss,reduce_g_loss) = distributed_train_step(batch)
  	total_d_loss += reduce_d_loss
  	total_g_loss += reduce_g_loss
  	num_batches += 1
  train_d_loss = total_d_loss / num_batches
  train_g_loss = total_g_loss / num_batches

  template = ("Epoch {}, Loss disc: {}, Loss gen: {}")#, Accuracy: {}, Test Loss: {}, " "Test Accuracy: {}

  print (template.format(epoch+1, train_d_loss, train_g_loss))
  #, train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100

  #test_loss.reset_states()
  #train_accuracy.reset_states()
  #test_accuracy.reset_states()