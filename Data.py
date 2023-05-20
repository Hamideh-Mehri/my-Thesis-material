import tensorflow as tf
import numpy as np
import itertools
rand = np.random.RandomState(2)

def read_data_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Pre-processing (normalize)
    train_images = np.divide(x_train, 255, dtype=np.float32)
    test_images = np.divide(x_test, 255, dtype=np.float32)
    mean = np.mean(train_images)
    std = np.std(train_images)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std
    train_labels = y_train
    test_labels = y_test 
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)
    train_labels = np.expand_dims(train_labels, axis=1)
    test_labels = np.expand_dims(test_labels, axis=1)
    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels}
    }
    return dataset

def read_data_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Pre-processing (normalize)
    train_images = np.divide(x_train, 255, dtype=np.float32)
    test_images = np.divide(x_test, 255, dtype=np.float32)
    mean = np.mean(train_images)
    std = np.std(train_images)
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std
    train_labels = y_train
    test_labels = y_test 
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)
    train_labels = np.expand_dims(train_labels, axis=1)
    test_labels = np.expand_dims(test_labels, axis=1)
    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels}
    }
    return dataset


def read_data_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Pre-processing (normalize)
    train_images = x_train
    test_images = x_test
    train_images = np.divide(train_images, 255, dtype=np.float32)
    test_images = np.divide(test_images, 255, dtype=np.float32)
    channel_mean = np.mean(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    channel_std = np.std(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    train_images = (train_images - channel_mean) / channel_std
    test_images = (test_images - channel_mean) / channel_std
    train_labels = y_train
    test_labels = y_test
    #create dictionary of data
    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels},
    }
    return dataset

def read_data_cifar100():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    # Pre-processing (normalize)
    train_images = x_train
    test_images = x_test
    train_images = np.divide(train_images, 255, dtype=np.float32)
    test_images = np.divide(test_images, 255, dtype=np.float32)
    channel_mean = np.mean(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    channel_std = np.std(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    train_images = (train_images - channel_mean) / channel_std
    test_images = (test_images - channel_mean) / channel_std
    train_labels = y_train
    test_labels = y_test
    #create dictionary of data
    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels},
    }
    return dataset

class Dataset(object):
    def __init__(self, datasource, **kwargs):
        self.datasource = datasource
        self.rand = np.random.RandomState(2)

        if self.datasource == 'mnist':
            self.num_classes = 10
            self.dataset = read_data_mnist()
        elif self.datasource == 'fashion-mnist':
            self.num_classes = 10
            self.dataset = read_data_fashion_mnist()
        elif self.datasource == 'cifar-10':
            self.num_classes = 10
            self.dataset = read_data_cifar10()
        elif self.datasource == 'cifar-100':
            self.num_classes = 100
            self.dataset = read_data_cifar100()
        else:
            raise NotImplementedError

        self.split_dataset('train', 'val', int(self.dataset['train']['input'].shape[0] * 0.1), self.rand)
        self.num_example = {k: self.dataset[k]['input'].shape[0] for k in self.dataset.keys()}

        self.example_generator = {
            'train': self.iterate_example('train'),
            'val': self.iterate_example('val'),
            'test': self.iterate_example('test', shuffle=False),
        }

    def iterate_example(self, mode, shuffle=True):
        epochs = itertools.count()
        for i in epochs:
            example_ids = list(range(self.num_example[mode]))
            if shuffle:
                self.rand.shuffle(example_ids)
            for example_id in example_ids:
                yield {
                    'input': self.dataset[mode]['input'][example_id],
                    'label': self.dataset[mode]['label'][example_id],
                    'id': example_id,
                }

    def get_next_batch(self, mode, batch_size):
        inputs, labels, ids = [], [], []
        for i in range(batch_size):
            example = next(self.example_generator[mode])
            inputs.append(example['input'])
            labels.append(example['label'])
            ids.append(example['id'])
        return {
            'input': np.asarray(inputs),
            'label': np.asarray(labels),
            'id': np.asarray(ids),
        }

    def generate_example_epoch(self, mode):
        example_ids = range(self.num_example[mode])
        for example_id in example_ids:
            yield {
                'input': self.dataset[mode]['input'][example_id],
                'label': self.dataset[mode]['label'][example_id],
                'id': example_id,
            }
    def split_dataset(self, source, target, number, rand):
        keys = ['input', 'label']
        indices = list(range(self.dataset[source]['input'].shape[0]))
        rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}

def preparebatch(dataset, batchsize, typeOfbatch):
   dat = Dataset(dataset)
   if typeOfbatch == 'random':
      batch = dat.get_next_batch('train', batch_size=batchsize)
      inputs = batch['input']
      targets = batch['label']
   elif typeOfbatch == 'equal':
      samples_per_class = batchsize // dat.num_classes
      datas = [[] for _ in range(dat.num_classes)]
      labels = [[] for _ in range(dat.num_classes)]
      mark = dict()
      while True:
          batch = dat.get_next_batch('train', batch_size = batchsize)
          for idx in range(batch['input'].shape[0]):
             x, y = batch['input'][idx], batch['label'][idx]
             category = y[0]
             if len(datas[category]) == samples_per_class:
                 mark[category] = True
                 continue
             datas[category].append(x)
             labels[category].append(y)
          if len(mark) == dat.num_classes:
             break
      inputs = np.array(list(itertools.chain.from_iterable(datas)))   
      targets = np.array(list(itertools.chain.from_iterable(labels)))
   return inputs, targets 

def generate_tensordata(dataset, mode):
    dat = Dataset(dataset)
    x = dat.dataset[mode]['input']
    y = dat.dataset[mode]['label']

    num_sample = dat.num_example[mode]

    def sample_generator():
        ids = list(range(num_sample))
        rand.shuffle(ids)
        for i in ids:
             yield (x[i],y[i])

    tensordata = tf.data.Dataset.from_generator(sample_generator, output_types=(tf.float32, tf.float32))
    return tensordata

