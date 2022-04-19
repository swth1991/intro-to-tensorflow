#%%
import hashlib
#%%
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

os.chdir("/home/jhkim/src/vscode/machine-learning/intro-to-tensorflow")

#%%

def download(url, file):
    if not os.path.isfile(file):
        print("Downloading " + file + ".....")
        urlretrieve(url, file)
        print("Download Finished")

download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',\
        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',\
        'notMNIST_test.zip file is corrupted.  Remove the file and try again.'

print('All files downloaded.')


#%%
def uncompress_features_labels(file):
    features = []
    labels = []

    with ZipFile(file) as zipf:
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        for filename in filenames_pbar:
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()

                    feature = np.array(image, dtype=np.float32).flatten()
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

train_features, train_labels = uncompress_features_labels("notMNIST_train.zip")
test_features, test_labels = uncompress_features_labels("notMNIST_test.zip")
#%%
print("Labels....")
print(train_labels[30000:30010])


#%%
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

is_features_normal = False
is_labels_encod = False

print("All features and labes uncompressed.")

#%%

def normalize_grayscale(image_data):
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True

print("Featues Normalized.")

if not is_labels_encod:
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True
print("Labels On-Hot Encode.")

#%%

train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)
print('Training features and labels randomized and split.')
#%%
pickle_file = 'notMNIST.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('notMNIST.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')
#%%

import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt 

from tensorflow.python.client import device_lib

print(tf.__version__)
print(device_lib.list_local_devices())

#%%
# In order to debugging active path which is sometimes different between ipython kernel and vscode debugger.
"""
print (os.path.curdir)
print(os.listdir())
pickle_file = 'notMNIST.pickle'
print(os.path.isfile(pickle_file))
print(os.path.isfile('notMNIST.pickle'))

if os.path.isfile(pickle_file):
    with open(pickle_file, 'rb') as f:
        print(f.name)
"""

#%%
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    train_features = pickle_data['train_dataset']
    train_labels = pickle_data['train_labels']
    valid_features = pickle_data['valid_dataset']
    valid_labels = pickle_data['valid_labels']
    test_features = pickle_data['test_dataset']
    test_labels = pickle_data['test_labels']

print("Data and modules loaded.")


#%%
tf.compat.v1.disable_eager_execution()

features_count = 784
labels_count = 10

features = tf.compat.v1.placeholder(tf.float32, shape=[None, features_count])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, labels_count])

weights = tf.Variable(tf.random.truncated_normal((features_count, labels_count)))
biases = tf.Variable(tf.zeros(labels_count))

#%%
#Test Cases
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features.shape == None or (\
    features.shape.dims[0].value is None and\
    features.shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels.shape  == None or (\
    labels.shape.dims[0].value is None and\
    labels.shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
assert weights.shape == (784, 10), 'The shape of weights is incorrect'
assert biases.shape == (10), 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'

#%%

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# Linear Function WX + b
logits = tf.matmul(features, weights) + biases

prediction = tf.nn.softmax(logits)

# Cross entropy
#cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
# Training loss
#loss = tf.reduce_mean(cross_entropy)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(labels), logits=logits)

# Create an operation that initializes all variables
init = tf.compat.v1.global_variables_initializer()

# Test Cases
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=config) as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    session.run(loss, feed_dict=test_feed_dict)
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'

print('Tests Passed!')

#%%
is_correct_prediction = tf.equal(tf.argmax(input=prediction, axis=1), tf.argmax(input=labels, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(is_correct_prediction, tf.float32))


batch_size = 128
epochs = 10
learning_rate = 0.002

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
validation_accuracy = 0.0

log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.compat.v1.Session() as session:
    session.run(init)
    batch_count = math.ceil(len(train_features)/batch_size)

    for epoch_i in range(epochs):
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

        for batch_i in batches_pbar:
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start+batch_size]
            batch_labels = train_labels[batch_start:batch_start+batch_size]

            _, l = session.run([optimizer, loss], feed_dict={features:batch_features, labels:batch_labels})

            if not batch_i % log_batch_step:
                train_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(train_accuracy)
                valid_acc_batch.append(validation_accuracy)

        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title("Loss")
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title("Accuracy")
acc_plot.plot(batches, train_acc_batch, 'r', label="Training Accuracy")
acc_plot.plot(batches, valid_acc_batch, 'x', label="Validation Accuracy")
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print("Validation accuracy at {}".format(validation_accuracy))

#%%
test_accuracy = 0.0
epochs = 1

with tf.compat.v1.Session() as session:
    
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)


assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))

# %%
