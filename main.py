#Questions on model.variables and evaluate()

import tensorflow as tf
from utils import get_data_info, read_data, load_word_embeddings
from model import IAN
from evals import *
import os
import time
import math


#initiallizing environmental variables

#disables tensorflow from printing warning and error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#specifies which gpu tensorflow can use (tensorflow automatically uses all of them)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.enable_eager_execution(config=config)

#defining flag variables and setting initial values
#flags for defining values for layers
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')

#flags for initializing matrices
tf.app.flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
tf.app.flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
tf.app.flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')

#defining parameters of network
batch_size = 128
learning_rate = 0.01
n_epoch = 20
#pre_processed is a boolean variable that determines if the data has gone through transfer.py
pre_processed = 1
#specifying embedding vectors
embedding_file_name = 'data/glove.840B.300d.txt'
#destination of outputs
dataset = 'data/laptop/'
logdir = 'logs/'

#running training data and testing data through the model
def run(model, train_data, test_data):
    print('Training ...')
    #defining min values for f-score and accuracy + step value
    #f-score is measure of model's accuracy
    max_acc, max_f1, step = 0., 0., -1

    #length of tensor
    train_data_size = len(train_data[0])
    #slices of train_data tensor in the form of objects
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    #shuffles tensor along first dimension and form batches
    #buffer_size affects randomness of transformation
    train_data = train_data.shuffle(buffer_size=train_data_size).batch(batch_size, drop_remainder=True)

    #same for test_data, no shuffle
    test_data_size = len(test_data[0])
    test_data = tf.data.Dataset.from_tensor_slices(test_data)
    test_data = test_data.batch(batch_size, drop_remainder=True)

    #create an iterator for the train_data, not initialized
    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    #implements Adam algorithm using predefined learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #creates a summary file to directory
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    #loop through # of epochs
    for i in range(n_epoch):
        #define initial cost and initialize prediction and label lists
        cost, predict_list, labels_list = 0., [], []
        #initialize iterator to train_data
        iterator.make_initializer(train_data)
        #loop through each batch in train_data
        for _ in range(math.floor(train_data_size / batch_size)):
            #set data to the next value
            data = iterator.get_next()
            #records relevant operations executed inside tf.GradientTape onto a "tape", use that tape to compute the gradients of recorded computation using reverse mode differentiation.
            with tf.GradientTape() as tape:
                #send data through model with defined droupout rate, output predictions and labels
                predict, labels = model(data, dropout=0.5)
                #calculate loss
                loss_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=labels)
                loss = tf.reduce_mean(loss_t)
                #calculate cost
                cost += tf.reduce_sum(loss_t)
            #recalculate gradients on tape using loss and values
            grads = tape.gradient(loss, model.variables)
            #creates list of gradients and values and passes them to optimizer
            optimizer.apply_gradients(zip(grads, model.variables))
            #adds all elements of numpy array after passing functions
            predict_list.extend(tf.argmax(tf.nn.softmax(predict), 1).numpy())
            #adds all elements of numpy array after passing functions
            labels_list.extend(tf.argmax(labels, 1).numpy())
        #somehow evaluates accuracy and f1-score
        train_acc, train_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
        #calculate loss of all train_data
        train_loss = cost / train_data_size
        #adds to summary file
        tf.contrib.summary.scalar('train_loss', train_loss)
        tf.contrib.summary.scalar('train_acc', train_acc)
        tf.contrib.summary.scalar('train_f1', train_f1)

        #similar process for test_data, no need to calculate gradients
        cost, predict_list, labels_list = 0., [], []
        iterator.make_initializer(test_data)
        for _ in range(math.floor(test_data_size / batch_size)):
            data = iterator.get_next()
            predict, labels = model(data, dropout=1.0)
            loss_t = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predict, labels=labels)
            cost += tf.reduce_sum(loss_t)
            predict_list.extend(tf.argmax(tf.nn.softmax(predict), 1).numpy())
            labels_list.extend(tf.argmax(labels, 1).numpy())
        test_acc, test_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
        test_loss = cost / test_data_size
        tf.contrib.summary.scalar('test_loss', test_loss)
        tf.contrib.summary.scalar('test_acc', test_acc)
        tf.contrib.summary.scalar('test_f1', test_f1)

        #change values of max_acc and max_f1 from initialization of 0
        if test_acc + test_f1 > max_acc + max_f1:
            max_acc = test_acc
            max_f1 = test_f1
            #step changes to max # of epochs passed
            step = i
            #save model variables to model folder
            saver = tf.contrib.eager.Saver(model.variables)
            saver.save('models/model_iter', global_step=step)
        print(
            'epoch %s: train-loss=%.6f; train-acc=%.6f; train-f1=%.6f; test-loss=%.6f; test-acc=%.6f; test-f1=%.6f.' % (
                str(i), train_loss, train_acc, train_f1, test_loss, test_acc, test_f1))

    saver.save('models/model_final')
    print('The max accuracy of testing results: acc %.6f and macro-f1 %.6f of step %s' % (max_acc, max_f1, step))


def main(_):
    #start timer
    start_time = time.time()

    #load data
    print('Loading data info ...')
    word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(dataset, pre_processed)

    #load train_data and test_data
    print('Loading training data and testing data ...')
    train_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'train', pre_processed)
    test_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'test', pre_processed)

    #load word vectors
    print('Loading pre-trained word vectors ...')
    FLAGS.embedding_matrix = load_word_embeddings(embedding_file_name, FLAGS.embedding_dim, word2id)

    #define model using initialized FLAG values
    model = IAN(FLAGS)
    #run model with train_data and test_data
    run(model, train_data, test_data)

    #calculate time cost
    end_time = time.time()
    print('Time Costing: %s' % (end_time - start_time))

#checks for main function in file
if __name__ == '__main__':
    #sets up flags globally and runs main function
    tf.app.run()
