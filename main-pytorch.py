import torch
from utils import get_data_info, read_data, load_word_embeddings
from model import IAN
from evals import *
import os
import time
import math

#setting os.environ

#configuration settings

FLAGS = flag.FLAGS
flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
flags.DEFINE_integer('n_class', 3, 'number of distinct class')
flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')

flags.DEFINE_integer('max_aspect_len', 0, 'max length of aspects')
flags.DEFINE_integer('max_context_len', 0, 'max length of contexts')
flags.DEFINE_string('embedding_matrix', '', 'word ids to word vectors')

batch_size = 128
learning_rate = 0.01
n_epoch = 20
pre_processed = 1
embedding_file_name = 'data/glove.840B.300d.txt'
#setting dataset and log directory

def run(model, train_data, test_data):
  print('Training ...')
    max_acc, max_f1, step = 0., 0., -1

    train_data_size = len(train_data[0])
    train_data = torch.Tensor.narrow(train_data) #check on whether .Dataset matters
    train_data = train_data.shuffle(buffer_size=train_data_size).batch(batch_size, drop_remainder=True)
    
    test_data_size = len(test_data[0])
    test_data = torch.Tensor.narrow(test_data) #check on whether .Dataset matters
    test_data = test_data.batch(batch_size, drop_remainder=True)#check whether we need shuffle or not
    
    iterator = torchtext.data.iterator(train_data.output_types, train_data.output_shapes)
    optimizer = optim.Adam(lr=learning_rate)
    #write to log directory
    
    for i in range(n_epoch):
      cost, predict_list, labels_list = 0., [], []
      #not sure about this
      iterator = torchtext.data.iterator(train_data.output_types, train_data.output_shapes)
      for _ in range(math.floor(train_data_size / batch_size)):
        data = iterator.get_next()
        #replacing tape
        predict, labels = model(data, dropout = 0.5)
        #loss = compute_loss(model, x)
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #torch.autograd()
        
        predict_list.extend(tf.argmax(tf.nn.softmax(predict), 1).numpy())
        labels_list.extend(tf.argmax(labels, 1).numpy())
      train_acc, train_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
      train_loss = cost / train_data_size
      #write to summary
      
      cost, predict_list, labels_list = 0., [], []
      iterator = torchtext.data.iterator(test_data.output_types, test_data.output_shapes)
      for _ in range(math.floor(test_data_size / batch_size)):
            data = iterator.get_next()
            predict, labels = model(data, dropout=1.0)
            
