import torch
import numpy
import torch.nn.functional as F
from absl import flags
from absl import app
from utils import get_data_info, read_data, load_word_embeddings
from model import IAN
from evals import *
import os
import time
import math

#setting os.environ

#configuration settings

FLAGS = flags.FLAGS
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
embedding_file_name = '/content/IAN/data/Copy of glove.840B.300d.txt'
dataset = '/content/IAN/data/laptop/'
#setting dataset and log directory

def run(model, train_data, test_data):
  print('Training ...')
  max_acc, max_f1, step = 0., 0., -1

  train_data_size = len(train_data[0])
  print(type(train_data))
  #print(train_data)
  #train_data = list(train_data)
  print(view(train_data))
  #train_data = np.asarray(train_data)

  #train_data = torch.tensor(np.asarray(train_data))
  print(type(train_data))
  #print(train_data)

  train_data = torch.stack(torch.from_numpy(train_data), dim=0)

  train_data.narrow(0, 0, 2)
  print(train_data)
  #print(train_data[1])
  #train_data= train_data.Compose([train_data.ToTensor(), ])
  #train_data = train_data.reshape((2313, 1))
  #train_data = numpy.asarray(train_data[1])
  #print(train_data.size)
  #print(train_data[1])
  #train_data = torch.nn.utils.rnn.pad_sequence(train_data, batch_first=True)
  #train_data = torch.Tensor(train_data)
  #train_data = np.asarray(train_data[0])
  #train_data = torch.from_numpy(train_data[0])
  #train_data = torch.narrow(train_data, 1, 2, ) #check on whether .Dataset matters
  #train_data = train_data.shuffle(buffer_size=train_data_size).batch(batch_size, drop_remainder=True)
  train_data = torch.utils.data.BufferedShuffleDataset(train_data, buffer_size = train_data_size)
  print(list(torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last = True)))
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, drop_last = True)

  test_data_size = len(test_data[0])
  test_data = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True)
  #test_data = torch.Tensor(test_data)
  #test_data = test_data.numpy()
  #test_data = np.asarray(test_data)
  #test_data = torch.from_numpy(test_data)
  #test_data = torch.narrow(test_data[0]) #check on whether .Dataset matters
  #test_data = test_data.batch(batch_size, drop_remainder=True)#check whether we need shuffle or not
    
  #it_train_data = iter(train_data)
  #iterator_test_data = iter(test_data)
  #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  #write to log directory
    
  for i in range(n_epoch):
    cost, predict_list, labels_list = 0., [], []
    #print(it_train_data)
    x = 0
    #data = it_train_data
    #data = it(train_data)
    #not sure about this
    for _, data in enumerate(train_loader):
      predict, labels = data
      #print(it_train_data)
      x = x + 1
      #print(data)
      #replacing tape
      predict, labels = model(data, dropout = 0.5)
      print(labels)
      loss_t = F.nll_loss(F.LogSoftmax(predict), labels)
      loss = F.mean(loss_t)
      cost += F.sum(loss_t)
        
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #torch.autograd()

      print(x)  
      predict_list.extend(F.argmax(F.softmax(predict)).numpy())
      labels_list.extend(F.argmax(labels).numpy())
      print(labels_list)
        
        
    train_acc, train_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
    train_loss = cost / train_data_size
    #write to summary
      
    cost, predict_list, labels_list = 0., [], []
    for _ in range(math.floor(test_data_size / batch_size)):
        data = iterator_test_data.next()
        predict, labels = model(data, dropout=1.0)
        #torch.nn.functional.cross_entropy
        loss_t = F.nll_loss(F.LogSoftmax(predict), labels)
    
    test_acc, test_f1, _, _ = evaluate(pred=predict_list, gold=labels_list)
    test_loss = cost/test_data_size
    #write to summary
      
    if test_acc + test_f1 > max_acc + max_f1:
      max_acc = test_acc
      max_f1 = test_f1
      step = i
      #write to saver
            
    print('epoch %s: train-loss=%.6f; train-acc=%.6f; train-f1=%.6f; test-loss=%.6f; test-acc=%.6f; test-f1=%.6f.' % (str(i), train_loss, train_acc, train_f1, test_loss, test_acc, test_f1))  
      #print to console
  print('The max accuracy of testing results: acc %.6f and macro-f1 %.6f of step %s' % (max_acc, max_f1, step))
  #print to console
  
def main(_):
  #measure time
  #loading data information
  word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(dataset, pre_processed)
  
  #loading training and test data
  train_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'train', pre_processed)
  test_data = read_data(word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, dataset + 'test', pre_processed)
  
  #loading pre-trained word vectors
  FLAGS.embedding_matrix = load_word_embeddings(embedding_file_name, FLAGS.embedding_dim, word2id)
  
  model = IAN(FLAGS)
  run(model, train_data, test_data)
  
  #measure time
  #print time cost
  
if __name__ == '__main__':
  app.run(main)
