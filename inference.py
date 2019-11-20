# coding=utf-8
# @author: cer
import tensorflow as tf
from data import *
from model import Model
from my_metrics import *
from tensorflow.python import debug as tf_debug
import numpy as np


input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 871
slot_size = 122
intent_size = 22
epoch_num = 50


def get_model():
    model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size, n_layers)
    model.build()
    return model


def predict():
    #model = get_model()
    #sess = tf.Session()

    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)

    test_data = open("dataset/atis.test.w-intent.iob", "r").readlines()
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)

    

    unziped = list(zip(*index_test))
    
    #saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    tf.contrib.rnn
    saver = tf.train.import_meta_graph("/rnnIF/model.ckpt.meta")

    encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size],
                                             name='encoder_inputs')
        #The actual length of each sentence input, except for padding
    encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        
    with tf.Session() as sess:
        #tf.train.latest_checkpoint('/rnnIF/')
        saver.restore(sess, "/rnnIF/model.ckpt")
        #sess.run([y_pred], feed_dict={x: input_values})
        #output_feeds = [model.decoder_prediction, model.intent]
        #feed_dict = {model.encoder_inputs: np.transpose(unziped[0], [1, 0]),
        #                    model.encoder_inputs_actual_length: unziped[1]}
        decoder_prediction = []
        intent = []
        output_feeds = [decoder_prediction, intent]

        feed_dict = {encoder_inputs: np.transpose(unziped[0], [1, 0]),
                            encoder_inputs_actual_length: unziped[1]}


        results = sess.run(output_feeds, feed_dict=feed_dict)
        print(results)

    #saver.restore(sess, "/rnnIF/model.ckpt")

    

    



if __name__ == '__main__':
    predict()