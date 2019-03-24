# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:15:44 2018

@author: jbk48
@editor: lumyjuwon
"""

import os
import tensorflow as tf
import Bi_LSTM as Bi_LSTM
import Word2Vec as Word2Vec
import gensim
import numpy as np
import csv


def Convert2Vec(model_name, sentence):
    word_vec = []
    sub = []
    model = gensim.models.word2vec.Word2Vec.load(model_name)
    for word in sentence:
        if (word in model.wv.vocab):
            sub.append(model.wv[word])
        else:
            sub.append(np.random.uniform(-0.25, 0.25, 300))  # used for OOV words
    word_vec.append(sub)
    return word_vec

def Grade(sentence):
    tokens = W2V.tokenize(sentence)

    embedding = Convert2Vec('Data\\post.embedding', tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)
    global sess
    result = sess.run(prediction, feed_dict={X: zero_pad, seq_len: [len(tokens)]}) # tf.argmax(prediction, 1)이 여러 prediction 값중 max 값 1개만 가져옴
    point = result.ravel().tolist()
    Tag = ["IT과학", "경제", "정치", "e스포츠", "골프", "농구", "배구", "야구", "일반 스포츠", "축구", "사회", "생활문화"]
    for t, i in zip(Tag, point):
        print(t, round(i * 100, 2),"%")
        percent = t + str(round(i * 100, 2)) + "%"
        #text.write(percent)
        #text.write("\n")
    #text.write("\n")

W2V = Word2Vec.Word2Vec()

Batch_size = 1
Vector_size = 300
Maxseq_length = 500  # Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 12
keep_prob = 1.0

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)  # softmax

saver = tf.train.Saver()
init = tf.global_variables_initializer()
modelName = "Data\\Bi_LSTM.model"

sess = tf.Session()
sess.run(init)
saver.restore(sess, modelName)

while(True):
    try:
        s = input("문장을 입력하세요 : ")
        Grade(s)
    except:
        pass