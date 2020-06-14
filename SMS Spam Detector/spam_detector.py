"""Info about Dataset: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/"""
"""Dataset original: https://www.kaggle.com/uciml/sms-spam-collection-dataset"""

import tensorflow as tf
print('Tensorflow Version :', tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
import numpy as np
import csv
from acc_plotter import plot_accuracy

messages = []
labels = []
with open(r'spam-text-message-classification\spam.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for data in reader:
        messages.append(data[1])
        if data[0] == 'ham':
            labels.append(0)
        elif data[0] == 'spam':
            labels.append(1)

messages = np.array(messages)
labels = np.array(labels)
dataset_size = len(messages)


# HYPER-PARAMETERS
vocab_size = 1000
embedding_dims = 32
oov_token = '<OOV>'
max_length = 120


training_portion = 0.9
split = int(dataset_size * training_portion)
train_messages = messages[:split]
test_messages = messages[split:]
train_labels = labels[:split]
test_labels = labels[split:]
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_messages)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_messages)
test_sequences = tokenizer.texts_to_sequences(test_messages)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dims, input_length=max_length),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.Bidirectional(LSTM(32)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),

    ]
)
model.summary()

model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels), verbose=1)

plot_accuracy(history)

# FOR DECODING
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_padded[4]))
print(train_messages[4])


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

# todo importing vecs file and meta data for plotting
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
