
# Language Translation
In this project, I'm going to take a peek into the realm of neural network machine translation.  I'll train a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, I'll simply train with a small portion of the English corpus.


```python
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028

    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .

    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .


## Implement Preprocessing Function
### Text to Word Ids
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of each sentence from `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
from collections import Counter

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function

    source_id_text = [[source_vocab_to_int[word] for word in sentence.split()] for sentence in source_text.split('\n')]
    target_id_text = [[target_vocab_to_int[word] for word in sentence.split()] + [target_vocab_to_int['<EOS>']] for sentence in target_text.split('\n')]

    return (source_id_text, target_id_text)

```

    Tests Passed


### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0


## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoding_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.

Return the placeholders in the following the tuple (Input, Targets, Learing Rate, Keep Probability)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    input_ = tf.placeholder(tf.int32, [None, None], name='input')
    targets_ = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
    keep_probability_ = tf.placeholder(tf.float32, name='keep_prob')
    return (input_, targets_, learning_rate_, keep_probability_)

```

    Tests Passed


### Process Decoding Input
Implement `process_decoding_input` using TensorFlow to remove the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


```python
def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    #target_id_text = [[target_vocab_to_int['<GO>'] + [target_vocab_to_int[word] for word in sentence.split()]] for sentence in target_data[batch_size]]
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    target_id_text = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
    return target_id_text

```

    Tests Passed


### Encoding
Implement `encoding_layer()` to create a Encoder RNN layer using [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn).


```python
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    _, enc_state = tf.nn.dynamic_rnn(tf.contrib.rnn.DropoutWrapper(enc_cell,keep_prob), rnn_inputs, dtype=tf.float32)
    return enc_state

```

    Tests Passed


### Decoding - Training
Create training logits using [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).  Apply the `output_fn` to the [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) outputs.


```python
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    # TODO: Implement Function
    with tf.variable_scope("decoding") as decoding_scope:
        # Training Decoder
        train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
        train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, train_decoder_fn, dec_embed_input, sequence_length, scope=decoding_scope)

        # Apply output function
        train_logits =  output_fn(tf.nn.dropout(train_pred, keep_prob))
    return train_logits

```

    Tests Passed


### Decoding - Inference
Create inference logits using [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: Maximum length of
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    # TODO: Implement Function
    with tf.variable_scope("decoding") as decoding_scope:
        # Inference Decoder
        infer_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
        output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
        maximum_length, vocab_size)

        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, infer_decoder_fn, scope=decoding_scope)
    return tf.nn.dropout(inference_logits, keep_prob)

```

    Tests Passed


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

- Create RNN cell for decoding using `rnn_size` and `num_layers`.
- Create the output fuction using [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) to transform it's input, logits, to class logits.
- Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` function to get the training logits.
- Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    with tf.variable_scope("decoding", reuse=None) as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
        training_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)

    with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        inference_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], sequence_length-1, vocab_size, decoding_scope, output_fn, keep_prob)

    return training_logits, inference_logits


```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:

- Apply embedding to the input data for the encoder.
- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)`.
- Process target data using your `process_decoding_input(target_data, target_vocab_to_int, batch_size)` function.
- Apply embedding to the target data for the decoder.
- Decode the encoded input using your `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)`.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    encoder_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)

    # Decoding layer
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    train, infer = decoding_layer(dec_embed_input, dec_embeddings, encoder_state, target_vocab_size, sequence_length, rnn_size,
                              num_layers, target_vocab_to_int, keep_prob)

    return (train, infer)

```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability


```python
# Number of Epochs
epochs = 20
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 128
# Number of Layers
num_layers = 3
# Embedding Size
encoding_embedding_size = 64
decoding_embedding_size = 64
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.7
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
```

### Train
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


```python
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target_batch,
            [(0,0),(0,max_seq - target_batch.shape[1]), (0,0)],
            'constant')
    if max_seq - batch_train_logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})

            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})

            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/269 - Train Accuracy:  0.257, Validation Accuracy:  0.325, Loss:  5.886
    Epoch   0 Batch    1/269 - Train Accuracy:  0.244, Validation Accuracy:  0.322, Loss:  5.803
    Epoch   0 Batch    2/269 - Train Accuracy:  0.277, Validation Accuracy:  0.322, Loss:  5.678
    Epoch   0 Batch    3/269 - Train Accuracy:  0.256, Validation Accuracy:  0.322, Loss:  5.543
    Epoch   0 Batch    4/269 - Train Accuracy:  0.242, Validation Accuracy:  0.319, Loss:  5.352
    Epoch   0 Batch    5/269 - Train Accuracy:  0.238, Validation Accuracy:  0.315, Loss:  5.134
    Epoch   0 Batch    6/269 - Train Accuracy:  0.281, Validation Accuracy:  0.313, Loss:  4.795
    Epoch   0 Batch    7/269 - Train Accuracy:  0.283, Validation Accuracy:  0.316, Loss:  4.610
    Epoch   0 Batch    8/269 - Train Accuracy:  0.253, Validation Accuracy:  0.322, Loss:  4.582
    Epoch   0 Batch    9/269 - Train Accuracy:  0.280, Validation Accuracy:  0.322, Loss:  4.341
    Epoch   0 Batch   10/269 - Train Accuracy:  0.265, Validation Accuracy:  0.339, Loss:  4.375
    Epoch   0 Batch   11/269 - Train Accuracy:  0.305, Validation Accuracy:  0.341, Loss:  4.120
    Epoch   0 Batch   12/269 - Train Accuracy:  0.279, Validation Accuracy:  0.342, Loss:  4.161
    Epoch   0 Batch   13/269 - Train Accuracy:  0.344, Validation Accuracy:  0.343, Loss:  3.754
    Epoch   0 Batch   14/269 - Train Accuracy:  0.305, Validation Accuracy:  0.343, Loss:  3.842
    Epoch   0 Batch   15/269 - Train Accuracy:  0.296, Validation Accuracy:  0.343, Loss:  3.802
    Epoch   0 Batch   16/269 - Train Accuracy:  0.312, Validation Accuracy:  0.343, Loss:  3.699
    Epoch   0 Batch   17/269 - Train Accuracy:  0.302, Validation Accuracy:  0.343, Loss:  3.646
    Epoch   0 Batch   18/269 - Train Accuracy:  0.271, Validation Accuracy:  0.343, Loss:  3.729
    Epoch   0 Batch   19/269 - Train Accuracy:  0.341, Validation Accuracy:  0.343, Loss:  3.400
    Epoch   0 Batch   20/269 - Train Accuracy:  0.275, Validation Accuracy:  0.343, Loss:  3.630
    Epoch   0 Batch   21/269 - Train Accuracy:  0.277, Validation Accuracy:  0.343, Loss:  3.615
    Epoch   0 Batch   22/269 - Train Accuracy:  0.315, Validation Accuracy:  0.343, Loss:  3.423
    Epoch   0 Batch   23/269 - Train Accuracy:  0.326, Validation Accuracy:  0.343, Loss:  3.380
    Epoch   0 Batch   24/269 - Train Accuracy:  0.272, Validation Accuracy:  0.343, Loss:  3.516
    Epoch   0 Batch   25/269 - Train Accuracy:  0.278, Validation Accuracy:  0.343, Loss:  3.489
    Epoch   0 Batch   26/269 - Train Accuracy:  0.344, Validation Accuracy:  0.343, Loss:  3.169
    Epoch   0 Batch   27/269 - Train Accuracy:  0.328, Validation Accuracy:  0.360, Loss:  3.306
    Epoch   0 Batch   28/269 - Train Accuracy:  0.285, Validation Accuracy:  0.362, Loss:  3.459
    Epoch   0 Batch   29/269 - Train Accuracy:  0.305, Validation Accuracy:  0.371, Loss:  3.404
    Epoch   0 Batch   30/269 - Train Accuracy:  0.330, Validation Accuracy:  0.368, Loss:  3.256
    Epoch   0 Batch   31/269 - Train Accuracy:  0.343, Validation Accuracy:  0.370, Loss:  3.223
    Epoch   0 Batch   32/269 - Train Accuracy:  0.334, Validation Accuracy:  0.371, Loss:  3.235
    Epoch   0 Batch   33/269 - Train Accuracy:  0.345, Validation Accuracy:  0.374, Loss:  3.155
    Epoch   0 Batch   34/269 - Train Accuracy:  0.344, Validation Accuracy:  0.375, Loss:  3.158
    Epoch   0 Batch   35/269 - Train Accuracy:  0.345, Validation Accuracy:  0.375, Loss:  3.127
    Epoch   0 Batch   36/269 - Train Accuracy:  0.344, Validation Accuracy:  0.375, Loss:  3.125
    Epoch   0 Batch   37/269 - Train Accuracy:  0.346, Validation Accuracy:  0.375, Loss:  3.093
    Epoch   0 Batch   38/269 - Train Accuracy:  0.342, Validation Accuracy:  0.373, Loss:  3.087
    Epoch   0 Batch   39/269 - Train Accuracy:  0.340, Validation Accuracy:  0.373, Loss:  3.059
    Epoch   0 Batch   40/269 - Train Accuracy:  0.313, Validation Accuracy:  0.376, Loss:  3.184
    Epoch   0 Batch   41/269 - Train Accuracy:  0.342, Validation Accuracy:  0.377, Loss:  3.030
    Epoch   0 Batch   42/269 - Train Accuracy:  0.373, Validation Accuracy:  0.376, Loss:  2.890
    Epoch   0 Batch   43/269 - Train Accuracy:  0.322, Validation Accuracy:  0.377, Loss:  3.126
    Epoch   0 Batch   44/269 - Train Accuracy:  0.350, Validation Accuracy:  0.377, Loss:  2.988
    Epoch   0 Batch   45/269 - Train Accuracy:  0.312, Validation Accuracy:  0.374, Loss:  3.123
    Epoch   0 Batch   46/269 - Train Accuracy:  0.306, Validation Accuracy:  0.374, Loss:  3.133
    Epoch   0 Batch   47/269 - Train Accuracy:  0.379, Validation Accuracy:  0.381, Loss:  2.823
    Epoch   0 Batch   48/269 - Train Accuracy:  0.354, Validation Accuracy:  0.381, Loss:  2.941
    Epoch   0 Batch   49/269 - Train Accuracy:  0.326, Validation Accuracy:  0.383, Loss:  3.068
    Epoch   0 Batch   50/269 - Train Accuracy:  0.341, Validation Accuracy:  0.393, Loss:  3.062
    Epoch   0 Batch   51/269 - Train Accuracy:  0.357, Validation Accuracy:  0.394, Loss:  2.969
    Epoch   0 Batch   52/269 - Train Accuracy:  0.367, Validation Accuracy:  0.395, Loss:  2.897
    Epoch   0 Batch   53/269 - Train Accuracy:  0.344, Validation Accuracy:  0.402, Loss:  3.031
    Epoch   0 Batch   54/269 - Train Accuracy:  0.343, Validation Accuracy:  0.403, Loss:  3.039
    Epoch   0 Batch   55/269 - Train Accuracy:  0.372, Validation Accuracy:  0.403, Loss:  2.875
    Epoch   0 Batch   56/269 - Train Accuracy:  0.378, Validation Accuracy:  0.407, Loss:  2.880
    Epoch   0 Batch   57/269 - Train Accuracy:  0.381, Validation Accuracy:  0.409, Loss:  2.874
    Epoch   0 Batch   58/269 - Train Accuracy:  0.375, Validation Accuracy:  0.404, Loss:  2.870
    Epoch   0 Batch   59/269 - Train Accuracy:  0.376, Validation Accuracy:  0.407, Loss:  2.846
    Epoch   0 Batch   60/269 - Train Accuracy:  0.401, Validation Accuracy:  0.414, Loss:  2.756
    Epoch   0 Batch   61/269 - Train Accuracy:  0.411, Validation Accuracy:  0.410, Loss:  2.710
    Epoch   0 Batch   62/269 - Train Accuracy:  0.405, Validation Accuracy:  0.414, Loss:  2.731
    Epoch   0 Batch   63/269 - Train Accuracy:  0.392, Validation Accuracy:  0.422, Loss:  2.810
    Epoch   0 Batch   64/269 - Train Accuracy:  0.382, Validation Accuracy:  0.414, Loss:  2.816
    Epoch   0 Batch   65/269 - Train Accuracy:  0.382, Validation Accuracy:  0.413, Loss:  2.789
    Epoch   0 Batch   66/269 - Train Accuracy:  0.413, Validation Accuracy:  0.421, Loss:  2.713
    Epoch   0 Batch   67/269 - Train Accuracy:  0.389, Validation Accuracy:  0.423, Loss:  2.808
    Epoch   0 Batch   68/269 - Train Accuracy:  0.385, Validation Accuracy:  0.414, Loss:  2.787
    Epoch   0 Batch   69/269 - Train Accuracy:  0.354, Validation Accuracy:  0.418, Loss:  2.932
    Epoch   0 Batch   70/269 - Train Accuracy:  0.406, Validation Accuracy:  0.427, Loss:  2.753
    Epoch   0 Batch   71/269 - Train Accuracy:  0.360, Validation Accuracy:  0.420, Loss:  2.895
    Epoch   0 Batch   72/269 - Train Accuracy:  0.427, Validation Accuracy:  0.431, Loss:  2.660
    Epoch   0 Batch   73/269 - Train Accuracy:  0.410, Validation Accuracy:  0.434, Loss:  2.756
    Epoch   0 Batch   74/269 - Train Accuracy:  0.368, Validation Accuracy:  0.425, Loss:  2.848
    Epoch   0 Batch   75/269 - Train Accuracy:  0.402, Validation Accuracy:  0.433, Loss:  2.727
    Epoch   0 Batch   76/269 - Train Accuracy:  0.397, Validation Accuracy:  0.438, Loss:  2.768
    Epoch   0 Batch   77/269 - Train Accuracy:  0.406, Validation Accuracy:  0.430, Loss:  2.714
    Epoch   0 Batch   78/269 - Train Accuracy:  0.410, Validation Accuracy:  0.438, Loss:  2.732
    Epoch   0 Batch   79/269 - Train Accuracy:  0.403, Validation Accuracy:  0.436, Loss:  2.714
    Epoch   0 Batch   80/269 - Train Accuracy:  0.412, Validation Accuracy:  0.433, Loss:  2.651
    Epoch   0 Batch   81/269 - Train Accuracy:  0.419, Validation Accuracy:  0.445, Loss:  2.713
    Epoch   0 Batch   82/269 - Train Accuracy:  0.430, Validation Accuracy:  0.450, Loss:  2.652
    Epoch   0 Batch   83/269 - Train Accuracy:  0.418, Validation Accuracy:  0.441, Loss:  2.638
    Epoch   0 Batch   84/269 - Train Accuracy:  0.417, Validation Accuracy:  0.444, Loss:  2.653
    Epoch   0 Batch   85/269 - Train Accuracy:  0.414, Validation Accuracy:  0.448, Loss:  2.677
    Epoch   0 Batch   86/269 - Train Accuracy:  0.407, Validation Accuracy:  0.442, Loss:  2.675
    Epoch   0 Batch   87/269 - Train Accuracy:  0.373, Validation Accuracy:  0.438, Loss:  2.810
    Epoch   0 Batch   88/269 - Train Accuracy:  0.422, Validation Accuracy:  0.448, Loss:  2.635
    Epoch   0 Batch   89/269 - Train Accuracy:  0.427, Validation Accuracy:  0.450, Loss:  2.600
    Epoch   0 Batch   90/269 - Train Accuracy:  0.392, Validation Accuracy:  0.457, Loss:  2.765
    Epoch   0 Batch   91/269 - Train Accuracy:  0.411, Validation Accuracy:  0.440, Loss:  2.617
    Epoch   0 Batch   92/269 - Train Accuracy:  0.425, Validation Accuracy:  0.451, Loss:  2.606
    Epoch   0 Batch   93/269 - Train Accuracy:  0.444, Validation Accuracy:  0.448, Loss:  2.496
    Epoch   0 Batch   94/269 - Train Accuracy:  0.424, Validation Accuracy:  0.451, Loss:  2.594
    Epoch   0 Batch   95/269 - Train Accuracy:  0.429, Validation Accuracy:  0.455, Loss:  2.585
    Epoch   0 Batch   96/269 - Train Accuracy:  0.434, Validation Accuracy:  0.462, Loss:  2.563
    Epoch   0 Batch   97/269 - Train Accuracy:  0.428, Validation Accuracy:  0.453, Loss:  2.576
    Epoch   0 Batch   98/269 - Train Accuracy:  0.437, Validation Accuracy:  0.454, Loss:  2.516
    Epoch   0 Batch   99/269 - Train Accuracy:  0.417, Validation Accuracy:  0.468, Loss:  2.683
    Epoch   0 Batch  100/269 - Train Accuracy:  0.439, Validation Accuracy:  0.452, Loss:  2.515
    Epoch   0 Batch  101/269 - Train Accuracy:  0.403, Validation Accuracy:  0.458, Loss:  2.661
    Epoch   0 Batch  102/269 - Train Accuracy:  0.438, Validation Accuracy:  0.467, Loss:  2.521
    Epoch   0 Batch  103/269 - Train Accuracy:  0.440, Validation Accuracy:  0.467, Loss:  2.524
    Epoch   0 Batch  104/269 - Train Accuracy:  0.427, Validation Accuracy:  0.457, Loss:  2.513
    Epoch   0 Batch  105/269 - Train Accuracy:  0.444, Validation Accuracy:  0.474, Loss:  2.511
    Epoch   0 Batch  106/269 - Train Accuracy:  0.448, Validation Accuracy:  0.482, Loss:  2.512
    Epoch   0 Batch  107/269 - Train Accuracy:  0.395, Validation Accuracy:  0.463, Loss:  2.644
    Epoch   0 Batch  108/269 - Train Accuracy:  0.427, Validation Accuracy:  0.463, Loss:  2.493
    Epoch   0 Batch  109/269 - Train Accuracy:  0.435, Validation Accuracy:  0.469, Loss:  2.489
    Epoch   0 Batch  110/269 - Train Accuracy:  0.433, Validation Accuracy:  0.464, Loss:  2.479
    Epoch   0 Batch  111/269 - Train Accuracy:  0.409, Validation Accuracy:  0.466, Loss:  2.601
    Epoch   0 Batch  112/269 - Train Accuracy:  0.450, Validation Accuracy:  0.482, Loss:  2.435
    Epoch   0 Batch  113/269 - Train Accuracy:  0.470, Validation Accuracy:  0.475, Loss:  2.335
    Epoch   0 Batch  114/269 - Train Accuracy:  0.436, Validation Accuracy:  0.469, Loss:  2.424
    Epoch   0 Batch  115/269 - Train Accuracy:  0.421, Validation Accuracy:  0.473, Loss:  2.526
    Epoch   0 Batch  116/269 - Train Accuracy:  0.438, Validation Accuracy:  0.465, Loss:  2.421
    Epoch   0 Batch  117/269 - Train Accuracy:  0.443, Validation Accuracy:  0.474, Loss:  2.416
    Epoch   0 Batch  118/269 - Train Accuracy:  0.446, Validation Accuracy:  0.468, Loss:  2.339
    Epoch   0 Batch  119/269 - Train Accuracy:  0.442, Validation Accuracy:  0.485, Loss:  2.494
    Epoch   0 Batch  120/269 - Train Accuracy:  0.386, Validation Accuracy:  0.449, Loss:  2.514
    Epoch   0 Batch  121/269 - Train Accuracy:  0.452, Validation Accuracy:  0.485, Loss:  2.469
    Epoch   0 Batch  122/269 - Train Accuracy:  0.462, Validation Accuracy:  0.482, Loss:  2.373
    Epoch   0 Batch  123/269 - Train Accuracy:  0.386, Validation Accuracy:  0.456, Loss:  2.491
    Epoch   0 Batch  124/269 - Train Accuracy:  0.449, Validation Accuracy:  0.478, Loss:  2.417
    Epoch   0 Batch  125/269 - Train Accuracy:  0.461, Validation Accuracy:  0.487, Loss:  2.339
    Epoch   0 Batch  126/269 - Train Accuracy:  0.450, Validation Accuracy:  0.471, Loss:  2.343
    Epoch   0 Batch  127/269 - Train Accuracy:  0.413, Validation Accuracy:  0.471, Loss:  2.467
    Epoch   0 Batch  128/269 - Train Accuracy:  0.472, Validation Accuracy:  0.485, Loss:  2.328
    Epoch   0 Batch  129/269 - Train Accuracy:  0.441, Validation Accuracy:  0.472, Loss:  2.374
    Epoch   0 Batch  130/269 - Train Accuracy:  0.391, Validation Accuracy:  0.466, Loss:  2.483
    Epoch   0 Batch  131/269 - Train Accuracy:  0.445, Validation Accuracy:  0.489, Loss:  2.429
    Epoch   0 Batch  132/269 - Train Accuracy:  0.453, Validation Accuracy:  0.487, Loss:  2.348
    Epoch   0 Batch  133/269 - Train Accuracy:  0.449, Validation Accuracy:  0.482, Loss:  2.306
    Epoch   0 Batch  134/269 - Train Accuracy:  0.430, Validation Accuracy:  0.489, Loss:  2.401
    Epoch   0 Batch  135/269 - Train Accuracy:  0.424, Validation Accuracy:  0.491, Loss:  2.461
    Epoch   0 Batch  136/269 - Train Accuracy:  0.418, Validation Accuracy:  0.481, Loss:  2.420
    Epoch   0 Batch  137/269 - Train Accuracy:  0.447, Validation Accuracy:  0.494, Loss:  2.406
    Epoch   0 Batch  138/269 - Train Accuracy:  0.456, Validation Accuracy:  0.496, Loss:  2.348
    Epoch   0 Batch  139/269 - Train Accuracy:  0.460, Validation Accuracy:  0.482, Loss:  2.258
    Epoch   0 Batch  140/269 - Train Accuracy:  0.470, Validation Accuracy:  0.490, Loss:  2.282
    Epoch   0 Batch  141/269 - Train Accuracy:  0.459, Validation Accuracy:  0.492, Loss:  2.320
    Epoch   0 Batch  142/269 - Train Accuracy:  0.453, Validation Accuracy:  0.481, Loss:  2.260
    Epoch   0 Batch  143/269 - Train Accuracy:  0.471, Validation Accuracy:  0.494, Loss:  2.283
    Epoch   0 Batch  144/269 - Train Accuracy:  0.468, Validation Accuracy:  0.492, Loss:  2.252
    Epoch   0 Batch  145/269 - Train Accuracy:  0.454, Validation Accuracy:  0.490, Loss:  2.254
    Epoch   0 Batch  146/269 - Train Accuracy:  0.467, Validation Accuracy:  0.492, Loss:  2.234
    Epoch   0 Batch  147/269 - Train Accuracy:  0.488, Validation Accuracy:  0.490, Loss:  2.162
    Epoch   0 Batch  148/269 - Train Accuracy:  0.459, Validation Accuracy:  0.494, Loss:  2.294
    Epoch   0 Batch  149/269 - Train Accuracy:  0.471, Validation Accuracy:  0.493, Loss:  2.223
    Epoch   0 Batch  150/269 - Train Accuracy:  0.470, Validation Accuracy:  0.496, Loss:  2.227
    Epoch   0 Batch  151/269 - Train Accuracy:  0.504, Validation Accuracy:  0.498, Loss:  2.120
    Epoch   0 Batch  152/269 - Train Accuracy:  0.464, Validation Accuracy:  0.496, Loss:  2.225
    Epoch   0 Batch  153/269 - Train Accuracy:  0.469, Validation Accuracy:  0.494, Loss:  2.216
    Epoch   0 Batch  154/269 - Train Accuracy:  0.435, Validation Accuracy:  0.499, Loss:  2.354
    Epoch   0 Batch  155/269 - Train Accuracy:  0.488, Validation Accuracy:  0.488, Loss:  2.108
    Epoch   0 Batch  156/269 - Train Accuracy:  0.465, Validation Accuracy:  0.504, Loss:  2.268
    Epoch   0 Batch  157/269 - Train Accuracy:  0.470, Validation Accuracy:  0.497, Loss:  2.214
    Epoch   0 Batch  158/269 - Train Accuracy:  0.456, Validation Accuracy:  0.488, Loss:  2.175
    Epoch   0 Batch  159/269 - Train Accuracy:  0.471, Validation Accuracy:  0.494, Loss:  2.215
    Epoch   0 Batch  160/269 - Train Accuracy:  0.461, Validation Accuracy:  0.494, Loss:  2.231
    Epoch   0 Batch  161/269 - Train Accuracy:  0.461, Validation Accuracy:  0.497, Loss:  2.210
    Epoch   0 Batch  162/269 - Train Accuracy:  0.483, Validation Accuracy:  0.504, Loss:  2.179
    Epoch   0 Batch  163/269 - Train Accuracy:  0.452, Validation Accuracy:  0.484, Loss:  2.190
    Epoch   0 Batch  164/269 - Train Accuracy:  0.481, Validation Accuracy:  0.503, Loss:  2.191
    Epoch   0 Batch  165/269 - Train Accuracy:  0.445, Validation Accuracy:  0.499, Loss:  2.261
    Epoch   0 Batch  166/269 - Train Accuracy:  0.484, Validation Accuracy:  0.489, Loss:  2.048
    Epoch   0 Batch  167/269 - Train Accuracy:  0.480, Validation Accuracy:  0.501, Loss:  2.176
    Epoch   0 Batch  168/269 - Train Accuracy:  0.479, Validation Accuracy:  0.503, Loss:  2.171
    Epoch   0 Batch  169/269 - Train Accuracy:  0.445, Validation Accuracy:  0.485, Loss:  2.164
    Epoch   0 Batch  170/269 - Train Accuracy:  0.476, Validation Accuracy:  0.501, Loss:  2.164
    Epoch   0 Batch  171/269 - Train Accuracy:  0.457, Validation Accuracy:  0.503, Loss:  2.207
    Epoch   0 Batch  172/269 - Train Accuracy:  0.445, Validation Accuracy:  0.477, Loss:  2.163
    Epoch   0 Batch  173/269 - Train Accuracy:  0.475, Validation Accuracy:  0.503, Loss:  2.170
    Epoch   0 Batch  174/269 - Train Accuracy:  0.475, Validation Accuracy:  0.505, Loss:  2.154
    Epoch   0 Batch  175/269 - Train Accuracy:  0.445, Validation Accuracy:  0.481, Loss:  2.152
    Epoch   0 Batch  176/269 - Train Accuracy:  0.456, Validation Accuracy:  0.502, Loss:  2.252
    Epoch   0 Batch  177/269 - Train Accuracy:  0.492, Validation Accuracy:  0.502, Loss:  2.071
    Epoch   0 Batch  178/269 - Train Accuracy:  0.422, Validation Accuracy:  0.480, Loss:  2.222
    Epoch   0 Batch  179/269 - Train Accuracy:  0.479, Validation Accuracy:  0.502, Loss:  2.164
    Epoch   0 Batch  180/269 - Train Accuracy:  0.475, Validation Accuracy:  0.500, Loss:  2.099
    Epoch   0 Batch  181/269 - Train Accuracy:  0.441, Validation Accuracy:  0.474, Loss:  2.118
    Epoch   0 Batch  182/269 - Train Accuracy:  0.460, Validation Accuracy:  0.493, Loss:  2.151
    Epoch   0 Batch  183/269 - Train Accuracy:  0.544, Validation Accuracy:  0.499, Loss:  1.837
    Epoch   0 Batch  184/269 - Train Accuracy:  0.429, Validation Accuracy:  0.484, Loss:  2.230
    Epoch   0 Batch  185/269 - Train Accuracy:  0.464, Validation Accuracy:  0.485, Loss:  2.075
    Epoch   0 Batch  186/269 - Train Accuracy:  0.452, Validation Accuracy:  0.501, Loss:  2.189
    Epoch   0 Batch  187/269 - Train Accuracy:  0.480, Validation Accuracy:  0.498, Loss:  2.065
    Epoch   0 Batch  188/269 - Train Accuracy:  0.459, Validation Accuracy:  0.472, Loss:  2.025
    Epoch   0 Batch  189/269 - Train Accuracy:  0.486, Validation Accuracy:  0.499, Loss:  2.102
    Epoch   0 Batch  190/269 - Train Accuracy:  0.487, Validation Accuracy:  0.502, Loss:  2.062
    Epoch   0 Batch  191/269 - Train Accuracy:  0.454, Validation Accuracy:  0.487, Loss:  2.116
    Epoch   0 Batch  192/269 - Train Accuracy:  0.446, Validation Accuracy:  0.484, Loss:  2.096
    Epoch   0 Batch  193/269 - Train Accuracy:  0.478, Validation Accuracy:  0.504, Loss:  2.088
    Epoch   0 Batch  194/269 - Train Accuracy:  0.483, Validation Accuracy:  0.501, Loss:  2.099
    Epoch   0 Batch  195/269 - Train Accuracy:  0.434, Validation Accuracy:  0.481, Loss:  2.109
    Epoch   0 Batch  196/269 - Train Accuracy:  0.474, Validation Accuracy:  0.503, Loss:  2.059
    Epoch   0 Batch  197/269 - Train Accuracy:  0.449, Validation Accuracy:  0.499, Loss:  2.146
    Epoch   0 Batch  198/269 - Train Accuracy:  0.421, Validation Accuracy:  0.487, Loss:  2.193
    Epoch   0 Batch  199/269 - Train Accuracy:  0.458, Validation Accuracy:  0.497, Loss:  2.101
    Epoch   0 Batch  200/269 - Train Accuracy:  0.454, Validation Accuracy:  0.498, Loss:  2.126
    Epoch   0 Batch  201/269 - Train Accuracy:  0.456, Validation Accuracy:  0.486, Loss:  2.052
    Epoch   0 Batch  202/269 - Train Accuracy:  0.460, Validation Accuracy:  0.496, Loss:  2.076
    Epoch   0 Batch  203/269 - Train Accuracy:  0.466, Validation Accuracy:  0.503, Loss:  2.119
    Epoch   0 Batch  204/269 - Train Accuracy:  0.436, Validation Accuracy:  0.487, Loss:  2.136
    Epoch   0 Batch  205/269 - Train Accuracy:  0.450, Validation Accuracy:  0.494, Loss:  2.053
    Epoch   0 Batch  206/269 - Train Accuracy:  0.450, Validation Accuracy:  0.504, Loss:  2.162
    Epoch   0 Batch  207/269 - Train Accuracy:  0.487, Validation Accuracy:  0.495, Loss:  1.961
    Epoch   0 Batch  208/269 - Train Accuracy:  0.436, Validation Accuracy:  0.489, Loss:  2.157
    Epoch   0 Batch  209/269 - Train Accuracy:  0.457, Validation Accuracy:  0.502, Loss:  2.113
    Epoch   0 Batch  210/269 - Train Accuracy:  0.474, Validation Accuracy:  0.496, Loss:  2.023
    Epoch   0 Batch  211/269 - Train Accuracy:  0.464, Validation Accuracy:  0.491, Loss:  2.013
    Epoch   0 Batch  212/269 - Train Accuracy:  0.493, Validation Accuracy:  0.499, Loss:  1.966
    Epoch   0 Batch  213/269 - Train Accuracy:  0.479, Validation Accuracy:  0.498, Loss:  1.987
    Epoch   0 Batch  214/269 - Train Accuracy:  0.468, Validation Accuracy:  0.490, Loss:  1.994
    Epoch   0 Batch  215/269 - Train Accuracy:  0.507, Validation Accuracy:  0.499, Loss:  1.892
    Epoch   0 Batch  216/269 - Train Accuracy:  0.450, Validation Accuracy:  0.505, Loss:  2.131
    Epoch   0 Batch  217/269 - Train Accuracy:  0.426, Validation Accuracy:  0.484, Loss:  2.105
    Epoch   0 Batch  218/269 - Train Accuracy:  0.453, Validation Accuracy:  0.500, Loss:  2.103
    Epoch   0 Batch  219/269 - Train Accuracy:  0.461, Validation Accuracy:  0.501, Loss:  2.056
    Epoch   0 Batch  220/269 - Train Accuracy:  0.485, Validation Accuracy:  0.496, Loss:  1.914
    Epoch   0 Batch  221/269 - Train Accuracy:  0.478, Validation Accuracy:  0.497, Loss:  1.981
    Epoch   0 Batch  222/269 - Train Accuracy:  0.490, Validation Accuracy:  0.504, Loss:  1.941
    Epoch   0 Batch  223/269 - Train Accuracy:  0.467, Validation Accuracy:  0.493, Loss:  1.946
    Epoch   0 Batch  224/269 - Train Accuracy:  0.474, Validation Accuracy:  0.496, Loss:  1.990
    Epoch   0 Batch  225/269 - Train Accuracy:  0.462, Validation Accuracy:  0.506, Loss:  2.070
    Epoch   0 Batch  226/269 - Train Accuracy:  0.468, Validation Accuracy:  0.495, Loss:  1.962
    Epoch   0 Batch  227/269 - Train Accuracy:  0.543, Validation Accuracy:  0.500, Loss:  1.735
    Epoch   0 Batch  228/269 - Train Accuracy:  0.476, Validation Accuracy:  0.506, Loss:  1.993
    Epoch   0 Batch  229/269 - Train Accuracy:  0.482, Validation Accuracy:  0.499, Loss:  1.964
    Epoch   0 Batch  230/269 - Train Accuracy:  0.468, Validation Accuracy:  0.502, Loss:  1.988
    Epoch   0 Batch  231/269 - Train Accuracy:  0.457, Validation Accuracy:  0.502, Loss:  2.042
    Epoch   0 Batch  232/269 - Train Accuracy:  0.433, Validation Accuracy:  0.497, Loss:  2.065
    Epoch   0 Batch  233/269 - Train Accuracy:  0.487, Validation Accuracy:  0.505, Loss:  1.979
    Epoch   0 Batch  234/269 - Train Accuracy:  0.478, Validation Accuracy:  0.501, Loss:  1.965
    Epoch   0 Batch  235/269 - Train Accuracy:  0.477, Validation Accuracy:  0.501, Loss:  1.965
    Epoch   0 Batch  236/269 - Train Accuracy:  0.477, Validation Accuracy:  0.504, Loss:  1.948
    Epoch   0 Batch  237/269 - Train Accuracy:  0.466, Validation Accuracy:  0.490, Loss:  1.951
    Epoch   0 Batch  238/269 - Train Accuracy:  0.487, Validation Accuracy:  0.507, Loss:  1.974
    Epoch   0 Batch  239/269 - Train Accuracy:  0.485, Validation Accuracy:  0.503, Loss:  1.938
    Epoch   0 Batch  240/269 - Train Accuracy:  0.507, Validation Accuracy:  0.498, Loss:  1.813
    Epoch   0 Batch  241/269 - Train Accuracy:  0.486, Validation Accuracy:  0.502, Loss:  1.923
    Epoch   0 Batch  242/269 - Train Accuracy:  0.469, Validation Accuracy:  0.501, Loss:  1.934
    Epoch   0 Batch  243/269 - Train Accuracy:  0.486, Validation Accuracy:  0.498, Loss:  1.894
    Epoch   0 Batch  244/269 - Train Accuracy:  0.485, Validation Accuracy:  0.500, Loss:  1.925
    Epoch   0 Batch  245/269 - Train Accuracy:  0.435, Validation Accuracy:  0.487, Loss:  2.067
    Epoch   0 Batch  246/269 - Train Accuracy:  0.483, Validation Accuracy:  0.503, Loss:  1.957
    Epoch   0 Batch  247/269 - Train Accuracy:  0.470, Validation Accuracy:  0.504, Loss:  1.991
    Epoch   0 Batch  248/269 - Train Accuracy:  0.461, Validation Accuracy:  0.494, Loss:  1.926
    Epoch   0 Batch  249/269 - Train Accuracy:  0.506, Validation Accuracy:  0.504, Loss:  1.868
    Epoch   0 Batch  250/269 - Train Accuracy:  0.469, Validation Accuracy:  0.503, Loss:  1.980
    Epoch   0 Batch  251/269 - Train Accuracy:  0.472, Validation Accuracy:  0.496, Loss:  1.916
    Epoch   0 Batch  252/269 - Train Accuracy:  0.471, Validation Accuracy:  0.508, Loss:  1.965
    Epoch   0 Batch  253/269 - Train Accuracy:  0.475, Validation Accuracy:  0.500, Loss:  1.923
    Epoch   0 Batch  254/269 - Train Accuracy:  0.483, Validation Accuracy:  0.502, Loss:  1.874
    Epoch   0 Batch  255/269 - Train Accuracy:  0.510, Validation Accuracy:  0.502, Loss:  1.847
    Epoch   0 Batch  256/269 - Train Accuracy:  0.457, Validation Accuracy:  0.503, Loss:  1.950
    Epoch   0 Batch  257/269 - Train Accuracy:  0.468, Validation Accuracy:  0.504, Loss:  1.919
    Epoch   0 Batch  258/269 - Train Accuracy:  0.473, Validation Accuracy:  0.503, Loss:  1.928
    Epoch   0 Batch  259/269 - Train Accuracy:  0.478, Validation Accuracy:  0.498, Loss:  1.890
    Epoch   0 Batch  260/269 - Train Accuracy:  0.469, Validation Accuracy:  0.505, Loss:  1.980
    Epoch   0 Batch  261/269 - Train Accuracy:  0.444, Validation Accuracy:  0.506, Loss:  2.014
    Epoch   0 Batch  262/269 - Train Accuracy:  0.476, Validation Accuracy:  0.502, Loss:  1.906
    Epoch   0 Batch  263/269 - Train Accuracy:  0.462, Validation Accuracy:  0.499, Loss:  1.964
    Epoch   0 Batch  264/269 - Train Accuracy:  0.452, Validation Accuracy:  0.502, Loss:  1.987
    Epoch   0 Batch  265/269 - Train Accuracy:  0.454, Validation Accuracy:  0.502, Loss:  1.939
    Epoch   0 Batch  266/269 - Train Accuracy:  0.484, Validation Accuracy:  0.501, Loss:  1.874
    Epoch   0 Batch  267/269 - Train Accuracy:  0.475, Validation Accuracy:  0.504, Loss:  1.911
    Epoch   1 Batch    0/269 - Train Accuracy:  0.465, Validation Accuracy:  0.503, Loss:  1.963
    Epoch   1 Batch    1/269 - Train Accuracy:  0.447, Validation Accuracy:  0.500, Loss:  1.960
    Epoch   1 Batch    2/269 - Train Accuracy:  0.466, Validation Accuracy:  0.502, Loss:  1.907
    Epoch   1 Batch    3/269 - Train Accuracy:  0.449, Validation Accuracy:  0.499, Loss:  1.949
    Epoch   1 Batch    4/269 - Train Accuracy:  0.445, Validation Accuracy:  0.495, Loss:  1.940
    Epoch   1 Batch    5/269 - Train Accuracy:  0.448, Validation Accuracy:  0.503, Loss:  1.978
    Epoch   1 Batch    6/269 - Train Accuracy:  0.489, Validation Accuracy:  0.504, Loss:  1.802
    Epoch   1 Batch    7/269 - Train Accuracy:  0.485, Validation Accuracy:  0.504, Loss:  1.846
    Epoch   1 Batch    8/269 - Train Accuracy:  0.456, Validation Accuracy:  0.502, Loss:  1.921
    Epoch   1 Batch    9/269 - Train Accuracy:  0.473, Validation Accuracy:  0.501, Loss:  1.864
    Epoch   1 Batch   10/269 - Train Accuracy:  0.454, Validation Accuracy:  0.499, Loss:  1.912
    Epoch   1 Batch   11/269 - Train Accuracy:  0.478, Validation Accuracy:  0.503, Loss:  1.882
    Epoch   1 Batch   12/269 - Train Accuracy:  0.460, Validation Accuracy:  0.505, Loss:  1.934
    Epoch   1 Batch   13/269 - Train Accuracy:  0.517, Validation Accuracy:  0.506, Loss:  1.725
    Epoch   1 Batch   14/269 - Train Accuracy:  0.475, Validation Accuracy:  0.501, Loss:  1.828
    Epoch   1 Batch   15/269 - Train Accuracy:  0.465, Validation Accuracy:  0.502, Loss:  1.849
    Epoch   1 Batch   16/269 - Train Accuracy:  0.485, Validation Accuracy:  0.505, Loss:  1.856
    Epoch   1 Batch   17/269 - Train Accuracy:  0.478, Validation Accuracy:  0.508, Loss:  1.816
    Epoch   1 Batch   18/269 - Train Accuracy:  0.456, Validation Accuracy:  0.504, Loss:  1.908
    Epoch   1 Batch   19/269 - Train Accuracy:  0.515, Validation Accuracy:  0.505, Loss:  1.754
    Epoch   1 Batch   20/269 - Train Accuracy:  0.469, Validation Accuracy:  0.507, Loss:  1.899
    Epoch   1 Batch   21/269 - Train Accuracy:  0.456, Validation Accuracy:  0.503, Loss:  1.939
    Epoch   1 Batch   22/269 - Train Accuracy:  0.491, Validation Accuracy:  0.505, Loss:  1.822
    Epoch   1 Batch   23/269 - Train Accuracy:  0.491, Validation Accuracy:  0.509, Loss:  1.831
    Epoch   1 Batch   24/269 - Train Accuracy:  0.455, Validation Accuracy:  0.508, Loss:  1.889
    Epoch   1 Batch   25/269 - Train Accuracy:  0.450, Validation Accuracy:  0.503, Loss:  1.897
    Epoch   1 Batch   26/269 - Train Accuracy:  0.516, Validation Accuracy:  0.509, Loss:  1.742
    Epoch   1 Batch   27/269 - Train Accuracy:  0.476, Validation Accuracy:  0.506, Loss:  1.828
    Epoch   1 Batch   28/269 - Train Accuracy:  0.434, Validation Accuracy:  0.503, Loss:  1.929
    Epoch   1 Batch   29/269 - Train Accuracy:  0.443, Validation Accuracy:  0.497, Loss:  1.895
    Epoch   1 Batch   30/269 - Train Accuracy:  0.481, Validation Accuracy:  0.502, Loss:  1.859
    Epoch   1 Batch   31/269 - Train Accuracy:  0.489, Validation Accuracy:  0.504, Loss:  1.814
    Epoch   1 Batch   32/269 - Train Accuracy:  0.468, Validation Accuracy:  0.501, Loss:  1.797
    Epoch   1 Batch   33/269 - Train Accuracy:  0.490, Validation Accuracy:  0.501, Loss:  1.761
    Epoch   1 Batch   34/269 - Train Accuracy:  0.480, Validation Accuracy:  0.504, Loss:  1.807
    Epoch   1 Batch   35/269 - Train Accuracy:  0.487, Validation Accuracy:  0.506, Loss:  1.766
    Epoch   1 Batch   36/269 - Train Accuracy:  0.485, Validation Accuracy:  0.504, Loss:  1.804
    Epoch   1 Batch   37/269 - Train Accuracy:  0.482, Validation Accuracy:  0.506, Loss:  1.811
    Epoch   1 Batch   38/269 - Train Accuracy:  0.475, Validation Accuracy:  0.507, Loss:  1.788
    Epoch   1 Batch   39/269 - Train Accuracy:  0.481, Validation Accuracy:  0.509, Loss:  1.781
    Epoch   1 Batch   40/269 - Train Accuracy:  0.453, Validation Accuracy:  0.504, Loss:  1.836
    Epoch   1 Batch   41/269 - Train Accuracy:  0.473, Validation Accuracy:  0.500, Loss:  1.771
    Epoch   1 Batch   42/269 - Train Accuracy:  0.496, Validation Accuracy:  0.501, Loss:  1.711
    Epoch   1 Batch   43/269 - Train Accuracy:  0.458, Validation Accuracy:  0.506, Loss:  1.818
    Epoch   1 Batch   44/269 - Train Accuracy:  0.496, Validation Accuracy:  0.509, Loss:  1.758
    Epoch   1 Batch   45/269 - Train Accuracy:  0.466, Validation Accuracy:  0.504, Loss:  1.823
    Epoch   1 Batch   46/269 - Train Accuracy:  0.452, Validation Accuracy:  0.502, Loss:  1.858
    Epoch   1 Batch   47/269 - Train Accuracy:  0.511, Validation Accuracy:  0.502, Loss:  1.657
    Epoch   1 Batch   48/269 - Train Accuracy:  0.481, Validation Accuracy:  0.503, Loss:  1.695
    Epoch   1 Batch   49/269 - Train Accuracy:  0.459, Validation Accuracy:  0.504, Loss:  1.799
    Epoch   1 Batch   50/269 - Train Accuracy:  0.454, Validation Accuracy:  0.503, Loss:  1.819
    Epoch   1 Batch   51/269 - Train Accuracy:  0.472, Validation Accuracy:  0.504, Loss:  1.775
    Epoch   1 Batch   52/269 - Train Accuracy:  0.473, Validation Accuracy:  0.501, Loss:  1.723
    Epoch   1 Batch   53/269 - Train Accuracy:  0.453, Validation Accuracy:  0.497, Loss:  1.831
    Epoch   1 Batch   54/269 - Train Accuracy:  0.463, Validation Accuracy:  0.502, Loss:  1.834
    Epoch   1 Batch   55/269 - Train Accuracy:  0.484, Validation Accuracy:  0.503, Loss:  1.736
    Epoch   1 Batch   56/269 - Train Accuracy:  0.493, Validation Accuracy:  0.503, Loss:  1.720
    Epoch   1 Batch   57/269 - Train Accuracy:  0.482, Validation Accuracy:  0.502, Loss:  1.730
    Epoch   1 Batch   58/269 - Train Accuracy:  0.491, Validation Accuracy:  0.500, Loss:  1.741
    Epoch   1 Batch   59/269 - Train Accuracy:  0.486, Validation Accuracy:  0.505, Loss:  1.706
    Epoch   1 Batch   60/269 - Train Accuracy:  0.488, Validation Accuracy:  0.506, Loss:  1.675
    Epoch   1 Batch   61/269 - Train Accuracy:  0.507, Validation Accuracy:  0.506, Loss:  1.627
    Epoch   1 Batch   62/269 - Train Accuracy:  0.506, Validation Accuracy:  0.504, Loss:  1.660
    Epoch   1 Batch   63/269 - Train Accuracy:  0.487, Validation Accuracy:  0.510, Loss:  1.698
    Epoch   1 Batch   64/269 - Train Accuracy:  0.483, Validation Accuracy:  0.504, Loss:  1.700
    Epoch   1 Batch   65/269 - Train Accuracy:  0.485, Validation Accuracy:  0.506, Loss:  1.698
    Epoch   1 Batch   66/269 - Train Accuracy:  0.499, Validation Accuracy:  0.505, Loss:  1.656
    Epoch   1 Batch   67/269 - Train Accuracy:  0.476, Validation Accuracy:  0.501, Loss:  1.712
    Epoch   1 Batch   68/269 - Train Accuracy:  0.480, Validation Accuracy:  0.510, Loss:  1.700
    Epoch   1 Batch   69/269 - Train Accuracy:  0.459, Validation Accuracy:  0.507, Loss:  1.819
    Epoch   1 Batch   70/269 - Train Accuracy:  0.501, Validation Accuracy:  0.510, Loss:  1.675
    Epoch   1 Batch   71/269 - Train Accuracy:  0.464, Validation Accuracy:  0.506, Loss:  1.757
    Epoch   1 Batch   72/269 - Train Accuracy:  0.491, Validation Accuracy:  0.498, Loss:  1.620
    Epoch   1 Batch   73/269 - Train Accuracy:  0.495, Validation Accuracy:  0.511, Loss:  1.704
    Epoch   1 Batch   74/269 - Train Accuracy:  0.476, Validation Accuracy:  0.508, Loss:  1.708
    Epoch   1 Batch   75/269 - Train Accuracy:  0.482, Validation Accuracy:  0.503, Loss:  1.655
    Epoch   1 Batch   76/269 - Train Accuracy:  0.472, Validation Accuracy:  0.510, Loss:  1.701
    Epoch   1 Batch   77/269 - Train Accuracy:  0.483, Validation Accuracy:  0.486, Loss:  1.699
    Epoch   1 Batch   78/269 - Train Accuracy:  0.486, Validation Accuracy:  0.511, Loss:  1.695
    Epoch   1 Batch   79/269 - Train Accuracy:  0.485, Validation Accuracy:  0.510, Loss:  1.644
    Epoch   1 Batch   80/269 - Train Accuracy:  0.495, Validation Accuracy:  0.508, Loss:  1.604
    Epoch   1 Batch   81/269 - Train Accuracy:  0.491, Validation Accuracy:  0.510, Loss:  1.678
    Epoch   1 Batch   82/269 - Train Accuracy:  0.502, Validation Accuracy:  0.509, Loss:  1.589
    Epoch   1 Batch   83/269 - Train Accuracy:  0.489, Validation Accuracy:  0.497, Loss:  1.606
    Epoch   1 Batch   84/269 - Train Accuracy:  0.497, Validation Accuracy:  0.511, Loss:  1.616
    Epoch   1 Batch   85/269 - Train Accuracy:  0.482, Validation Accuracy:  0.506, Loss:  1.638
    Epoch   1 Batch   86/269 - Train Accuracy:  0.481, Validation Accuracy:  0.495, Loss:  1.629
    Epoch   1 Batch   87/269 - Train Accuracy:  0.451, Validation Accuracy:  0.505, Loss:  1.734
    Epoch   1 Batch   88/269 - Train Accuracy:  0.484, Validation Accuracy:  0.501, Loss:  1.619
    Epoch   1 Batch   89/269 - Train Accuracy:  0.501, Validation Accuracy:  0.509, Loss:  1.604
    Epoch   1 Batch   90/269 - Train Accuracy:  0.451, Validation Accuracy:  0.507, Loss:  1.688
    Epoch   1 Batch   91/269 - Train Accuracy:  0.485, Validation Accuracy:  0.501, Loss:  1.575
    Epoch   1 Batch   92/269 - Train Accuracy:  0.477, Validation Accuracy:  0.503, Loss:  1.571
    Epoch   1 Batch   93/269 - Train Accuracy:  0.498, Validation Accuracy:  0.499, Loss:  1.534
    Epoch   1 Batch   94/269 - Train Accuracy:  0.494, Validation Accuracy:  0.508, Loss:  1.597
    Epoch   1 Batch   95/269 - Train Accuracy:  0.496, Validation Accuracy:  0.502, Loss:  1.572
    Epoch   1 Batch   96/269 - Train Accuracy:  0.483, Validation Accuracy:  0.498, Loss:  1.555
    Epoch   1 Batch   97/269 - Train Accuracy:  0.488, Validation Accuracy:  0.511, Loss:  1.582
    Epoch   1 Batch   98/269 - Train Accuracy:  0.508, Validation Accuracy:  0.507, Loss:  1.545
    Epoch   1 Batch   99/269 - Train Accuracy:  0.470, Validation Accuracy:  0.507, Loss:  1.648
    Epoch   1 Batch  100/269 - Train Accuracy:  0.516, Validation Accuracy:  0.515, Loss:  1.514
    Epoch   1 Batch  101/269 - Train Accuracy:  0.464, Validation Accuracy:  0.509, Loss:  1.648
    Epoch   1 Batch  102/269 - Train Accuracy:  0.481, Validation Accuracy:  0.501, Loss:  1.523
    Epoch   1 Batch  103/269 - Train Accuracy:  0.486, Validation Accuracy:  0.507, Loss:  1.534
    Epoch   1 Batch  104/269 - Train Accuracy:  0.480, Validation Accuracy:  0.499, Loss:  1.558
    Epoch   1 Batch  105/269 - Train Accuracy:  0.479, Validation Accuracy:  0.507, Loss:  1.574
    Epoch   1 Batch  106/269 - Train Accuracy:  0.474, Validation Accuracy:  0.502, Loss:  1.564
    Epoch   1 Batch  107/269 - Train Accuracy:  0.443, Validation Accuracy:  0.503, Loss:  1.615
    Epoch   1 Batch  108/269 - Train Accuracy:  0.482, Validation Accuracy:  0.506, Loss:  1.511
    Epoch   1 Batch  109/269 - Train Accuracy:  0.475, Validation Accuracy:  0.506, Loss:  1.542
    Epoch   1 Batch  110/269 - Train Accuracy:  0.492, Validation Accuracy:  0.505, Loss:  1.514
    Epoch   1 Batch  111/269 - Train Accuracy:  0.465, Validation Accuracy:  0.508, Loss:  1.623
    Epoch   1 Batch  112/269 - Train Accuracy:  0.482, Validation Accuracy:  0.503, Loss:  1.496
    Epoch   1 Batch  113/269 - Train Accuracy:  0.505, Validation Accuracy:  0.501, Loss:  1.435
    Epoch   1 Batch  114/269 - Train Accuracy:  0.474, Validation Accuracy:  0.497, Loss:  1.497
    Epoch   1 Batch  115/269 - Train Accuracy:  0.454, Validation Accuracy:  0.492, Loss:  1.536
    Epoch   1 Batch  116/269 - Train Accuracy:  0.482, Validation Accuracy:  0.495, Loss:  1.502
    Epoch   1 Batch  117/269 - Train Accuracy:  0.471, Validation Accuracy:  0.505, Loss:  1.491
    Epoch   1 Batch  118/269 - Train Accuracy:  0.490, Validation Accuracy:  0.498, Loss:  1.438
    Epoch   1 Batch  119/269 - Train Accuracy:  0.468, Validation Accuracy:  0.503, Loss:  1.565
    Epoch   1 Batch  120/269 - Train Accuracy:  0.460, Validation Accuracy:  0.502, Loss:  1.545
    Epoch   1 Batch  121/269 - Train Accuracy:  0.472, Validation Accuracy:  0.498, Loss:  1.483
    Epoch   1 Batch  122/269 - Train Accuracy:  0.492, Validation Accuracy:  0.501, Loss:  1.460
    Epoch   1 Batch  123/269 - Train Accuracy:  0.451, Validation Accuracy:  0.496, Loss:  1.554
    Epoch   1 Batch  124/269 - Train Accuracy:  0.482, Validation Accuracy:  0.508, Loss:  1.456
    Epoch   1 Batch  125/269 - Train Accuracy:  0.466, Validation Accuracy:  0.494, Loss:  1.438
    Epoch   1 Batch  126/269 - Train Accuracy:  0.482, Validation Accuracy:  0.499, Loss:  1.450
    Epoch   1 Batch  127/269 - Train Accuracy:  0.456, Validation Accuracy:  0.501, Loss:  1.539
    Epoch   1 Batch  128/269 - Train Accuracy:  0.481, Validation Accuracy:  0.493, Loss:  1.453
    Epoch   1 Batch  129/269 - Train Accuracy:  0.478, Validation Accuracy:  0.504, Loss:  1.485
    Epoch   1 Batch  130/269 - Train Accuracy:  0.438, Validation Accuracy:  0.490, Loss:  1.575
    Epoch   1 Batch  131/269 - Train Accuracy:  0.456, Validation Accuracy:  0.503, Loss:  1.521
    Epoch   1 Batch  132/269 - Train Accuracy:  0.474, Validation Accuracy:  0.507, Loss:  1.470
    Epoch   1 Batch  133/269 - Train Accuracy:  0.480, Validation Accuracy:  0.507, Loss:  1.431
    Epoch   1 Batch  134/269 - Train Accuracy:  0.438, Validation Accuracy:  0.502, Loss:  1.525
    Epoch   1 Batch  135/269 - Train Accuracy:  0.440, Validation Accuracy:  0.509, Loss:  1.599
    Epoch   1 Batch  136/269 - Train Accuracy:  0.418, Validation Accuracy:  0.483, Loss:  1.609
    Epoch   1 Batch  137/269 - Train Accuracy:  0.442, Validation Accuracy:  0.487, Loss:  1.679
    Epoch   1 Batch  138/269 - Train Accuracy:  0.459, Validation Accuracy:  0.508, Loss:  1.592
    Epoch   1 Batch  139/269 - Train Accuracy:  0.467, Validation Accuracy:  0.495, Loss:  1.428
    Epoch   1 Batch  140/269 - Train Accuracy:  0.487, Validation Accuracy:  0.507, Loss:  1.494
    Epoch   1 Batch  141/269 - Train Accuracy:  0.467, Validation Accuracy:  0.512, Loss:  1.543
    Epoch   1 Batch  142/269 - Train Accuracy:  0.481, Validation Accuracy:  0.493, Loss:  1.414
    Epoch   1 Batch  143/269 - Train Accuracy:  0.476, Validation Accuracy:  0.502, Loss:  1.501
    Epoch   1 Batch  144/269 - Train Accuracy:  0.494, Validation Accuracy:  0.513, Loss:  1.457
    Epoch   1 Batch  145/269 - Train Accuracy:  0.451, Validation Accuracy:  0.491, Loss:  1.448
    Epoch   1 Batch  146/269 - Train Accuracy:  0.484, Validation Accuracy:  0.503, Loss:  1.443
    Epoch   1 Batch  147/269 - Train Accuracy:  0.500, Validation Accuracy:  0.509, Loss:  1.377
    Epoch   1 Batch  148/269 - Train Accuracy:  0.477, Validation Accuracy:  0.514, Loss:  1.492
    Epoch   1 Batch  149/269 - Train Accuracy:  0.477, Validation Accuracy:  0.494, Loss:  1.417
    Epoch   1 Batch  150/269 - Train Accuracy:  0.475, Validation Accuracy:  0.512, Loss:  1.466
    Epoch   1 Batch  151/269 - Train Accuracy:  0.516, Validation Accuracy:  0.511, Loss:  1.346
    Epoch   1 Batch  152/269 - Train Accuracy:  0.479, Validation Accuracy:  0.503, Loss:  1.460
    Epoch   1 Batch  153/269 - Train Accuracy:  0.493, Validation Accuracy:  0.507, Loss:  1.402
    Epoch   1 Batch  154/269 - Train Accuracy:  0.457, Validation Accuracy:  0.511, Loss:  1.509
    Epoch   1 Batch  155/269 - Train Accuracy:  0.511, Validation Accuracy:  0.512, Loss:  1.348
    Epoch   1 Batch  156/269 - Train Accuracy:  0.453, Validation Accuracy:  0.499, Loss:  1.475
    Epoch   1 Batch  157/269 - Train Accuracy:  0.479, Validation Accuracy:  0.506, Loss:  1.427
    Epoch   1 Batch  158/269 - Train Accuracy:  0.491, Validation Accuracy:  0.517, Loss:  1.407
    Epoch   1 Batch  159/269 - Train Accuracy:  0.480, Validation Accuracy:  0.518, Loss:  1.403
    Epoch   1 Batch  160/269 - Train Accuracy:  0.474, Validation Accuracy:  0.498, Loss:  1.429
    Epoch   1 Batch  161/269 - Train Accuracy:  0.462, Validation Accuracy:  0.501, Loss:  1.417
    Epoch   1 Batch  162/269 - Train Accuracy:  0.481, Validation Accuracy:  0.507, Loss:  1.400
    Epoch   1 Batch  163/269 - Train Accuracy:  0.498, Validation Accuracy:  0.519, Loss:  1.393
    Epoch   1 Batch  164/269 - Train Accuracy:  0.463, Validation Accuracy:  0.506, Loss:  1.393
    Epoch   1 Batch  165/269 - Train Accuracy:  0.440, Validation Accuracy:  0.500, Loss:  1.431
    Epoch   1 Batch  166/269 - Train Accuracy:  0.512, Validation Accuracy:  0.515, Loss:  1.320
    Epoch   1 Batch  167/269 - Train Accuracy:  0.499, Validation Accuracy:  0.519, Loss:  1.377
    Epoch   1 Batch  168/269 - Train Accuracy:  0.476, Validation Accuracy:  0.508, Loss:  1.402
    Epoch   1 Batch  169/269 - Train Accuracy:  0.466, Validation Accuracy:  0.503, Loss:  1.380
    Epoch   1 Batch  170/269 - Train Accuracy:  0.485, Validation Accuracy:  0.511, Loss:  1.371
    Epoch   1 Batch  171/269 - Train Accuracy:  0.473, Validation Accuracy:  0.516, Loss:  1.409
    Epoch   1 Batch  172/269 - Train Accuracy:  0.485, Validation Accuracy:  0.514, Loss:  1.389
    Epoch   1 Batch  173/269 - Train Accuracy:  0.480, Validation Accuracy:  0.504, Loss:  1.377
    Epoch   1 Batch  174/269 - Train Accuracy:  0.469, Validation Accuracy:  0.503, Loss:  1.380
    Epoch   1 Batch  175/269 - Train Accuracy:  0.485, Validation Accuracy:  0.517, Loss:  1.384
    Epoch   1 Batch  176/269 - Train Accuracy:  0.463, Validation Accuracy:  0.518, Loss:  1.426
    Epoch   1 Batch  177/269 - Train Accuracy:  0.496, Validation Accuracy:  0.514, Loss:  1.336
    Epoch   1 Batch  178/269 - Train Accuracy:  0.472, Validation Accuracy:  0.517, Loss:  1.417
    Epoch   1 Batch  179/269 - Train Accuracy:  0.492, Validation Accuracy:  0.520, Loss:  1.342
    Epoch   1 Batch  180/269 - Train Accuracy:  0.480, Validation Accuracy:  0.519, Loss:  1.336
    Epoch   1 Batch  181/269 - Train Accuracy:  0.483, Validation Accuracy:  0.512, Loss:  1.343
    Epoch   1 Batch  182/269 - Train Accuracy:  0.483, Validation Accuracy:  0.510, Loss:  1.368
    Epoch   1 Batch  183/269 - Train Accuracy:  0.554, Validation Accuracy:  0.522, Loss:  1.165
    Epoch   1 Batch  184/269 - Train Accuracy:  0.466, Validation Accuracy:  0.516, Loss:  1.395
    Epoch   1 Batch  185/269 - Train Accuracy:  0.483, Validation Accuracy:  0.507, Loss:  1.333
    Epoch   1 Batch  186/269 - Train Accuracy:  0.456, Validation Accuracy:  0.514, Loss:  1.401
    Epoch   1 Batch  187/269 - Train Accuracy:  0.497, Validation Accuracy:  0.521, Loss:  1.309
    Epoch   1 Batch  188/269 - Train Accuracy:  0.493, Validation Accuracy:  0.513, Loss:  1.310
    Epoch   1 Batch  189/269 - Train Accuracy:  0.492, Validation Accuracy:  0.504, Loss:  1.323
    Epoch   1 Batch  190/269 - Train Accuracy:  0.486, Validation Accuracy:  0.519, Loss:  1.320
    Epoch   1 Batch  191/269 - Train Accuracy:  0.491, Validation Accuracy:  0.521, Loss:  1.343
    Epoch   1 Batch  192/269 - Train Accuracy:  0.480, Validation Accuracy:  0.511, Loss:  1.340
    Epoch   1 Batch  193/269 - Train Accuracy:  0.467, Validation Accuracy:  0.502, Loss:  1.323
    Epoch   1 Batch  194/269 - Train Accuracy:  0.479, Validation Accuracy:  0.506, Loss:  1.327
    Epoch   1 Batch  195/269 - Train Accuracy:  0.468, Validation Accuracy:  0.516, Loss:  1.346
    Epoch   1 Batch  196/269 - Train Accuracy:  0.479, Validation Accuracy:  0.516, Loss:  1.311
    Epoch   1 Batch  197/269 - Train Accuracy:  0.454, Validation Accuracy:  0.514, Loss:  1.377
    Epoch   1 Batch  198/269 - Train Accuracy:  0.468, Validation Accuracy:  0.522, Loss:  1.419
    Epoch   1 Batch  199/269 - Train Accuracy:  0.482, Validation Accuracy:  0.516, Loss:  1.356
    Epoch   1 Batch  200/269 - Train Accuracy:  0.475, Validation Accuracy:  0.522, Loss:  1.373
    Epoch   1 Batch  201/269 - Train Accuracy:  0.479, Validation Accuracy:  0.502, Loss:  1.310
    Epoch   1 Batch  202/269 - Train Accuracy:  0.472, Validation Accuracy:  0.513, Loss:  1.336
    Epoch   1 Batch  203/269 - Train Accuracy:  0.471, Validation Accuracy:  0.521, Loss:  1.379
    Epoch   1 Batch  204/269 - Train Accuracy:  0.460, Validation Accuracy:  0.518, Loss:  1.356
    Epoch   1 Batch  205/269 - Train Accuracy:  0.465, Validation Accuracy:  0.508, Loss:  1.302
    Epoch   1 Batch  206/269 - Train Accuracy:  0.461, Validation Accuracy:  0.525, Loss:  1.403
    Epoch   1 Batch  207/269 - Train Accuracy:  0.507, Validation Accuracy:  0.525, Loss:  1.255
    Epoch   1 Batch  208/269 - Train Accuracy:  0.460, Validation Accuracy:  0.517, Loss:  1.397
    Epoch   1 Batch  209/269 - Train Accuracy:  0.457, Validation Accuracy:  0.515, Loss:  1.343
    Epoch   1 Batch  210/269 - Train Accuracy:  0.484, Validation Accuracy:  0.519, Loss:  1.292
    Epoch   1 Batch  211/269 - Train Accuracy:  0.480, Validation Accuracy:  0.519, Loss:  1.291
    Epoch   1 Batch  212/269 - Train Accuracy:  0.488, Validation Accuracy:  0.506, Loss:  1.273
    Epoch   1 Batch  213/269 - Train Accuracy:  0.479, Validation Accuracy:  0.505, Loss:  1.272
    Epoch   1 Batch  214/269 - Train Accuracy:  0.486, Validation Accuracy:  0.511, Loss:  1.288
    Epoch   1 Batch  215/269 - Train Accuracy:  0.509, Validation Accuracy:  0.509, Loss:  1.220
    Epoch   1 Batch  216/269 - Train Accuracy:  0.458, Validation Accuracy:  0.508, Loss:  1.369
    Epoch   1 Batch  217/269 - Train Accuracy:  0.448, Validation Accuracy:  0.508, Loss:  1.350
    Epoch   1 Batch  218/269 - Train Accuracy:  0.469, Validation Accuracy:  0.509, Loss:  1.341
    Epoch   1 Batch  219/269 - Train Accuracy:  0.469, Validation Accuracy:  0.515, Loss:  1.324
    Epoch   1 Batch  220/269 - Train Accuracy:  0.495, Validation Accuracy:  0.511, Loss:  1.228
    Epoch   1 Batch  221/269 - Train Accuracy:  0.482, Validation Accuracy:  0.504, Loss:  1.278
    Epoch   1 Batch  222/269 - Train Accuracy:  0.497, Validation Accuracy:  0.508, Loss:  1.235
    Epoch   1 Batch  223/269 - Train Accuracy:  0.495, Validation Accuracy:  0.514, Loss:  1.244
    Epoch   1 Batch  224/269 - Train Accuracy:  0.491, Validation Accuracy:  0.513, Loss:  1.295
    Epoch   1 Batch  225/269 - Train Accuracy:  0.457, Validation Accuracy:  0.513, Loss:  1.323
    Epoch   1 Batch  226/269 - Train Accuracy:  0.479, Validation Accuracy:  0.511, Loss:  1.273
    Epoch   1 Batch  227/269 - Train Accuracy:  0.546, Validation Accuracy:  0.505, Loss:  1.117
    Epoch   1 Batch  228/269 - Train Accuracy:  0.478, Validation Accuracy:  0.504, Loss:  1.279
    Epoch   1 Batch  229/269 - Train Accuracy:  0.477, Validation Accuracy:  0.508, Loss:  1.275
    Epoch   1 Batch  230/269 - Train Accuracy:  0.472, Validation Accuracy:  0.515, Loss:  1.281
    Epoch   1 Batch  231/269 - Train Accuracy:  0.441, Validation Accuracy:  0.513, Loss:  1.337
    Epoch   1 Batch  232/269 - Train Accuracy:  0.436, Validation Accuracy:  0.517, Loss:  1.318
    Epoch   1 Batch  233/269 - Train Accuracy:  0.489, Validation Accuracy:  0.514, Loss:  1.268
    Epoch   1 Batch  234/269 - Train Accuracy:  0.473, Validation Accuracy:  0.513, Loss:  1.263
    Epoch   1 Batch  235/269 - Train Accuracy:  0.480, Validation Accuracy:  0.504, Loss:  1.258
    Epoch   1 Batch  236/269 - Train Accuracy:  0.480, Validation Accuracy:  0.503, Loss:  1.258
    Epoch   1 Batch  237/269 - Train Accuracy:  0.481, Validation Accuracy:  0.504, Loss:  1.252
    Epoch   1 Batch  238/269 - Train Accuracy:  0.492, Validation Accuracy:  0.506, Loss:  1.262
    Epoch   1 Batch  239/269 - Train Accuracy:  0.485, Validation Accuracy:  0.511, Loss:  1.243
    Epoch   1 Batch  240/269 - Train Accuracy:  0.525, Validation Accuracy:  0.517, Loss:  1.159
    Epoch   1 Batch  241/269 - Train Accuracy:  0.499, Validation Accuracy:  0.517, Loss:  1.252
    Epoch   1 Batch  242/269 - Train Accuracy:  0.481, Validation Accuracy:  0.517, Loss:  1.243
    Epoch   1 Batch  243/269 - Train Accuracy:  0.502, Validation Accuracy:  0.509, Loss:  1.212
    Epoch   1 Batch  244/269 - Train Accuracy:  0.472, Validation Accuracy:  0.510, Loss:  1.227
    Epoch   1 Batch  245/269 - Train Accuracy:  0.462, Validation Accuracy:  0.508, Loss:  1.301
    Epoch   1 Batch  246/269 - Train Accuracy:  0.470, Validation Accuracy:  0.507, Loss:  1.242
    Epoch   1 Batch  247/269 - Train Accuracy:  0.476, Validation Accuracy:  0.513, Loss:  1.272
    Epoch   1 Batch  248/269 - Train Accuracy:  0.486, Validation Accuracy:  0.516, Loss:  1.228
    Epoch   1 Batch  249/269 - Train Accuracy:  0.514, Validation Accuracy:  0.517, Loss:  1.182
    Epoch   1 Batch  250/269 - Train Accuracy:  0.479, Validation Accuracy:  0.520, Loss:  1.265
    Epoch   1 Batch  251/269 - Train Accuracy:  0.500, Validation Accuracy:  0.519, Loss:  1.214
    Epoch   1 Batch  252/269 - Train Accuracy:  0.479, Validation Accuracy:  0.517, Loss:  1.244
    Epoch   1 Batch  253/269 - Train Accuracy:  0.477, Validation Accuracy:  0.509, Loss:  1.232
    Epoch   1 Batch  254/269 - Train Accuracy:  0.467, Validation Accuracy:  0.510, Loss:  1.213
    Epoch   1 Batch  255/269 - Train Accuracy:  0.516, Validation Accuracy:  0.513, Loss:  1.180
    Epoch   1 Batch  256/269 - Train Accuracy:  0.467, Validation Accuracy:  0.509, Loss:  1.251
    Epoch   1 Batch  257/269 - Train Accuracy:  0.471, Validation Accuracy:  0.502, Loss:  1.238
    Epoch   1 Batch  258/269 - Train Accuracy:  0.474, Validation Accuracy:  0.499, Loss:  1.237
    Epoch   1 Batch  259/269 - Train Accuracy:  0.505, Validation Accuracy:  0.512, Loss:  1.228
    Epoch   1 Batch  260/269 - Train Accuracy:  0.469, Validation Accuracy:  0.512, Loss:  1.272
    Epoch   1 Batch  261/269 - Train Accuracy:  0.438, Validation Accuracy:  0.505, Loss:  1.285
    Epoch   1 Batch  262/269 - Train Accuracy:  0.492, Validation Accuracy:  0.505, Loss:  1.228
    Epoch   1 Batch  263/269 - Train Accuracy:  0.471, Validation Accuracy:  0.503, Loss:  1.258
    Epoch   1 Batch  264/269 - Train Accuracy:  0.463, Validation Accuracy:  0.504, Loss:  1.290
    Epoch   1 Batch  265/269 - Train Accuracy:  0.453, Validation Accuracy:  0.504, Loss:  1.241
    Epoch   1 Batch  266/269 - Train Accuracy:  0.484, Validation Accuracy:  0.507, Loss:  1.189
    Epoch   1 Batch  267/269 - Train Accuracy:  0.486, Validation Accuracy:  0.510, Loss:  1.223
    Epoch   2 Batch    0/269 - Train Accuracy:  0.467, Validation Accuracy:  0.513, Loss:  1.276
    Epoch   2 Batch    1/269 - Train Accuracy:  0.450, Validation Accuracy:  0.508, Loss:  1.245
    Epoch   2 Batch    2/269 - Train Accuracy:  0.465, Validation Accuracy:  0.508, Loss:  1.233
    Epoch   2 Batch    3/269 - Train Accuracy:  0.467, Validation Accuracy:  0.506, Loss:  1.248
    Epoch   2 Batch    4/269 - Train Accuracy:  0.451, Validation Accuracy:  0.506, Loss:  1.239
    Epoch   2 Batch    5/269 - Train Accuracy:  0.436, Validation Accuracy:  0.502, Loss:  1.240
    Epoch   2 Batch    6/269 - Train Accuracy:  0.501, Validation Accuracy:  0.501, Loss:  1.140
    Epoch   2 Batch    7/269 - Train Accuracy:  0.484, Validation Accuracy:  0.509, Loss:  1.190
    Epoch   2 Batch    8/269 - Train Accuracy:  0.460, Validation Accuracy:  0.510, Loss:  1.244
    Epoch   2 Batch    9/269 - Train Accuracy:  0.476, Validation Accuracy:  0.511, Loss:  1.204
    Epoch   2 Batch   10/269 - Train Accuracy:  0.477, Validation Accuracy:  0.508, Loss:  1.213
    Epoch   2 Batch   11/269 - Train Accuracy:  0.462, Validation Accuracy:  0.509, Loss:  1.197
    Epoch   2 Batch   12/269 - Train Accuracy:  0.453, Validation Accuracy:  0.508, Loss:  1.242
    Epoch   2 Batch   13/269 - Train Accuracy:  0.510, Validation Accuracy:  0.504, Loss:  1.104
    Epoch   2 Batch   14/269 - Train Accuracy:  0.470, Validation Accuracy:  0.504, Loss:  1.173
    Epoch   2 Batch   15/269 - Train Accuracy:  0.466, Validation Accuracy:  0.508, Loss:  1.179
    Epoch   2 Batch   16/269 - Train Accuracy:  0.487, Validation Accuracy:  0.500, Loss:  1.185
    Epoch   2 Batch   17/269 - Train Accuracy:  0.479, Validation Accuracy:  0.504, Loss:  1.155
    Epoch   2 Batch   18/269 - Train Accuracy:  0.448, Validation Accuracy:  0.505, Loss:  1.217
    Epoch   2 Batch   19/269 - Train Accuracy:  0.516, Validation Accuracy:  0.503, Loss:  1.110
    Epoch   2 Batch   20/269 - Train Accuracy:  0.454, Validation Accuracy:  0.498, Loss:  1.219
    Epoch   2 Batch   21/269 - Train Accuracy:  0.454, Validation Accuracy:  0.499, Loss:  1.254
    Epoch   2 Batch   22/269 - Train Accuracy:  0.490, Validation Accuracy:  0.502, Loss:  1.156
    Epoch   2 Batch   23/269 - Train Accuracy:  0.486, Validation Accuracy:  0.508, Loss:  1.176
    Epoch   2 Batch   24/269 - Train Accuracy:  0.447, Validation Accuracy:  0.513, Loss:  1.212
    Epoch   2 Batch   25/269 - Train Accuracy:  0.459, Validation Accuracy:  0.512, Loss:  1.211
    Epoch   2 Batch   26/269 - Train Accuracy:  0.519, Validation Accuracy:  0.513, Loss:  1.086
    Epoch   2 Batch   27/269 - Train Accuracy:  0.489, Validation Accuracy:  0.512, Loss:  1.160
    Epoch   2 Batch   28/269 - Train Accuracy:  0.452, Validation Accuracy:  0.506, Loss:  1.231
    Epoch   2 Batch   29/269 - Train Accuracy:  0.471, Validation Accuracy:  0.503, Loss:  1.188
    Epoch   2 Batch   30/269 - Train Accuracy:  0.483, Validation Accuracy:  0.498, Loss:  1.149
    Epoch   2 Batch   31/269 - Train Accuracy:  0.494, Validation Accuracy:  0.499, Loss:  1.140
    Epoch   2 Batch   32/269 - Train Accuracy:  0.470, Validation Accuracy:  0.495, Loss:  1.149
    Epoch   2 Batch   33/269 - Train Accuracy:  0.492, Validation Accuracy:  0.486, Loss:  1.112
    Epoch   2 Batch   34/269 - Train Accuracy:  0.477, Validation Accuracy:  0.494, Loss:  1.149
    Epoch   2 Batch   35/269 - Train Accuracy:  0.483, Validation Accuracy:  0.499, Loss:  1.144
    Epoch   2 Batch   36/269 - Train Accuracy:  0.472, Validation Accuracy:  0.487, Loss:  1.163
    Epoch   2 Batch   37/269 - Train Accuracy:  0.490, Validation Accuracy:  0.495, Loss:  1.157
    Epoch   2 Batch   38/269 - Train Accuracy:  0.472, Validation Accuracy:  0.498, Loss:  1.146
    Epoch   2 Batch   39/269 - Train Accuracy:  0.476, Validation Accuracy:  0.502, Loss:  1.140
    Epoch   2 Batch   40/269 - Train Accuracy:  0.452, Validation Accuracy:  0.506, Loss:  1.186
    Epoch   2 Batch   41/269 - Train Accuracy:  0.467, Validation Accuracy:  0.506, Loss:  1.150
    Epoch   2 Batch   42/269 - Train Accuracy:  0.508, Validation Accuracy:  0.509, Loss:  1.077
    Epoch   2 Batch   43/269 - Train Accuracy:  0.483, Validation Accuracy:  0.512, Loss:  1.174
    Epoch   2 Batch   44/269 - Train Accuracy:  0.502, Validation Accuracy:  0.515, Loss:  1.139
    Epoch   2 Batch   45/269 - Train Accuracy:  0.462, Validation Accuracy:  0.513, Loss:  1.185
    Epoch   2 Batch   46/269 - Train Accuracy:  0.457, Validation Accuracy:  0.511, Loss:  1.177
    Epoch   2 Batch   47/269 - Train Accuracy:  0.502, Validation Accuracy:  0.508, Loss:  1.054
    Epoch   2 Batch   48/269 - Train Accuracy:  0.493, Validation Accuracy:  0.510, Loss:  1.102
    Epoch   2 Batch   49/269 - Train Accuracy:  0.459, Validation Accuracy:  0.507, Loss:  1.158
    Epoch   2 Batch   50/269 - Train Accuracy:  0.460, Validation Accuracy:  0.502, Loss:  1.177
    Epoch   2 Batch   51/269 - Train Accuracy:  0.483, Validation Accuracy:  0.504, Loss:  1.143
    Epoch   2 Batch   52/269 - Train Accuracy:  0.473, Validation Accuracy:  0.500, Loss:  1.096
    Epoch   2 Batch   53/269 - Train Accuracy:  0.450, Validation Accuracy:  0.492, Loss:  1.189
    Epoch   2 Batch   54/269 - Train Accuracy:  0.461, Validation Accuracy:  0.500, Loss:  1.163
    Epoch   2 Batch   55/269 - Train Accuracy:  0.478, Validation Accuracy:  0.502, Loss:  1.112
    Epoch   2 Batch   56/269 - Train Accuracy:  0.488, Validation Accuracy:  0.502, Loss:  1.119
    Epoch   2 Batch   57/269 - Train Accuracy:  0.467, Validation Accuracy:  0.490, Loss:  1.129
    Epoch   2 Batch   58/269 - Train Accuracy:  0.486, Validation Accuracy:  0.504, Loss:  1.109
    Epoch   2 Batch   59/269 - Train Accuracy:  0.493, Validation Accuracy:  0.507, Loss:  1.086
    Epoch   2 Batch   60/269 - Train Accuracy:  0.495, Validation Accuracy:  0.484, Loss:  1.072
    Epoch   2 Batch   61/269 - Train Accuracy:  0.518, Validation Accuracy:  0.511, Loss:  1.048
    Epoch   2 Batch   62/269 - Train Accuracy:  0.511, Validation Accuracy:  0.510, Loss:  1.077
    Epoch   2 Batch   63/269 - Train Accuracy:  0.457, Validation Accuracy:  0.495, Loss:  1.123
    Epoch   2 Batch   64/269 - Train Accuracy:  0.470, Validation Accuracy:  0.497, Loss:  1.098
    Epoch   2 Batch   65/269 - Train Accuracy:  0.481, Validation Accuracy:  0.497, Loss:  1.087
    Epoch   2 Batch   66/269 - Train Accuracy:  0.506, Validation Accuracy:  0.495, Loss:  1.067
    Epoch   2 Batch   67/269 - Train Accuracy:  0.468, Validation Accuracy:  0.489, Loss:  1.121
    Epoch   2 Batch   68/269 - Train Accuracy:  0.481, Validation Accuracy:  0.492, Loss:  1.116
    Epoch   2 Batch   69/269 - Train Accuracy:  0.440, Validation Accuracy:  0.494, Loss:  1.201
    Epoch   2 Batch   70/269 - Train Accuracy:  0.494, Validation Accuracy:  0.494, Loss:  1.092
    Epoch   2 Batch   71/269 - Train Accuracy:  0.468, Validation Accuracy:  0.494, Loss:  1.138
    Epoch   2 Batch   72/269 - Train Accuracy:  0.498, Validation Accuracy:  0.501, Loss:  1.065
    Epoch   2 Batch   73/269 - Train Accuracy:  0.491, Validation Accuracy:  0.501, Loss:  1.123
    Epoch   2 Batch   74/269 - Train Accuracy:  0.453, Validation Accuracy:  0.501, Loss:  1.114
    Epoch   2 Batch   75/269 - Train Accuracy:  0.469, Validation Accuracy:  0.502, Loss:  1.091
    Epoch   2 Batch   76/269 - Train Accuracy:  0.451, Validation Accuracy:  0.506, Loss:  1.111
    Epoch   2 Batch   77/269 - Train Accuracy:  0.483, Validation Accuracy:  0.507, Loss:  1.082
    Epoch   2 Batch   78/269 - Train Accuracy:  0.474, Validation Accuracy:  0.503, Loss:  1.086
    Epoch   2 Batch   79/269 - Train Accuracy:  0.477, Validation Accuracy:  0.500, Loss:  1.076
    Epoch   2 Batch   80/269 - Train Accuracy:  0.490, Validation Accuracy:  0.504, Loss:  1.060
    Epoch   2 Batch   81/269 - Train Accuracy:  0.475, Validation Accuracy:  0.498, Loss:  1.098
    Epoch   2 Batch   82/269 - Train Accuracy:  0.488, Validation Accuracy:  0.494, Loss:  1.046
    Epoch   2 Batch   83/269 - Train Accuracy:  0.485, Validation Accuracy:  0.493, Loss:  1.070
    Epoch   2 Batch   84/269 - Train Accuracy:  0.491, Validation Accuracy:  0.507, Loss:  1.061
    Epoch   2 Batch   85/269 - Train Accuracy:  0.467, Validation Accuracy:  0.497, Loss:  1.080
    Epoch   2 Batch   86/269 - Train Accuracy:  0.469, Validation Accuracy:  0.503, Loss:  1.083
    Epoch   2 Batch   87/269 - Train Accuracy:  0.449, Validation Accuracy:  0.507, Loss:  1.149
    Epoch   2 Batch   88/269 - Train Accuracy:  0.496, Validation Accuracy:  0.510, Loss:  1.075
    Epoch   2 Batch   89/269 - Train Accuracy:  0.487, Validation Accuracy:  0.503, Loss:  1.062
    Epoch   2 Batch   90/269 - Train Accuracy:  0.445, Validation Accuracy:  0.502, Loss:  1.149
    Epoch   2 Batch   91/269 - Train Accuracy:  0.481, Validation Accuracy:  0.506, Loss:  1.054
    Epoch   2 Batch   92/269 - Train Accuracy:  0.483, Validation Accuracy:  0.501, Loss:  1.052
    Epoch   2 Batch   93/269 - Train Accuracy:  0.511, Validation Accuracy:  0.499, Loss:  1.033
    Epoch   2 Batch   94/269 - Train Accuracy:  0.479, Validation Accuracy:  0.499, Loss:  1.086
    Epoch   2 Batch   95/269 - Train Accuracy:  0.479, Validation Accuracy:  0.498, Loss:  1.070
    Epoch   2 Batch   96/269 - Train Accuracy:  0.472, Validation Accuracy:  0.498, Loss:  1.054
    Epoch   2 Batch   97/269 - Train Accuracy:  0.476, Validation Accuracy:  0.495, Loss:  1.066
    Epoch   2 Batch   98/269 - Train Accuracy:  0.496, Validation Accuracy:  0.505, Loss:  1.049
    Epoch   2 Batch   99/269 - Train Accuracy:  0.458, Validation Accuracy:  0.504, Loss:  1.125
    Epoch   2 Batch  100/269 - Train Accuracy:  0.491, Validation Accuracy:  0.490, Loss:  1.042
    Epoch   2 Batch  101/269 - Train Accuracy:  0.453, Validation Accuracy:  0.501, Loss:  1.121
    Epoch   2 Batch  102/269 - Train Accuracy:  0.475, Validation Accuracy:  0.497, Loss:  1.043
    Epoch   2 Batch  103/269 - Train Accuracy:  0.491, Validation Accuracy:  0.498, Loss:  1.053
    Epoch   2 Batch  104/269 - Train Accuracy:  0.467, Validation Accuracy:  0.495, Loss:  1.067
    Epoch   2 Batch  105/269 - Train Accuracy:  0.455, Validation Accuracy:  0.490, Loss:  1.063
    Epoch   2 Batch  106/269 - Train Accuracy:  0.457, Validation Accuracy:  0.486, Loss:  1.051
    Epoch   2 Batch  107/269 - Train Accuracy:  0.439, Validation Accuracy:  0.484, Loss:  1.097
    Epoch   2 Batch  108/269 - Train Accuracy:  0.476, Validation Accuracy:  0.494, Loss:  1.040
    Epoch   2 Batch  109/269 - Train Accuracy:  0.472, Validation Accuracy:  0.495, Loss:  1.063
    Epoch   2 Batch  110/269 - Train Accuracy:  0.462, Validation Accuracy:  0.490, Loss:  1.046
    Epoch   2 Batch  111/269 - Train Accuracy:  0.455, Validation Accuracy:  0.496, Loss:  1.120
    Epoch   2 Batch  112/269 - Train Accuracy:  0.491, Validation Accuracy:  0.501, Loss:  1.037
    Epoch   2 Batch  113/269 - Train Accuracy:  0.494, Validation Accuracy:  0.497, Loss:  0.997
    Epoch   2 Batch  114/269 - Train Accuracy:  0.472, Validation Accuracy:  0.500, Loss:  1.041
    Epoch   2 Batch  115/269 - Train Accuracy:  0.453, Validation Accuracy:  0.503, Loss:  1.070
    Epoch   2 Batch  116/269 - Train Accuracy:  0.480, Validation Accuracy:  0.499, Loss:  1.044
    Epoch   2 Batch  117/269 - Train Accuracy:  0.463, Validation Accuracy:  0.486, Loss:  1.032
    Epoch   2 Batch  118/269 - Train Accuracy:  0.491, Validation Accuracy:  0.491, Loss:  1.002
    Epoch   2 Batch  119/269 - Train Accuracy:  0.470, Validation Accuracy:  0.496, Loss:  1.084
    Epoch   2 Batch  120/269 - Train Accuracy:  0.440, Validation Accuracy:  0.489, Loss:  1.080
    Epoch   2 Batch  121/269 - Train Accuracy:  0.473, Validation Accuracy:  0.490, Loss:  1.026
    Epoch   2 Batch  122/269 - Train Accuracy:  0.489, Validation Accuracy:  0.496, Loss:  1.016
    Epoch   2 Batch  123/269 - Train Accuracy:  0.467, Validation Accuracy:  0.513, Loss:  1.081
    Epoch   2 Batch  124/269 - Train Accuracy:  0.511, Validation Accuracy:  0.518, Loss:  1.002
    Epoch   2 Batch  125/269 - Train Accuracy:  0.505, Validation Accuracy:  0.521, Loss:  1.002
    Epoch   2 Batch  126/269 - Train Accuracy:  0.501, Validation Accuracy:  0.517, Loss:  1.009
    Epoch   2 Batch  127/269 - Train Accuracy:  0.492, Validation Accuracy:  0.520, Loss:  1.068
    Epoch   2 Batch  128/269 - Train Accuracy:  0.516, Validation Accuracy:  0.518, Loss:  1.007
    Epoch   2 Batch  129/269 - Train Accuracy:  0.496, Validation Accuracy:  0.513, Loss:  1.031
    Epoch   2 Batch  130/269 - Train Accuracy:  0.456, Validation Accuracy:  0.494, Loss:  1.085
    Epoch   2 Batch  131/269 - Train Accuracy:  0.475, Validation Accuracy:  0.508, Loss:  1.058
    Epoch   2 Batch  132/269 - Train Accuracy:  0.478, Validation Accuracy:  0.508, Loss:  1.020
    Epoch   2 Batch  133/269 - Train Accuracy:  0.483, Validation Accuracy:  0.502, Loss:  0.996
    Epoch   2 Batch  134/269 - Train Accuracy:  0.452, Validation Accuracy:  0.500, Loss:  1.057
    Epoch   2 Batch  135/269 - Train Accuracy:  0.441, Validation Accuracy:  0.498, Loss:  1.093
    Epoch   2 Batch  136/269 - Train Accuracy:  0.448, Validation Accuracy:  0.505, Loss:  1.086
    Epoch   2 Batch  137/269 - Train Accuracy:  0.483, Validation Accuracy:  0.507, Loss:  1.064
    Epoch   2 Batch  138/269 - Train Accuracy:  0.461, Validation Accuracy:  0.502, Loss:  1.027
    Epoch   2 Batch  139/269 - Train Accuracy:  0.502, Validation Accuracy:  0.511, Loss:  0.995
    Epoch   2 Batch  140/269 - Train Accuracy:  0.504, Validation Accuracy:  0.521, Loss:  1.021
    Epoch   2 Batch  141/269 - Train Accuracy:  0.484, Validation Accuracy:  0.513, Loss:  1.026
    Epoch   2 Batch  142/269 - Train Accuracy:  0.517, Validation Accuracy:  0.518, Loss:  0.986
    Epoch   2 Batch  143/269 - Train Accuracy:  0.501, Validation Accuracy:  0.517, Loss:  1.004
    Epoch   2 Batch  144/269 - Train Accuracy:  0.498, Validation Accuracy:  0.511, Loss:  0.985
    Epoch   2 Batch  145/269 - Train Accuracy:  0.474, Validation Accuracy:  0.503, Loss:  0.997
    Epoch   2 Batch  146/269 - Train Accuracy:  0.501, Validation Accuracy:  0.502, Loss:  0.990
    Epoch   2 Batch  147/269 - Train Accuracy:  0.501, Validation Accuracy:  0.502, Loss:  0.961
    Epoch   2 Batch  148/269 - Train Accuracy:  0.497, Validation Accuracy:  0.515, Loss:  1.031
    Epoch   2 Batch  149/269 - Train Accuracy:  0.520, Validation Accuracy:  0.530, Loss:  1.010
    Epoch   2 Batch  150/269 - Train Accuracy:  0.513, Validation Accuracy:  0.528, Loss:  1.016
    Epoch   2 Batch  151/269 - Train Accuracy:  0.553, Validation Accuracy:  0.529, Loss:  0.959
    Epoch   2 Batch  152/269 - Train Accuracy:  0.501, Validation Accuracy:  0.526, Loss:  1.000
    Epoch   2 Batch  153/269 - Train Accuracy:  0.507, Validation Accuracy:  0.525, Loss:  0.983
    Epoch   2 Batch  154/269 - Train Accuracy:  0.485, Validation Accuracy:  0.524, Loss:  1.027
    Epoch   2 Batch  155/269 - Train Accuracy:  0.544, Validation Accuracy:  0.522, Loss:  0.947
    Epoch   2 Batch  156/269 - Train Accuracy:  0.503, Validation Accuracy:  0.527, Loss:  1.028
    Epoch   2 Batch  157/269 - Train Accuracy:  0.512, Validation Accuracy:  0.532, Loss:  0.996
    Epoch   2 Batch  158/269 - Train Accuracy:  0.510, Validation Accuracy:  0.526, Loss:  0.994
    Epoch   2 Batch  159/269 - Train Accuracy:  0.500, Validation Accuracy:  0.505, Loss:  0.990
    Epoch   2 Batch  160/269 - Train Accuracy:  0.497, Validation Accuracy:  0.504, Loss:  0.988
    Epoch   2 Batch  161/269 - Train Accuracy:  0.481, Validation Accuracy:  0.514, Loss:  0.996
    Epoch   2 Batch  162/269 - Train Accuracy:  0.487, Validation Accuracy:  0.513, Loss:  0.980
    Epoch   2 Batch  163/269 - Train Accuracy:  0.489, Validation Accuracy:  0.502, Loss:  0.986
    Epoch   2 Batch  164/269 - Train Accuracy:  0.488, Validation Accuracy:  0.502, Loss:  0.978
    Epoch   2 Batch  165/269 - Train Accuracy:  0.478, Validation Accuracy:  0.518, Loss:  1.007
    Epoch   2 Batch  166/269 - Train Accuracy:  0.525, Validation Accuracy:  0.515, Loss:  0.930
    Epoch   2 Batch  167/269 - Train Accuracy:  0.523, Validation Accuracy:  0.514, Loss:  0.988
    Epoch   2 Batch  168/269 - Train Accuracy:  0.496, Validation Accuracy:  0.513, Loss:  0.986
    Epoch   2 Batch  169/269 - Train Accuracy:  0.512, Validation Accuracy:  0.520, Loss:  0.989
    Epoch   2 Batch  170/269 - Train Accuracy:  0.500, Validation Accuracy:  0.520, Loss:  0.971
    Epoch   2 Batch  171/269 - Train Accuracy:  0.480, Validation Accuracy:  0.522, Loss:  1.016
    Epoch   2 Batch  172/269 - Train Accuracy:  0.505, Validation Accuracy:  0.522, Loss:  0.986
    Epoch   2 Batch  173/269 - Train Accuracy:  0.509, Validation Accuracy:  0.525, Loss:  0.968
    Epoch   2 Batch  174/269 - Train Accuracy:  0.510, Validation Accuracy:  0.523, Loss:  0.979
    Epoch   2 Batch  175/269 - Train Accuracy:  0.509, Validation Accuracy:  0.525, Loss:  0.986
    Epoch   2 Batch  176/269 - Train Accuracy:  0.507, Validation Accuracy:  0.532, Loss:  1.019
    Epoch   2 Batch  177/269 - Train Accuracy:  0.532, Validation Accuracy:  0.527, Loss:  0.946
    Epoch   2 Batch  178/269 - Train Accuracy:  0.499, Validation Accuracy:  0.529, Loss:  1.011
    Epoch   2 Batch  179/269 - Train Accuracy:  0.505, Validation Accuracy:  0.523, Loss:  0.968
    Epoch   2 Batch  180/269 - Train Accuracy:  0.512, Validation Accuracy:  0.523, Loss:  0.962
    Epoch   2 Batch  181/269 - Train Accuracy:  0.517, Validation Accuracy:  0.519, Loss:  0.961
    Epoch   2 Batch  182/269 - Train Accuracy:  0.516, Validation Accuracy:  0.516, Loss:  0.987
    Epoch   2 Batch  183/269 - Train Accuracy:  0.575, Validation Accuracy:  0.522, Loss:  0.835
    Epoch   2 Batch  184/269 - Train Accuracy:  0.482, Validation Accuracy:  0.518, Loss:  0.998
    Epoch   2 Batch  185/269 - Train Accuracy:  0.515, Validation Accuracy:  0.520, Loss:  0.957
    Epoch   2 Batch  186/269 - Train Accuracy:  0.488, Validation Accuracy:  0.534, Loss:  0.999
    Epoch   2 Batch  187/269 - Train Accuracy:  0.527, Validation Accuracy:  0.524, Loss:  0.943
    Epoch   2 Batch  188/269 - Train Accuracy:  0.518, Validation Accuracy:  0.526, Loss:  0.934
    Epoch   2 Batch  189/269 - Train Accuracy:  0.528, Validation Accuracy:  0.523, Loss:  0.946
    Epoch   2 Batch  190/269 - Train Accuracy:  0.500, Validation Accuracy:  0.523, Loss:  0.944
    Epoch   2 Batch  191/269 - Train Accuracy:  0.516, Validation Accuracy:  0.524, Loss:  0.954
    Epoch   2 Batch  192/269 - Train Accuracy:  0.516, Validation Accuracy:  0.522, Loss:  0.961
    Epoch   2 Batch  193/269 - Train Accuracy:  0.502, Validation Accuracy:  0.520, Loss:  0.953
    Epoch   2 Batch  194/269 - Train Accuracy:  0.527, Validation Accuracy:  0.522, Loss:  0.967
    Epoch   2 Batch  195/269 - Train Accuracy:  0.514, Validation Accuracy:  0.536, Loss:  0.962
    Epoch   2 Batch  196/269 - Train Accuracy:  0.508, Validation Accuracy:  0.534, Loss:  0.944
    Epoch   2 Batch  197/269 - Train Accuracy:  0.488, Validation Accuracy:  0.530, Loss:  0.996
    Epoch   2 Batch  198/269 - Train Accuracy:  0.490, Validation Accuracy:  0.530, Loss:  1.012
    Epoch   2 Batch  199/269 - Train Accuracy:  0.521, Validation Accuracy:  0.536, Loss:  0.982
    Epoch   2 Batch  200/269 - Train Accuracy:  0.502, Validation Accuracy:  0.538, Loss:  0.986
    Epoch   2 Batch  201/269 - Train Accuracy:  0.511, Validation Accuracy:  0.535, Loss:  0.946
    Epoch   2 Batch  202/269 - Train Accuracy:  0.498, Validation Accuracy:  0.530, Loss:  0.954
    Epoch   2 Batch  203/269 - Train Accuracy:  0.485, Validation Accuracy:  0.527, Loss:  1.011
    Epoch   2 Batch  204/269 - Train Accuracy:  0.501, Validation Accuracy:  0.528, Loss:  0.978
    Epoch   2 Batch  205/269 - Train Accuracy:  0.518, Validation Accuracy:  0.527, Loss:  0.934
    Epoch   2 Batch  206/269 - Train Accuracy:  0.475, Validation Accuracy:  0.520, Loss:  1.001
    Epoch   2 Batch  207/269 - Train Accuracy:  0.525, Validation Accuracy:  0.520, Loss:  0.915
    Epoch   2 Batch  208/269 - Train Accuracy:  0.481, Validation Accuracy:  0.531, Loss:  0.997
    Epoch   2 Batch  209/269 - Train Accuracy:  0.488, Validation Accuracy:  0.533, Loss:  0.969
    Epoch   2 Batch  210/269 - Train Accuracy:  0.525, Validation Accuracy:  0.524, Loss:  0.937
    Epoch   2 Batch  211/269 - Train Accuracy:  0.506, Validation Accuracy:  0.507, Loss:  0.947
    Epoch   2 Batch  212/269 - Train Accuracy:  0.499, Validation Accuracy:  0.507, Loss:  0.923
    Epoch   2 Batch  213/269 - Train Accuracy:  0.523, Validation Accuracy:  0.527, Loss:  0.925
    Epoch   2 Batch  214/269 - Train Accuracy:  0.517, Validation Accuracy:  0.525, Loss:  0.932
    Epoch   2 Batch  215/269 - Train Accuracy:  0.529, Validation Accuracy:  0.512, Loss:  0.883
    Epoch   2 Batch  216/269 - Train Accuracy:  0.473, Validation Accuracy:  0.502, Loss:  1.003
    Epoch   2 Batch  217/269 - Train Accuracy:  0.462, Validation Accuracy:  0.504, Loss:  0.972
    Epoch   2 Batch  218/269 - Train Accuracy:  0.484, Validation Accuracy:  0.506, Loss:  0.977
    Epoch   2 Batch  219/269 - Train Accuracy:  0.482, Validation Accuracy:  0.505, Loss:  0.975
    Epoch   2 Batch  220/269 - Train Accuracy:  0.518, Validation Accuracy:  0.513, Loss:  0.894
    Epoch   2 Batch  221/269 - Train Accuracy:  0.516, Validation Accuracy:  0.507, Loss:  0.926
    Epoch   2 Batch  222/269 - Train Accuracy:  0.501, Validation Accuracy:  0.500, Loss:  0.895
    Epoch   2 Batch  223/269 - Train Accuracy:  0.488, Validation Accuracy:  0.493, Loss:  0.908
    Epoch   2 Batch  224/269 - Train Accuracy:  0.510, Validation Accuracy:  0.505, Loss:  0.950
    Epoch   2 Batch  225/269 - Train Accuracy:  0.483, Validation Accuracy:  0.519, Loss:  0.952
    Epoch   2 Batch  226/269 - Train Accuracy:  0.512, Validation Accuracy:  0.525, Loss:  0.922
    Epoch   2 Batch  227/269 - Train Accuracy:  0.592, Validation Accuracy:  0.530, Loss:  0.822
    Epoch   2 Batch  228/269 - Train Accuracy:  0.514, Validation Accuracy:  0.533, Loss:  0.925
    Epoch   2 Batch  229/269 - Train Accuracy:  0.516, Validation Accuracy:  0.531, Loss:  0.923
    Epoch   2 Batch  230/269 - Train Accuracy:  0.512, Validation Accuracy:  0.533, Loss:  0.924
    Epoch   2 Batch  231/269 - Train Accuracy:  0.471, Validation Accuracy:  0.528, Loss:  0.977
    Epoch   2 Batch  232/269 - Train Accuracy:  0.473, Validation Accuracy:  0.509, Loss:  0.971
    Epoch   2 Batch  233/269 - Train Accuracy:  0.491, Validation Accuracy:  0.508, Loss:  0.927
    Epoch   2 Batch  234/269 - Train Accuracy:  0.491, Validation Accuracy:  0.505, Loss:  0.917
    Epoch   2 Batch  235/269 - Train Accuracy:  0.519, Validation Accuracy:  0.522, Loss:  0.916
    Epoch   2 Batch  236/269 - Train Accuracy:  0.514, Validation Accuracy:  0.524, Loss:  0.913
    Epoch   2 Batch  237/269 - Train Accuracy:  0.516, Validation Accuracy:  0.526, Loss:  0.918
    Epoch   2 Batch  238/269 - Train Accuracy:  0.540, Validation Accuracy:  0.536, Loss:  0.914
    Epoch   2 Batch  239/269 - Train Accuracy:  0.533, Validation Accuracy:  0.538, Loss:  0.909
    Epoch   2 Batch  240/269 - Train Accuracy:  0.557, Validation Accuracy:  0.538, Loss:  0.847
    Epoch   2 Batch  241/269 - Train Accuracy:  0.527, Validation Accuracy:  0.531, Loss:  0.920
    Epoch   2 Batch  242/269 - Train Accuracy:  0.508, Validation Accuracy:  0.533, Loss:  0.909
    Epoch   2 Batch  243/269 - Train Accuracy:  0.544, Validation Accuracy:  0.534, Loss:  0.887
    Epoch   2 Batch  244/269 - Train Accuracy:  0.515, Validation Accuracy:  0.530, Loss:  0.907
    Epoch   2 Batch  245/269 - Train Accuracy:  0.494, Validation Accuracy:  0.529, Loss:  0.954
    Epoch   2 Batch  246/269 - Train Accuracy:  0.503, Validation Accuracy:  0.516, Loss:  0.913
    Epoch   2 Batch  247/269 - Train Accuracy:  0.508, Validation Accuracy:  0.527, Loss:  0.942
    Epoch   2 Batch  248/269 - Train Accuracy:  0.516, Validation Accuracy:  0.542, Loss:  0.896
    Epoch   2 Batch  249/269 - Train Accuracy:  0.558, Validation Accuracy:  0.539, Loss:  0.866
    Epoch   2 Batch  250/269 - Train Accuracy:  0.509, Validation Accuracy:  0.536, Loss:  0.929
    Epoch   2 Batch  251/269 - Train Accuracy:  0.542, Validation Accuracy:  0.537, Loss:  0.891
    Epoch   2 Batch  252/269 - Train Accuracy:  0.518, Validation Accuracy:  0.537, Loss:  0.912
    Epoch   2 Batch  253/269 - Train Accuracy:  0.509, Validation Accuracy:  0.535, Loss:  0.911
    Epoch   2 Batch  254/269 - Train Accuracy:  0.521, Validation Accuracy:  0.534, Loss:  0.896
    Epoch   2 Batch  255/269 - Train Accuracy:  0.542, Validation Accuracy:  0.526, Loss:  0.861
    Epoch   2 Batch  256/269 - Train Accuracy:  0.490, Validation Accuracy:  0.511, Loss:  0.916
    Epoch   2 Batch  257/269 - Train Accuracy:  0.489, Validation Accuracy:  0.511, Loss:  0.913
    Epoch   2 Batch  258/269 - Train Accuracy:  0.487, Validation Accuracy:  0.515, Loss:  0.899
    Epoch   2 Batch  259/269 - Train Accuracy:  0.513, Validation Accuracy:  0.514, Loss:  0.904
    Epoch   2 Batch  260/269 - Train Accuracy:  0.483, Validation Accuracy:  0.516, Loss:  0.943
    Epoch   2 Batch  261/269 - Train Accuracy:  0.475, Validation Accuracy:  0.521, Loss:  0.951
    Epoch   2 Batch  262/269 - Train Accuracy:  0.514, Validation Accuracy:  0.523, Loss:  0.899
    Epoch   2 Batch  263/269 - Train Accuracy:  0.518, Validation Accuracy:  0.540, Loss:  0.927
    Epoch   2 Batch  264/269 - Train Accuracy:  0.510, Validation Accuracy:  0.536, Loss:  0.951
    Epoch   2 Batch  265/269 - Train Accuracy:  0.501, Validation Accuracy:  0.536, Loss:  0.922
    Epoch   2 Batch  266/269 - Train Accuracy:  0.531, Validation Accuracy:  0.532, Loss:  0.877
    Epoch   2 Batch  267/269 - Train Accuracy:  0.513, Validation Accuracy:  0.530, Loss:  0.909
    Epoch   3 Batch    0/269 - Train Accuracy:  0.497, Validation Accuracy:  0.533, Loss:  0.937
    Epoch   3 Batch    1/269 - Train Accuracy:  0.491, Validation Accuracy:  0.538, Loss:  0.909
    Epoch   3 Batch    2/269 - Train Accuracy:  0.503, Validation Accuracy:  0.535, Loss:  0.908
    Epoch   3 Batch    3/269 - Train Accuracy:  0.502, Validation Accuracy:  0.536, Loss:  0.915
    Epoch   3 Batch    4/269 - Train Accuracy:  0.499, Validation Accuracy:  0.536, Loss:  0.931
    Epoch   3 Batch    5/269 - Train Accuracy:  0.490, Validation Accuracy:  0.534, Loss:  0.920
    Epoch   3 Batch    6/269 - Train Accuracy:  0.541, Validation Accuracy:  0.544, Loss:  0.850
    Epoch   3 Batch    7/269 - Train Accuracy:  0.530, Validation Accuracy:  0.535, Loss:  0.877
    Epoch   3 Batch    8/269 - Train Accuracy:  0.506, Validation Accuracy:  0.539, Loss:  0.932
    Epoch   3 Batch    9/269 - Train Accuracy:  0.495, Validation Accuracy:  0.533, Loss:  0.905
    Epoch   3 Batch   10/269 - Train Accuracy:  0.490, Validation Accuracy:  0.534, Loss:  0.911
    Epoch   3 Batch   11/269 - Train Accuracy:  0.515, Validation Accuracy:  0.539, Loss:  0.895
    Epoch   3 Batch   12/269 - Train Accuracy:  0.486, Validation Accuracy:  0.540, Loss:  0.936
    Epoch   3 Batch   13/269 - Train Accuracy:  0.550, Validation Accuracy:  0.532, Loss:  0.823
    Epoch   3 Batch   14/269 - Train Accuracy:  0.526, Validation Accuracy:  0.537, Loss:  0.869
    Epoch   3 Batch   15/269 - Train Accuracy:  0.513, Validation Accuracy:  0.537, Loss:  0.874
    Epoch   3 Batch   16/269 - Train Accuracy:  0.536, Validation Accuracy:  0.537, Loss:  0.878
    Epoch   3 Batch   17/269 - Train Accuracy:  0.519, Validation Accuracy:  0.541, Loss:  0.862
    Epoch   3 Batch   18/269 - Train Accuracy:  0.498, Validation Accuracy:  0.535, Loss:  0.905
    Epoch   3 Batch   19/269 - Train Accuracy:  0.550, Validation Accuracy:  0.530, Loss:  0.824
    Epoch   3 Batch   20/269 - Train Accuracy:  0.516, Validation Accuracy:  0.535, Loss:  0.908
    Epoch   3 Batch   21/269 - Train Accuracy:  0.517, Validation Accuracy:  0.543, Loss:  0.934
    Epoch   3 Batch   22/269 - Train Accuracy:  0.531, Validation Accuracy:  0.530, Loss:  0.858
    Epoch   3 Batch   23/269 - Train Accuracy:  0.535, Validation Accuracy:  0.533, Loss:  0.875
    Epoch   3 Batch   24/269 - Train Accuracy:  0.512, Validation Accuracy:  0.532, Loss:  0.908
    Epoch   3 Batch   25/269 - Train Accuracy:  0.486, Validation Accuracy:  0.523, Loss:  0.918
    Epoch   3 Batch   26/269 - Train Accuracy:  0.553, Validation Accuracy:  0.528, Loss:  0.810
    Epoch   3 Batch   27/269 - Train Accuracy:  0.524, Validation Accuracy:  0.524, Loss:  0.864
    Epoch   3 Batch   28/269 - Train Accuracy:  0.491, Validation Accuracy:  0.519, Loss:  0.932
    Epoch   3 Batch   29/269 - Train Accuracy:  0.487, Validation Accuracy:  0.515, Loss:  0.890
    Epoch   3 Batch   30/269 - Train Accuracy:  0.517, Validation Accuracy:  0.514, Loss:  0.862
    Epoch   3 Batch   31/269 - Train Accuracy:  0.520, Validation Accuracy:  0.516, Loss:  0.850
    Epoch   3 Batch   32/269 - Train Accuracy:  0.511, Validation Accuracy:  0.516, Loss:  0.866
    Epoch   3 Batch   33/269 - Train Accuracy:  0.527, Validation Accuracy:  0.514, Loss:  0.831
    Epoch   3 Batch   34/269 - Train Accuracy:  0.518, Validation Accuracy:  0.517, Loss:  0.855
    Epoch   3 Batch   35/269 - Train Accuracy:  0.522, Validation Accuracy:  0.519, Loss:  0.875
    Epoch   3 Batch   36/269 - Train Accuracy:  0.539, Validation Accuracy:  0.543, Loss:  0.871
    Epoch   3 Batch   37/269 - Train Accuracy:  0.520, Validation Accuracy:  0.538, Loss:  0.861
    Epoch   3 Batch   38/269 - Train Accuracy:  0.510, Validation Accuracy:  0.535, Loss:  0.865
    Epoch   3 Batch   39/269 - Train Accuracy:  0.521, Validation Accuracy:  0.531, Loss:  0.852
    Epoch   3 Batch   40/269 - Train Accuracy:  0.496, Validation Accuracy:  0.531, Loss:  0.891
    Epoch   3 Batch   41/269 - Train Accuracy:  0.514, Validation Accuracy:  0.518, Loss:  0.880
    Epoch   3 Batch   42/269 - Train Accuracy:  0.533, Validation Accuracy:  0.526, Loss:  0.809
    Epoch   3 Batch   43/269 - Train Accuracy:  0.511, Validation Accuracy:  0.539, Loss:  0.884
    Epoch   3 Batch   44/269 - Train Accuracy:  0.531, Validation Accuracy:  0.538, Loss:  0.864
    Epoch   3 Batch   45/269 - Train Accuracy:  0.493, Validation Accuracy:  0.538, Loss:  0.896
    Epoch   3 Batch   46/269 - Train Accuracy:  0.501, Validation Accuracy:  0.528, Loss:  0.891
    Epoch   3 Batch   47/269 - Train Accuracy:  0.548, Validation Accuracy:  0.527, Loss:  0.801
    Epoch   3 Batch   48/269 - Train Accuracy:  0.520, Validation Accuracy:  0.526, Loss:  0.830
    Epoch   3 Batch   49/269 - Train Accuracy:  0.503, Validation Accuracy:  0.526, Loss:  0.874
    Epoch   3 Batch   50/269 - Train Accuracy:  0.497, Validation Accuracy:  0.529, Loss:  0.888
    Epoch   3 Batch   51/269 - Train Accuracy:  0.510, Validation Accuracy:  0.541, Loss:  0.862
    Epoch   3 Batch   52/269 - Train Accuracy:  0.508, Validation Accuracy:  0.514, Loss:  0.826
    Epoch   3 Batch   53/269 - Train Accuracy:  0.498, Validation Accuracy:  0.524, Loss:  0.899
    Epoch   3 Batch   54/269 - Train Accuracy:  0.525, Validation Accuracy:  0.528, Loss:  0.882
    Epoch   3 Batch   55/269 - Train Accuracy:  0.497, Validation Accuracy:  0.518, Loss:  0.841
    Epoch   3 Batch   56/269 - Train Accuracy:  0.508, Validation Accuracy:  0.510, Loss:  0.850
    Epoch   3 Batch   57/269 - Train Accuracy:  0.523, Validation Accuracy:  0.513, Loss:  0.862
    Epoch   3 Batch   58/269 - Train Accuracy:  0.514, Validation Accuracy:  0.514, Loss:  0.834
    Epoch   3 Batch   59/269 - Train Accuracy:  0.531, Validation Accuracy:  0.526, Loss:  0.822
    Epoch   3 Batch   60/269 - Train Accuracy:  0.531, Validation Accuracy:  0.541, Loss:  0.811
    Epoch   3 Batch   61/269 - Train Accuracy:  0.547, Validation Accuracy:  0.541, Loss:  0.798
    Epoch   3 Batch   62/269 - Train Accuracy:  0.541, Validation Accuracy:  0.537, Loss:  0.820
    Epoch   3 Batch   63/269 - Train Accuracy:  0.496, Validation Accuracy:  0.516, Loss:  0.861
    Epoch   3 Batch   64/269 - Train Accuracy:  0.512, Validation Accuracy:  0.509, Loss:  0.833
    Epoch   3 Batch   65/269 - Train Accuracy:  0.506, Validation Accuracy:  0.509, Loss:  0.834
    Epoch   3 Batch   66/269 - Train Accuracy:  0.520, Validation Accuracy:  0.518, Loss:  0.811
    Epoch   3 Batch   67/269 - Train Accuracy:  0.508, Validation Accuracy:  0.523, Loss:  0.859
    Epoch   3 Batch   68/269 - Train Accuracy:  0.490, Validation Accuracy:  0.517, Loss:  0.843
    Epoch   3 Batch   69/269 - Train Accuracy:  0.492, Validation Accuracy:  0.515, Loss:  0.916
    Epoch   3 Batch   70/269 - Train Accuracy:  0.523, Validation Accuracy:  0.536, Loss:  0.842
    Epoch   3 Batch   71/269 - Train Accuracy:  0.512, Validation Accuracy:  0.532, Loss:  0.867
    Epoch   3 Batch   72/269 - Train Accuracy:  0.535, Validation Accuracy:  0.523, Loss:  0.820
    Epoch   3 Batch   73/269 - Train Accuracy:  0.525, Validation Accuracy:  0.524, Loss:  0.861
    Epoch   3 Batch   74/269 - Train Accuracy:  0.518, Validation Accuracy:  0.531, Loss:  0.841
    Epoch   3 Batch   75/269 - Train Accuracy:  0.529, Validation Accuracy:  0.539, Loss:  0.834
    Epoch   3 Batch   76/269 - Train Accuracy:  0.521, Validation Accuracy:  0.545, Loss:  0.851
    Epoch   3 Batch   77/269 - Train Accuracy:  0.560, Validation Accuracy:  0.542, Loss:  0.829
    Epoch   3 Batch   78/269 - Train Accuracy:  0.528, Validation Accuracy:  0.527, Loss:  0.820
    Epoch   3 Batch   79/269 - Train Accuracy:  0.516, Validation Accuracy:  0.527, Loss:  0.820
    Epoch   3 Batch   80/269 - Train Accuracy:  0.535, Validation Accuracy:  0.536, Loss:  0.831
    Epoch   3 Batch   81/269 - Train Accuracy:  0.541, Validation Accuracy:  0.543, Loss:  0.844
    Epoch   3 Batch   82/269 - Train Accuracy:  0.547, Validation Accuracy:  0.543, Loss:  0.804
    Epoch   3 Batch   83/269 - Train Accuracy:  0.554, Validation Accuracy:  0.546, Loss:  0.834
    Epoch   3 Batch   84/269 - Train Accuracy:  0.548, Validation Accuracy:  0.545, Loss:  0.813
    Epoch   3 Batch   85/269 - Train Accuracy:  0.543, Validation Accuracy:  0.547, Loss:  0.826
    Epoch   3 Batch   86/269 - Train Accuracy:  0.520, Validation Accuracy:  0.544, Loss:  0.832
    Epoch   3 Batch   87/269 - Train Accuracy:  0.518, Validation Accuracy:  0.545, Loss:  0.879
    Epoch   3 Batch   88/269 - Train Accuracy:  0.537, Validation Accuracy:  0.544, Loss:  0.832
    Epoch   3 Batch   89/269 - Train Accuracy:  0.561, Validation Accuracy:  0.550, Loss:  0.826
    Epoch   3 Batch   90/269 - Train Accuracy:  0.500, Validation Accuracy:  0.545, Loss:  0.879
    Epoch   3 Batch   91/269 - Train Accuracy:  0.532, Validation Accuracy:  0.541, Loss:  0.803
    Epoch   3 Batch   92/269 - Train Accuracy:  0.547, Validation Accuracy:  0.545, Loss:  0.809
    Epoch   3 Batch   93/269 - Train Accuracy:  0.568, Validation Accuracy:  0.550, Loss:  0.784
    Epoch   3 Batch   94/269 - Train Accuracy:  0.557, Validation Accuracy:  0.548, Loss:  0.835
    Epoch   3 Batch   95/269 - Train Accuracy:  0.531, Validation Accuracy:  0.546, Loss:  0.832
    Epoch   3 Batch   96/269 - Train Accuracy:  0.548, Validation Accuracy:  0.541, Loss:  0.812
    Epoch   3 Batch   97/269 - Train Accuracy:  0.525, Validation Accuracy:  0.536, Loss:  0.818
    Epoch   3 Batch   98/269 - Train Accuracy:  0.546, Validation Accuracy:  0.548, Loss:  0.816
    Epoch   3 Batch   99/269 - Train Accuracy:  0.520, Validation Accuracy:  0.545, Loss:  0.860
    Epoch   3 Batch  100/269 - Train Accuracy:  0.536, Validation Accuracy:  0.526, Loss:  0.809
    Epoch   3 Batch  101/269 - Train Accuracy:  0.483, Validation Accuracy:  0.531, Loss:  0.865
    Epoch   3 Batch  102/269 - Train Accuracy:  0.527, Validation Accuracy:  0.543, Loss:  0.810
    Epoch   3 Batch  103/269 - Train Accuracy:  0.541, Validation Accuracy:  0.538, Loss:  0.813
    Epoch   3 Batch  104/269 - Train Accuracy:  0.505, Validation Accuracy:  0.521, Loss:  0.818
    Epoch   3 Batch  105/269 - Train Accuracy:  0.548, Validation Accuracy:  0.545, Loss:  0.831
    Epoch   3 Batch  106/269 - Train Accuracy:  0.547, Validation Accuracy:  0.558, Loss:  0.806
    Epoch   3 Batch  107/269 - Train Accuracy:  0.505, Validation Accuracy:  0.531, Loss:  0.862
    Epoch   3 Batch  108/269 - Train Accuracy:  0.528, Validation Accuracy:  0.526, Loss:  0.808
    Epoch   3 Batch  109/269 - Train Accuracy:  0.536, Validation Accuracy:  0.558, Loss:  0.825
    Epoch   3 Batch  110/269 - Train Accuracy:  0.543, Validation Accuracy:  0.554, Loss:  0.806
    Epoch   3 Batch  111/269 - Train Accuracy:  0.524, Validation Accuracy:  0.553, Loss:  0.866
    Epoch   3 Batch  112/269 - Train Accuracy:  0.546, Validation Accuracy:  0.540, Loss:  0.815
    Epoch   3 Batch  113/269 - Train Accuracy:  0.549, Validation Accuracy:  0.539, Loss:  0.777
    Epoch   3 Batch  114/269 - Train Accuracy:  0.529, Validation Accuracy:  0.538, Loss:  0.802
    Epoch   3 Batch  115/269 - Train Accuracy:  0.527, Validation Accuracy:  0.541, Loss:  0.840
    Epoch   3 Batch  116/269 - Train Accuracy:  0.544, Validation Accuracy:  0.535, Loss:  0.816
    Epoch   3 Batch  117/269 - Train Accuracy:  0.526, Validation Accuracy:  0.539, Loss:  0.807
    Epoch   3 Batch  118/269 - Train Accuracy:  0.544, Validation Accuracy:  0.535, Loss:  0.785
    Epoch   3 Batch  119/269 - Train Accuracy:  0.517, Validation Accuracy:  0.537, Loss:  0.849
    Epoch   3 Batch  120/269 - Train Accuracy:  0.523, Validation Accuracy:  0.544, Loss:  0.835
    Epoch   3 Batch  121/269 - Train Accuracy:  0.550, Validation Accuracy:  0.549, Loss:  0.791
    Epoch   3 Batch  122/269 - Train Accuracy:  0.549, Validation Accuracy:  0.549, Loss:  0.794
    Epoch   3 Batch  123/269 - Train Accuracy:  0.537, Validation Accuracy:  0.542, Loss:  0.836
    Epoch   3 Batch  124/269 - Train Accuracy:  0.545, Validation Accuracy:  0.555, Loss:  0.788
    Epoch   3 Batch  125/269 - Train Accuracy:  0.559, Validation Accuracy:  0.562, Loss:  0.789
    Epoch   3 Batch  126/269 - Train Accuracy:  0.572, Validation Accuracy:  0.564, Loss:  0.791
    Epoch   3 Batch  127/269 - Train Accuracy:  0.550, Validation Accuracy:  0.556, Loss:  0.836
    Epoch   3 Batch  128/269 - Train Accuracy:  0.568, Validation Accuracy:  0.553, Loss:  0.798
    Epoch   3 Batch  129/269 - Train Accuracy:  0.568, Validation Accuracy:  0.564, Loss:  0.807
    Epoch   3 Batch  130/269 - Train Accuracy:  0.541, Validation Accuracy:  0.556, Loss:  0.853
    Epoch   3 Batch  131/269 - Train Accuracy:  0.553, Validation Accuracy:  0.554, Loss:  0.822
    Epoch   3 Batch  132/269 - Train Accuracy:  0.548, Validation Accuracy:  0.567, Loss:  0.806
    Epoch   3 Batch  133/269 - Train Accuracy:  0.547, Validation Accuracy:  0.555, Loss:  0.781
    Epoch   3 Batch  134/269 - Train Accuracy:  0.510, Validation Accuracy:  0.551, Loss:  0.827
    Epoch   3 Batch  135/269 - Train Accuracy:  0.529, Validation Accuracy:  0.549, Loss:  0.850
    Epoch   3 Batch  136/269 - Train Accuracy:  0.528, Validation Accuracy:  0.554, Loss:  0.853
    Epoch   3 Batch  137/269 - Train Accuracy:  0.538, Validation Accuracy:  0.555, Loss:  0.837
    Epoch   3 Batch  138/269 - Train Accuracy:  0.536, Validation Accuracy:  0.548, Loss:  0.819
    Epoch   3 Batch  139/269 - Train Accuracy:  0.576, Validation Accuracy:  0.564, Loss:  0.778
    Epoch   3 Batch  140/269 - Train Accuracy:  0.588, Validation Accuracy:  0.578, Loss:  0.809
    Epoch   3 Batch  141/269 - Train Accuracy:  0.557, Validation Accuracy:  0.563, Loss:  0.809
    Epoch   3 Batch  142/269 - Train Accuracy:  0.565, Validation Accuracy:  0.560, Loss:  0.772
    Epoch   3 Batch  143/269 - Train Accuracy:  0.580, Validation Accuracy:  0.575, Loss:  0.793
    Epoch   3 Batch  144/269 - Train Accuracy:  0.584, Validation Accuracy:  0.572, Loss:  0.760
    Epoch   3 Batch  145/269 - Train Accuracy:  0.574, Validation Accuracy:  0.561, Loss:  0.783
    Epoch   3 Batch  146/269 - Train Accuracy:  0.549, Validation Accuracy:  0.550, Loss:  0.771
    Epoch   3 Batch  147/269 - Train Accuracy:  0.584, Validation Accuracy:  0.557, Loss:  0.763
    Epoch   3 Batch  148/269 - Train Accuracy:  0.559, Validation Accuracy:  0.562, Loss:  0.797
    Epoch   3 Batch  149/269 - Train Accuracy:  0.569, Validation Accuracy:  0.561, Loss:  0.795
    Epoch   3 Batch  150/269 - Train Accuracy:  0.562, Validation Accuracy:  0.551, Loss:  0.792
    Epoch   3 Batch  151/269 - Train Accuracy:  0.597, Validation Accuracy:  0.556, Loss:  0.754
    Epoch   3 Batch  152/269 - Train Accuracy:  0.558, Validation Accuracy:  0.567, Loss:  0.789
    Epoch   3 Batch  153/269 - Train Accuracy:  0.571, Validation Accuracy:  0.570, Loss:  0.771
    Epoch   3 Batch  154/269 - Train Accuracy:  0.550, Validation Accuracy:  0.567, Loss:  0.799
    Epoch   3 Batch  155/269 - Train Accuracy:  0.599, Validation Accuracy:  0.569, Loss:  0.746
    Epoch   3 Batch  156/269 - Train Accuracy:  0.563, Validation Accuracy:  0.576, Loss:  0.820
    Epoch   3 Batch  157/269 - Train Accuracy:  0.572, Validation Accuracy:  0.564, Loss:  0.781
    Epoch   3 Batch  158/269 - Train Accuracy:  0.570, Validation Accuracy:  0.563, Loss:  0.784
    Epoch   3 Batch  159/269 - Train Accuracy:  0.564, Validation Accuracy:  0.556, Loss:  0.786
    Epoch   3 Batch  160/269 - Train Accuracy:  0.575, Validation Accuracy:  0.563, Loss:  0.775
    Epoch   3 Batch  161/269 - Train Accuracy:  0.561, Validation Accuracy:  0.574, Loss:  0.787
    Epoch   3 Batch  162/269 - Train Accuracy:  0.564, Validation Accuracy:  0.566, Loss:  0.769
    Epoch   3 Batch  163/269 - Train Accuracy:  0.574, Validation Accuracy:  0.566, Loss:  0.775
    Epoch   3 Batch  164/269 - Train Accuracy:  0.589, Validation Accuracy:  0.576, Loss:  0.769
    Epoch   3 Batch  165/269 - Train Accuracy:  0.554, Validation Accuracy:  0.572, Loss:  0.790
    Epoch   3 Batch  166/269 - Train Accuracy:  0.596, Validation Accuracy:  0.568, Loss:  0.739
    Epoch   3 Batch  167/269 - Train Accuracy:  0.558, Validation Accuracy:  0.562, Loss:  0.783
    Epoch   3 Batch  168/269 - Train Accuracy:  0.551, Validation Accuracy:  0.556, Loss:  0.783
    Epoch   3 Batch  169/269 - Train Accuracy:  0.560, Validation Accuracy:  0.564, Loss:  0.787
    Epoch   3 Batch  170/269 - Train Accuracy:  0.566, Validation Accuracy:  0.571, Loss:  0.766
    Epoch   3 Batch  171/269 - Train Accuracy:  0.563, Validation Accuracy:  0.574, Loss:  0.805
    Epoch   3 Batch  172/269 - Train Accuracy:  0.566, Validation Accuracy:  0.575, Loss:  0.780
    Epoch   3 Batch  173/269 - Train Accuracy:  0.573, Validation Accuracy:  0.573, Loss:  0.760
    Epoch   3 Batch  174/269 - Train Accuracy:  0.559, Validation Accuracy:  0.569, Loss:  0.774
    Epoch   3 Batch  175/269 - Train Accuracy:  0.577, Validation Accuracy:  0.567, Loss:  0.787
    Epoch   3 Batch  176/269 - Train Accuracy:  0.554, Validation Accuracy:  0.565, Loss:  0.813
    Epoch   3 Batch  177/269 - Train Accuracy:  0.573, Validation Accuracy:  0.565, Loss:  0.746
    Epoch   3 Batch  178/269 - Train Accuracy:  0.556, Validation Accuracy:  0.560, Loss:  0.795
    Epoch   3 Batch  179/269 - Train Accuracy:  0.566, Validation Accuracy:  0.558, Loss:  0.765
    Epoch   3 Batch  180/269 - Train Accuracy:  0.558, Validation Accuracy:  0.558, Loss:  0.758
    Epoch   3 Batch  181/269 - Train Accuracy:  0.559, Validation Accuracy:  0.565, Loss:  0.766
    Epoch   3 Batch  182/269 - Train Accuracy:  0.574, Validation Accuracy:  0.566, Loss:  0.771
    Epoch   3 Batch  183/269 - Train Accuracy:  0.618, Validation Accuracy:  0.563, Loss:  0.664
    Epoch   3 Batch  184/269 - Train Accuracy:  0.562, Validation Accuracy:  0.571, Loss:  0.791
    Epoch   3 Batch  185/269 - Train Accuracy:  0.584, Validation Accuracy:  0.577, Loss:  0.761
    Epoch   3 Batch  186/269 - Train Accuracy:  0.552, Validation Accuracy:  0.583, Loss:  0.783
    Epoch   3 Batch  187/269 - Train Accuracy:  0.576, Validation Accuracy:  0.574, Loss:  0.749
    Epoch   3 Batch  188/269 - Train Accuracy:  0.571, Validation Accuracy:  0.569, Loss:  0.744
    Epoch   3 Batch  189/269 - Train Accuracy:  0.583, Validation Accuracy:  0.575, Loss:  0.746
    Epoch   3 Batch  190/269 - Train Accuracy:  0.574, Validation Accuracy:  0.579, Loss:  0.749
    Epoch   3 Batch  191/269 - Train Accuracy:  0.589, Validation Accuracy:  0.576, Loss:  0.751
    Epoch   3 Batch  192/269 - Train Accuracy:  0.582, Validation Accuracy:  0.577, Loss:  0.762
    Epoch   3 Batch  193/269 - Train Accuracy:  0.587, Validation Accuracy:  0.586, Loss:  0.750
    Epoch   3 Batch  194/269 - Train Accuracy:  0.599, Validation Accuracy:  0.588, Loss:  0.765
    Epoch   3 Batch  195/269 - Train Accuracy:  0.577, Validation Accuracy:  0.578, Loss:  0.765
    Epoch   3 Batch  196/269 - Train Accuracy:  0.559, Validation Accuracy:  0.568, Loss:  0.752
    Epoch   3 Batch  197/269 - Train Accuracy:  0.549, Validation Accuracy:  0.574, Loss:  0.798
    Epoch   3 Batch  198/269 - Train Accuracy:  0.561, Validation Accuracy:  0.581, Loss:  0.800
    Epoch   3 Batch  199/269 - Train Accuracy:  0.574, Validation Accuracy:  0.588, Loss:  0.774
    Epoch   3 Batch  200/269 - Train Accuracy:  0.575, Validation Accuracy:  0.585, Loss:  0.783
    Epoch   3 Batch  201/269 - Train Accuracy:  0.596, Validation Accuracy:  0.589, Loss:  0.756
    Epoch   3 Batch  202/269 - Train Accuracy:  0.577, Validation Accuracy:  0.589, Loss:  0.755
    Epoch   3 Batch  203/269 - Train Accuracy:  0.565, Validation Accuracy:  0.591, Loss:  0.804
    Epoch   3 Batch  204/269 - Train Accuracy:  0.572, Validation Accuracy:  0.591, Loss:  0.779
    Epoch   3 Batch  205/269 - Train Accuracy:  0.580, Validation Accuracy:  0.586, Loss:  0.741
    Epoch   3 Batch  206/269 - Train Accuracy:  0.566, Validation Accuracy:  0.584, Loss:  0.795
    Epoch   3 Batch  207/269 - Train Accuracy:  0.599, Validation Accuracy:  0.581, Loss:  0.726
    Epoch   3 Batch  208/269 - Train Accuracy:  0.568, Validation Accuracy:  0.593, Loss:  0.800
    Epoch   3 Batch  209/269 - Train Accuracy:  0.573, Validation Accuracy:  0.588, Loss:  0.772
    Epoch   3 Batch  210/269 - Train Accuracy:  0.586, Validation Accuracy:  0.582, Loss:  0.742
    Epoch   3 Batch  211/269 - Train Accuracy:  0.589, Validation Accuracy:  0.584, Loss:  0.762
    Epoch   3 Batch  212/269 - Train Accuracy:  0.579, Validation Accuracy:  0.578, Loss:  0.745
    Epoch   3 Batch  213/269 - Train Accuracy:  0.587, Validation Accuracy:  0.584, Loss:  0.747
    Epoch   3 Batch  214/269 - Train Accuracy:  0.590, Validation Accuracy:  0.589, Loss:  0.744
    Epoch   3 Batch  215/269 - Train Accuracy:  0.605, Validation Accuracy:  0.585, Loss:  0.707
    Epoch   3 Batch  216/269 - Train Accuracy:  0.558, Validation Accuracy:  0.584, Loss:  0.800
    Epoch   3 Batch  217/269 - Train Accuracy:  0.553, Validation Accuracy:  0.588, Loss:  0.782
    Epoch   3 Batch  218/269 - Train Accuracy:  0.571, Validation Accuracy:  0.589, Loss:  0.777
    Epoch   3 Batch  219/269 - Train Accuracy:  0.584, Validation Accuracy:  0.582, Loss:  0.780
    Epoch   3 Batch  220/269 - Train Accuracy:  0.585, Validation Accuracy:  0.583, Loss:  0.713
    Epoch   3 Batch  221/269 - Train Accuracy:  0.604, Validation Accuracy:  0.581, Loss:  0.741
    Epoch   3 Batch  222/269 - Train Accuracy:  0.585, Validation Accuracy:  0.582, Loss:  0.714
    Epoch   3 Batch  223/269 - Train Accuracy:  0.579, Validation Accuracy:  0.579, Loss:  0.738
    Epoch   3 Batch  224/269 - Train Accuracy:  0.596, Validation Accuracy:  0.585, Loss:  0.766
    Epoch   3 Batch  225/269 - Train Accuracy:  0.567, Validation Accuracy:  0.584, Loss:  0.758
    Epoch   3 Batch  226/269 - Train Accuracy:  0.576, Validation Accuracy:  0.588, Loss:  0.741
    Epoch   3 Batch  227/269 - Train Accuracy:  0.643, Validation Accuracy:  0.586, Loss:  0.665
    Epoch   3 Batch  228/269 - Train Accuracy:  0.582, Validation Accuracy:  0.586, Loss:  0.736
    Epoch   3 Batch  229/269 - Train Accuracy:  0.579, Validation Accuracy:  0.587, Loss:  0.735
    Epoch   3 Batch  230/269 - Train Accuracy:  0.583, Validation Accuracy:  0.587, Loss:  0.742
    Epoch   3 Batch  231/269 - Train Accuracy:  0.562, Validation Accuracy:  0.590, Loss:  0.792
    Epoch   3 Batch  232/269 - Train Accuracy:  0.549, Validation Accuracy:  0.588, Loss:  0.779
    Epoch   3 Batch  233/269 - Train Accuracy:  0.587, Validation Accuracy:  0.582, Loss:  0.745
    Epoch   3 Batch  234/269 - Train Accuracy:  0.592, Validation Accuracy:  0.587, Loss:  0.740
    Epoch   3 Batch  235/269 - Train Accuracy:  0.597, Validation Accuracy:  0.594, Loss:  0.734
    Epoch   3 Batch  236/269 - Train Accuracy:  0.575, Validation Accuracy:  0.592, Loss:  0.739
    Epoch   3 Batch  237/269 - Train Accuracy:  0.576, Validation Accuracy:  0.585, Loss:  0.735
    Epoch   3 Batch  238/269 - Train Accuracy:  0.602, Validation Accuracy:  0.584, Loss:  0.731
    Epoch   3 Batch  239/269 - Train Accuracy:  0.600, Validation Accuracy:  0.587, Loss:  0.731
    Epoch   3 Batch  240/269 - Train Accuracy:  0.617, Validation Accuracy:  0.592, Loss:  0.671
    Epoch   3 Batch  241/269 - Train Accuracy:  0.587, Validation Accuracy:  0.582, Loss:  0.746
    Epoch   3 Batch  242/269 - Train Accuracy:  0.572, Validation Accuracy:  0.583, Loss:  0.736
    Epoch   3 Batch  243/269 - Train Accuracy:  0.598, Validation Accuracy:  0.585, Loss:  0.721
    Epoch   3 Batch  244/269 - Train Accuracy:  0.584, Validation Accuracy:  0.586, Loss:  0.740
    Epoch   3 Batch  245/269 - Train Accuracy:  0.565, Validation Accuracy:  0.588, Loss:  0.771
    Epoch   3 Batch  246/269 - Train Accuracy:  0.569, Validation Accuracy:  0.585, Loss:  0.738
    Epoch   3 Batch  247/269 - Train Accuracy:  0.587, Validation Accuracy:  0.593, Loss:  0.763
    Epoch   3 Batch  248/269 - Train Accuracy:  0.584, Validation Accuracy:  0.595, Loss:  0.723
    Epoch   3 Batch  249/269 - Train Accuracy:  0.614, Validation Accuracy:  0.591, Loss:  0.697
    Epoch   3 Batch  250/269 - Train Accuracy:  0.569, Validation Accuracy:  0.590, Loss:  0.749
    Epoch   3 Batch  251/269 - Train Accuracy:  0.602, Validation Accuracy:  0.592, Loss:  0.722
    Epoch   3 Batch  252/269 - Train Accuracy:  0.587, Validation Accuracy:  0.587, Loss:  0.742
    Epoch   3 Batch  253/269 - Train Accuracy:  0.575, Validation Accuracy:  0.588, Loss:  0.749
    Epoch   3 Batch  254/269 - Train Accuracy:  0.574, Validation Accuracy:  0.590, Loss:  0.725
    Epoch   3 Batch  255/269 - Train Accuracy:  0.609, Validation Accuracy:  0.588, Loss:  0.695
    Epoch   3 Batch  256/269 - Train Accuracy:  0.576, Validation Accuracy:  0.594, Loss:  0.740
    Epoch   3 Batch  257/269 - Train Accuracy:  0.574, Validation Accuracy:  0.590, Loss:  0.742
    Epoch   3 Batch  258/269 - Train Accuracy:  0.578, Validation Accuracy:  0.591, Loss:  0.733
    Epoch   3 Batch  259/269 - Train Accuracy:  0.607, Validation Accuracy:  0.591, Loss:  0.734
    Epoch   3 Batch  260/269 - Train Accuracy:  0.576, Validation Accuracy:  0.597, Loss:  0.764
    Epoch   3 Batch  261/269 - Train Accuracy:  0.571, Validation Accuracy:  0.598, Loss:  0.774
    Epoch   3 Batch  262/269 - Train Accuracy:  0.600, Validation Accuracy:  0.596, Loss:  0.731
    Epoch   3 Batch  263/269 - Train Accuracy:  0.593, Validation Accuracy:  0.596, Loss:  0.747
    Epoch   3 Batch  264/269 - Train Accuracy:  0.564, Validation Accuracy:  0.594, Loss:  0.766
    Epoch   3 Batch  265/269 - Train Accuracy:  0.571, Validation Accuracy:  0.595, Loss:  0.751
    Epoch   3 Batch  266/269 - Train Accuracy:  0.598, Validation Accuracy:  0.589, Loss:  0.714
    Epoch   3 Batch  267/269 - Train Accuracy:  0.593, Validation Accuracy:  0.586, Loss:  0.736
    Epoch   4 Batch    0/269 - Train Accuracy:  0.564, Validation Accuracy:  0.588, Loss:  0.767
    Epoch   4 Batch    1/269 - Train Accuracy:  0.575, Validation Accuracy:  0.595, Loss:  0.741
    Epoch   4 Batch    2/269 - Train Accuracy:  0.569, Validation Accuracy:  0.593, Loss:  0.737
    Epoch   4 Batch    3/269 - Train Accuracy:  0.581, Validation Accuracy:  0.587, Loss:  0.737
    Epoch   4 Batch    4/269 - Train Accuracy:  0.565, Validation Accuracy:  0.591, Loss:  0.761
    Epoch   4 Batch    5/269 - Train Accuracy:  0.570, Validation Accuracy:  0.594, Loss:  0.745
    Epoch   4 Batch    6/269 - Train Accuracy:  0.596, Validation Accuracy:  0.591, Loss:  0.699
    Epoch   4 Batch    7/269 - Train Accuracy:  0.582, Validation Accuracy:  0.588, Loss:  0.710
    Epoch   4 Batch    8/269 - Train Accuracy:  0.572, Validation Accuracy:  0.588, Loss:  0.761
    Epoch   4 Batch    9/269 - Train Accuracy:  0.579, Validation Accuracy:  0.589, Loss:  0.733
    Epoch   4 Batch   10/269 - Train Accuracy:  0.573, Validation Accuracy:  0.586, Loss:  0.739
    Epoch   4 Batch   11/269 - Train Accuracy:  0.588, Validation Accuracy:  0.589, Loss:  0.733
    Epoch   4 Batch   12/269 - Train Accuracy:  0.567, Validation Accuracy:  0.599, Loss:  0.757
    Epoch   4 Batch   13/269 - Train Accuracy:  0.619, Validation Accuracy:  0.596, Loss:  0.668
    Epoch   4 Batch   14/269 - Train Accuracy:  0.596, Validation Accuracy:  0.595, Loss:  0.715
    Epoch   4 Batch   15/269 - Train Accuracy:  0.585, Validation Accuracy:  0.594, Loss:  0.707
    Epoch   4 Batch   16/269 - Train Accuracy:  0.603, Validation Accuracy:  0.599, Loss:  0.711
    Epoch   4 Batch   17/269 - Train Accuracy:  0.595, Validation Accuracy:  0.599, Loss:  0.705
    Epoch   4 Batch   18/269 - Train Accuracy:  0.563, Validation Accuracy:  0.593, Loss:  0.739
    Epoch   4 Batch   19/269 - Train Accuracy:  0.613, Validation Accuracy:  0.585, Loss:  0.668
    Epoch   4 Batch   20/269 - Train Accuracy:  0.575, Validation Accuracy:  0.589, Loss:  0.740
    Epoch   4 Batch   21/269 - Train Accuracy:  0.582, Validation Accuracy:  0.596, Loss:  0.766
    Epoch   4 Batch   22/269 - Train Accuracy:  0.601, Validation Accuracy:  0.599, Loss:  0.698
    Epoch   4 Batch   23/269 - Train Accuracy:  0.600, Validation Accuracy:  0.594, Loss:  0.710
    Epoch   4 Batch   24/269 - Train Accuracy:  0.586, Validation Accuracy:  0.596, Loss:  0.739
    Epoch   4 Batch   25/269 - Train Accuracy:  0.583, Validation Accuracy:  0.600, Loss:  0.748
    Epoch   4 Batch   26/269 - Train Accuracy:  0.621, Validation Accuracy:  0.599, Loss:  0.662
    Epoch   4 Batch   27/269 - Train Accuracy:  0.590, Validation Accuracy:  0.599, Loss:  0.706
    Epoch   4 Batch   28/269 - Train Accuracy:  0.552, Validation Accuracy:  0.595, Loss:  0.769
    Epoch   4 Batch   29/269 - Train Accuracy:  0.584, Validation Accuracy:  0.596, Loss:  0.733
    Epoch   4 Batch   30/269 - Train Accuracy:  0.600, Validation Accuracy:  0.597, Loss:  0.703
    Epoch   4 Batch   31/269 - Train Accuracy:  0.597, Validation Accuracy:  0.598, Loss:  0.698
    Epoch   4 Batch   32/269 - Train Accuracy:  0.591, Validation Accuracy:  0.595, Loss:  0.702
    Epoch   4 Batch   33/269 - Train Accuracy:  0.600, Validation Accuracy:  0.593, Loss:  0.689
    Epoch   4 Batch   34/269 - Train Accuracy:  0.601, Validation Accuracy:  0.600, Loss:  0.706
    Epoch   4 Batch   35/269 - Train Accuracy:  0.598, Validation Accuracy:  0.600, Loss:  0.725
    Epoch   4 Batch   36/269 - Train Accuracy:  0.581, Validation Accuracy:  0.589, Loss:  0.713
    Epoch   4 Batch   37/269 - Train Accuracy:  0.597, Validation Accuracy:  0.588, Loss:  0.707
    Epoch   4 Batch   38/269 - Train Accuracy:  0.595, Validation Accuracy:  0.595, Loss:  0.707
    Epoch   4 Batch   39/269 - Train Accuracy:  0.598, Validation Accuracy:  0.597, Loss:  0.699
    Epoch   4 Batch   40/269 - Train Accuracy:  0.585, Validation Accuracy:  0.600, Loss:  0.734
    Epoch   4 Batch   41/269 - Train Accuracy:  0.592, Validation Accuracy:  0.602, Loss:  0.721
    Epoch   4 Batch   42/269 - Train Accuracy:  0.618, Validation Accuracy:  0.598, Loss:  0.668
    Epoch   4 Batch   43/269 - Train Accuracy:  0.583, Validation Accuracy:  0.596, Loss:  0.730
    Epoch   4 Batch   44/269 - Train Accuracy:  0.598, Validation Accuracy:  0.592, Loss:  0.709
    Epoch   4 Batch   45/269 - Train Accuracy:  0.563, Validation Accuracy:  0.587, Loss:  0.739
    Epoch   4 Batch   46/269 - Train Accuracy:  0.585, Validation Accuracy:  0.590, Loss:  0.734
    Epoch   4 Batch   47/269 - Train Accuracy:  0.620, Validation Accuracy:  0.589, Loss:  0.657
    Epoch   4 Batch   48/269 - Train Accuracy:  0.593, Validation Accuracy:  0.590, Loss:  0.690
    Epoch   4 Batch   49/269 - Train Accuracy:  0.577, Validation Accuracy:  0.595, Loss:  0.723
    Epoch   4 Batch   50/269 - Train Accuracy:  0.577, Validation Accuracy:  0.598, Loss:  0.735
    Epoch   4 Batch   51/269 - Train Accuracy:  0.591, Validation Accuracy:  0.603, Loss:  0.710
    Epoch   4 Batch   52/269 - Train Accuracy:  0.587, Validation Accuracy:  0.595, Loss:  0.678
    Epoch   4 Batch   53/269 - Train Accuracy:  0.571, Validation Accuracy:  0.591, Loss:  0.739
    Epoch   4 Batch   54/269 - Train Accuracy:  0.583, Validation Accuracy:  0.591, Loss:  0.725
    Epoch   4 Batch   55/269 - Train Accuracy:  0.604, Validation Accuracy:  0.596, Loss:  0.688
    Epoch   4 Batch   56/269 - Train Accuracy:  0.608, Validation Accuracy:  0.597, Loss:  0.699
    Epoch   4 Batch   57/269 - Train Accuracy:  0.594, Validation Accuracy:  0.591, Loss:  0.711
    Epoch   4 Batch   58/269 - Train Accuracy:  0.598, Validation Accuracy:  0.590, Loss:  0.692
    Epoch   4 Batch   59/269 - Train Accuracy:  0.601, Validation Accuracy:  0.596, Loss:  0.675
    Epoch   4 Batch   60/269 - Train Accuracy:  0.611, Validation Accuracy:  0.597, Loss:  0.669
    Epoch   4 Batch   61/269 - Train Accuracy:  0.606, Validation Accuracy:  0.593, Loss:  0.651
    Epoch   4 Batch   62/269 - Train Accuracy:  0.610, Validation Accuracy:  0.593, Loss:  0.677
    Epoch   4 Batch   63/269 - Train Accuracy:  0.581, Validation Accuracy:  0.596, Loss:  0.714
    Epoch   4 Batch   64/269 - Train Accuracy:  0.595, Validation Accuracy:  0.597, Loss:  0.694
    Epoch   4 Batch   65/269 - Train Accuracy:  0.585, Validation Accuracy:  0.588, Loss:  0.694
    Epoch   4 Batch   66/269 - Train Accuracy:  0.602, Validation Accuracy:  0.590, Loss:  0.671
    Epoch   4 Batch   67/269 - Train Accuracy:  0.590, Validation Accuracy:  0.600, Loss:  0.713
    Epoch   4 Batch   68/269 - Train Accuracy:  0.581, Validation Accuracy:  0.600, Loss:  0.695
    Epoch   4 Batch   69/269 - Train Accuracy:  0.567, Validation Accuracy:  0.590, Loss:  0.765
    Epoch   4 Batch   70/269 - Train Accuracy:  0.596, Validation Accuracy:  0.590, Loss:  0.697
    Epoch   4 Batch   71/269 - Train Accuracy:  0.589, Validation Accuracy:  0.601, Loss:  0.726
    Epoch   4 Batch   72/269 - Train Accuracy:  0.607, Validation Accuracy:  0.602, Loss:  0.683
    Epoch   4 Batch   73/269 - Train Accuracy:  0.597, Validation Accuracy:  0.600, Loss:  0.709
    Epoch   4 Batch   74/269 - Train Accuracy:  0.593, Validation Accuracy:  0.592, Loss:  0.700
    Epoch   4 Batch   75/269 - Train Accuracy:  0.597, Validation Accuracy:  0.598, Loss:  0.696
    Epoch   4 Batch   76/269 - Train Accuracy:  0.589, Validation Accuracy:  0.606, Loss:  0.702
    Epoch   4 Batch   77/269 - Train Accuracy:  0.623, Validation Accuracy:  0.604, Loss:  0.686
    Epoch   4 Batch   78/269 - Train Accuracy:  0.603, Validation Accuracy:  0.597, Loss:  0.679
    Epoch   4 Batch   79/269 - Train Accuracy:  0.590, Validation Accuracy:  0.596, Loss:  0.679
    Epoch   4 Batch   80/269 - Train Accuracy:  0.614, Validation Accuracy:  0.605, Loss:  0.686
    Epoch   4 Batch   81/269 - Train Accuracy:  0.600, Validation Accuracy:  0.607, Loss:  0.701
    Epoch   4 Batch   82/269 - Train Accuracy:  0.608, Validation Accuracy:  0.602, Loss:  0.663
    Epoch   4 Batch   83/269 - Train Accuracy:  0.600, Validation Accuracy:  0.597, Loss:  0.702
    Epoch   4 Batch   84/269 - Train Accuracy:  0.616, Validation Accuracy:  0.600, Loss:  0.674
    Epoch   4 Batch   85/269 - Train Accuracy:  0.600, Validation Accuracy:  0.605, Loss:  0.690
    Epoch   4 Batch   86/269 - Train Accuracy:  0.577, Validation Accuracy:  0.602, Loss:  0.685
    Epoch   4 Batch   87/269 - Train Accuracy:  0.571, Validation Accuracy:  0.593, Loss:  0.727
    Epoch   4 Batch   88/269 - Train Accuracy:  0.587, Validation Accuracy:  0.591, Loss:  0.693
    Epoch   4 Batch   89/269 - Train Accuracy:  0.618, Validation Accuracy:  0.596, Loss:  0.691
    Epoch   4 Batch   90/269 - Train Accuracy:  0.562, Validation Accuracy:  0.600, Loss:  0.729
    Epoch   4 Batch   91/269 - Train Accuracy:  0.595, Validation Accuracy:  0.594, Loss:  0.664
    Epoch   4 Batch   92/269 - Train Accuracy:  0.589, Validation Accuracy:  0.597, Loss:  0.674
    Epoch   4 Batch   93/269 - Train Accuracy:  0.616, Validation Accuracy:  0.603, Loss:  0.652
    Epoch   4 Batch   94/269 - Train Accuracy:  0.597, Validation Accuracy:  0.603, Loss:  0.697
    Epoch   4 Batch   95/269 - Train Accuracy:  0.602, Validation Accuracy:  0.603, Loss:  0.692
    Epoch   4 Batch   96/269 - Train Accuracy:  0.609, Validation Accuracy:  0.607, Loss:  0.682
    Epoch   4 Batch   97/269 - Train Accuracy:  0.594, Validation Accuracy:  0.603, Loss:  0.676
    Epoch   4 Batch   98/269 - Train Accuracy:  0.607, Validation Accuracy:  0.603, Loss:  0.682
    Epoch   4 Batch   99/269 - Train Accuracy:  0.584, Validation Accuracy:  0.598, Loss:  0.709
    Epoch   4 Batch  100/269 - Train Accuracy:  0.604, Validation Accuracy:  0.599, Loss:  0.678
    Epoch   4 Batch  101/269 - Train Accuracy:  0.562, Validation Accuracy:  0.591, Loss:  0.719
    Epoch   4 Batch  102/269 - Train Accuracy:  0.602, Validation Accuracy:  0.600, Loss:  0.683
    Epoch   4 Batch  103/269 - Train Accuracy:  0.591, Validation Accuracy:  0.601, Loss:  0.670
    Epoch   4 Batch  104/269 - Train Accuracy:  0.586, Validation Accuracy:  0.602, Loss:  0.684
    Epoch   4 Batch  105/269 - Train Accuracy:  0.598, Validation Accuracy:  0.602, Loss:  0.694
    Epoch   4 Batch  106/269 - Train Accuracy:  0.604, Validation Accuracy:  0.610, Loss:  0.675
    Epoch   4 Batch  107/269 - Train Accuracy:  0.575, Validation Accuracy:  0.607, Loss:  0.718
    Epoch   4 Batch  108/269 - Train Accuracy:  0.596, Validation Accuracy:  0.596, Loss:  0.674
    Epoch   4 Batch  109/269 - Train Accuracy:  0.574, Validation Accuracy:  0.595, Loss:  0.692
    Epoch   4 Batch  110/269 - Train Accuracy:  0.594, Validation Accuracy:  0.602, Loss:  0.674
    Epoch   4 Batch  111/269 - Train Accuracy:  0.567, Validation Accuracy:  0.601, Loss:  0.716
    Epoch   4 Batch  112/269 - Train Accuracy:  0.610, Validation Accuracy:  0.604, Loss:  0.686
    Epoch   4 Batch  113/269 - Train Accuracy:  0.600, Validation Accuracy:  0.598, Loss:  0.643
    Epoch   4 Batch  114/269 - Train Accuracy:  0.588, Validation Accuracy:  0.597, Loss:  0.676
    Epoch   4 Batch  115/269 - Train Accuracy:  0.587, Validation Accuracy:  0.602, Loss:  0.709
    Epoch   4 Batch  116/269 - Train Accuracy:  0.601, Validation Accuracy:  0.605, Loss:  0.689
    Epoch   4 Batch  117/269 - Train Accuracy:  0.584, Validation Accuracy:  0.598, Loss:  0.678
    Epoch   4 Batch  118/269 - Train Accuracy:  0.612, Validation Accuracy:  0.602, Loss:  0.662
    Epoch   4 Batch  119/269 - Train Accuracy:  0.583, Validation Accuracy:  0.602, Loss:  0.710
    Epoch   4 Batch  120/269 - Train Accuracy:  0.589, Validation Accuracy:  0.600, Loss:  0.698
    Epoch   4 Batch  121/269 - Train Accuracy:  0.597, Validation Accuracy:  0.598, Loss:  0.669
    Epoch   4 Batch  122/269 - Train Accuracy:  0.591, Validation Accuracy:  0.600, Loss:  0.667
    Epoch   4 Batch  123/269 - Train Accuracy:  0.580, Validation Accuracy:  0.600, Loss:  0.712
    Epoch   4 Batch  124/269 - Train Accuracy:  0.591, Validation Accuracy:  0.601, Loss:  0.662
    Epoch   4 Batch  125/269 - Train Accuracy:  0.594, Validation Accuracy:  0.601, Loss:  0.670
    Epoch   4 Batch  126/269 - Train Accuracy:  0.603, Validation Accuracy:  0.601, Loss:  0.673
    Epoch   4 Batch  127/269 - Train Accuracy:  0.582, Validation Accuracy:  0.595, Loss:  0.701
    Epoch   4 Batch  128/269 - Train Accuracy:  0.611, Validation Accuracy:  0.603, Loss:  0.675
    Epoch   4 Batch  129/269 - Train Accuracy:  0.603, Validation Accuracy:  0.603, Loss:  0.678
    Epoch   4 Batch  130/269 - Train Accuracy:  0.567, Validation Accuracy:  0.595, Loss:  0.707
    Epoch   4 Batch  131/269 - Train Accuracy:  0.579, Validation Accuracy:  0.600, Loss:  0.692
    Epoch   4 Batch  132/269 - Train Accuracy:  0.600, Validation Accuracy:  0.604, Loss:  0.680
    Epoch   4 Batch  133/269 - Train Accuracy:  0.601, Validation Accuracy:  0.603, Loss:  0.655
    Epoch   4 Batch  134/269 - Train Accuracy:  0.565, Validation Accuracy:  0.591, Loss:  0.695
    Epoch   4 Batch  135/269 - Train Accuracy:  0.569, Validation Accuracy:  0.597, Loss:  0.733
    Epoch   4 Batch  136/269 - Train Accuracy:  0.567, Validation Accuracy:  0.598, Loss:  0.727
    Epoch   4 Batch  137/269 - Train Accuracy:  0.583, Validation Accuracy:  0.597, Loss:  0.707
    Epoch   4 Batch  138/269 - Train Accuracy:  0.579, Validation Accuracy:  0.589, Loss:  0.691
    Epoch   4 Batch  139/269 - Train Accuracy:  0.611, Validation Accuracy:  0.597, Loss:  0.662
    Epoch   4 Batch  140/269 - Train Accuracy:  0.611, Validation Accuracy:  0.598, Loss:  0.690
    Epoch   4 Batch  141/269 - Train Accuracy:  0.592, Validation Accuracy:  0.595, Loss:  0.691
    Epoch   4 Batch  142/269 - Train Accuracy:  0.592, Validation Accuracy:  0.593, Loss:  0.650
    Epoch   4 Batch  143/269 - Train Accuracy:  0.601, Validation Accuracy:  0.603, Loss:  0.667
    Epoch   4 Batch  144/269 - Train Accuracy:  0.605, Validation Accuracy:  0.599, Loss:  0.641
    Epoch   4 Batch  145/269 - Train Accuracy:  0.600, Validation Accuracy:  0.593, Loss:  0.667
    Epoch   4 Batch  146/269 - Train Accuracy:  0.593, Validation Accuracy:  0.594, Loss:  0.653
    Epoch   4 Batch  147/269 - Train Accuracy:  0.611, Validation Accuracy:  0.598, Loss:  0.644
    Epoch   4 Batch  148/269 - Train Accuracy:  0.586, Validation Accuracy:  0.600, Loss:  0.678
    Epoch   4 Batch  149/269 - Train Accuracy:  0.600, Validation Accuracy:  0.604, Loss:  0.680
    Epoch   4 Batch  150/269 - Train Accuracy:  0.606, Validation Accuracy:  0.606, Loss:  0.671
    Epoch   4 Batch  151/269 - Train Accuracy:  0.633, Validation Accuracy:  0.607, Loss:  0.640
    Epoch   4 Batch  152/269 - Train Accuracy:  0.599, Validation Accuracy:  0.607, Loss:  0.672
    Epoch   4 Batch  153/269 - Train Accuracy:  0.612, Validation Accuracy:  0.610, Loss:  0.656
    Epoch   4 Batch  154/269 - Train Accuracy:  0.584, Validation Accuracy:  0.601, Loss:  0.678
    Epoch   4 Batch  155/269 - Train Accuracy:  0.625, Validation Accuracy:  0.603, Loss:  0.636
    Epoch   4 Batch  156/269 - Train Accuracy:  0.590, Validation Accuracy:  0.603, Loss:  0.701
    Epoch   4 Batch  157/269 - Train Accuracy:  0.606, Validation Accuracy:  0.607, Loss:  0.667
    Epoch   4 Batch  158/269 - Train Accuracy:  0.600, Validation Accuracy:  0.605, Loss:  0.668
    Epoch   4 Batch  159/269 - Train Accuracy:  0.598, Validation Accuracy:  0.600, Loss:  0.667
    Epoch   4 Batch  160/269 - Train Accuracy:  0.609, Validation Accuracy:  0.601, Loss:  0.660
    Epoch   4 Batch  161/269 - Train Accuracy:  0.596, Validation Accuracy:  0.605, Loss:  0.672
    Epoch   4 Batch  162/269 - Train Accuracy:  0.600, Validation Accuracy:  0.606, Loss:  0.664
    Epoch   4 Batch  163/269 - Train Accuracy:  0.614, Validation Accuracy:  0.600, Loss:  0.659
    Epoch   4 Batch  164/269 - Train Accuracy:  0.609, Validation Accuracy:  0.600, Loss:  0.652
    Epoch   4 Batch  165/269 - Train Accuracy:  0.579, Validation Accuracy:  0.605, Loss:  0.680
    Epoch   4 Batch  166/269 - Train Accuracy:  0.621, Validation Accuracy:  0.605, Loss:  0.629
    Epoch   4 Batch  167/269 - Train Accuracy:  0.596, Validation Accuracy:  0.604, Loss:  0.671
    Epoch   4 Batch  168/269 - Train Accuracy:  0.591, Validation Accuracy:  0.600, Loss:  0.668
    Epoch   4 Batch  169/269 - Train Accuracy:  0.604, Validation Accuracy:  0.602, Loss:  0.667
    Epoch   4 Batch  170/269 - Train Accuracy:  0.604, Validation Accuracy:  0.613, Loss:  0.653
    Epoch   4 Batch  171/269 - Train Accuracy:  0.602, Validation Accuracy:  0.613, Loss:  0.692
    Epoch   4 Batch  172/269 - Train Accuracy:  0.603, Validation Accuracy:  0.606, Loss:  0.669
    Epoch   4 Batch  173/269 - Train Accuracy:  0.604, Validation Accuracy:  0.609, Loss:  0.645
    Epoch   4 Batch  174/269 - Train Accuracy:  0.585, Validation Accuracy:  0.607, Loss:  0.663
    Epoch   4 Batch  175/269 - Train Accuracy:  0.611, Validation Accuracy:  0.609, Loss:  0.673
    Epoch   4 Batch  176/269 - Train Accuracy:  0.586, Validation Accuracy:  0.604, Loss:  0.700
    Epoch   4 Batch  177/269 - Train Accuracy:  0.611, Validation Accuracy:  0.608, Loss:  0.642
    Epoch   4 Batch  178/269 - Train Accuracy:  0.596, Validation Accuracy:  0.608, Loss:  0.683
    Epoch   4 Batch  179/269 - Train Accuracy:  0.612, Validation Accuracy:  0.605, Loss:  0.659
    Epoch   4 Batch  180/269 - Train Accuracy:  0.600, Validation Accuracy:  0.603, Loss:  0.650
    Epoch   4 Batch  181/269 - Train Accuracy:  0.592, Validation Accuracy:  0.610, Loss:  0.661
    Epoch   4 Batch  182/269 - Train Accuracy:  0.614, Validation Accuracy:  0.609, Loss:  0.659
    Epoch   4 Batch  183/269 - Train Accuracy:  0.661, Validation Accuracy:  0.602, Loss:  0.575
    Epoch   4 Batch  184/269 - Train Accuracy:  0.580, Validation Accuracy:  0.594, Loss:  0.682
    Epoch   4 Batch  185/269 - Train Accuracy:  0.608, Validation Accuracy:  0.603, Loss:  0.656
    Epoch   4 Batch  186/269 - Train Accuracy:  0.574, Validation Accuracy:  0.603, Loss:  0.679
    Epoch   4 Batch  187/269 - Train Accuracy:  0.600, Validation Accuracy:  0.598, Loss:  0.645
    Epoch   4 Batch  188/269 - Train Accuracy:  0.601, Validation Accuracy:  0.598, Loss:  0.636
    Epoch   4 Batch  189/269 - Train Accuracy:  0.601, Validation Accuracy:  0.602, Loss:  0.638
    Epoch   4 Batch  190/269 - Train Accuracy:  0.588, Validation Accuracy:  0.603, Loss:  0.641
    Epoch   4 Batch  191/269 - Train Accuracy:  0.614, Validation Accuracy:  0.603, Loss:  0.645
    Epoch   4 Batch  192/269 - Train Accuracy:  0.600, Validation Accuracy:  0.597, Loss:  0.653
    Epoch   4 Batch  193/269 - Train Accuracy:  0.602, Validation Accuracy:  0.604, Loss:  0.653
    Epoch   4 Batch  194/269 - Train Accuracy:  0.617, Validation Accuracy:  0.605, Loss:  0.657
    Epoch   4 Batch  195/269 - Train Accuracy:  0.605, Validation Accuracy:  0.606, Loss:  0.651
    Epoch   4 Batch  196/269 - Train Accuracy:  0.584, Validation Accuracy:  0.602, Loss:  0.647
    Epoch   4 Batch  197/269 - Train Accuracy:  0.593, Validation Accuracy:  0.614, Loss:  0.682
    Epoch   4 Batch  198/269 - Train Accuracy:  0.598, Validation Accuracy:  0.612, Loss:  0.688
    Epoch   4 Batch  199/269 - Train Accuracy:  0.592, Validation Accuracy:  0.608, Loss:  0.678
    Epoch   4 Batch  200/269 - Train Accuracy:  0.599, Validation Accuracy:  0.610, Loss:  0.680
    Epoch   4 Batch  201/269 - Train Accuracy:  0.613, Validation Accuracy:  0.610, Loss:  0.648
    Epoch   4 Batch  202/269 - Train Accuracy:  0.601, Validation Accuracy:  0.615, Loss:  0.655
    Epoch   4 Batch  203/269 - Train Accuracy:  0.589, Validation Accuracy:  0.613, Loss:  0.696
    Epoch   4 Batch  204/269 - Train Accuracy:  0.592, Validation Accuracy:  0.611, Loss:  0.676
    Epoch   4 Batch  205/269 - Train Accuracy:  0.615, Validation Accuracy:  0.617, Loss:  0.646
    Epoch   4 Batch  206/269 - Train Accuracy:  0.592, Validation Accuracy:  0.617, Loss:  0.676
    Epoch   4 Batch  207/269 - Train Accuracy:  0.627, Validation Accuracy:  0.607, Loss:  0.628
    Epoch   4 Batch  208/269 - Train Accuracy:  0.584, Validation Accuracy:  0.603, Loss:  0.683
    Epoch   4 Batch  209/269 - Train Accuracy:  0.603, Validation Accuracy:  0.605, Loss:  0.658
    Epoch   4 Batch  210/269 - Train Accuracy:  0.614, Validation Accuracy:  0.603, Loss:  0.645
    Epoch   4 Batch  211/269 - Train Accuracy:  0.614, Validation Accuracy:  0.609, Loss:  0.653
    Epoch   4 Batch  212/269 - Train Accuracy:  0.614, Validation Accuracy:  0.606, Loss:  0.646
    Epoch   4 Batch  213/269 - Train Accuracy:  0.608, Validation Accuracy:  0.605, Loss:  0.642
    Epoch   4 Batch  214/269 - Train Accuracy:  0.605, Validation Accuracy:  0.600, Loss:  0.641
    Epoch   4 Batch  215/269 - Train Accuracy:  0.620, Validation Accuracy:  0.600, Loss:  0.614
    Epoch   4 Batch  216/269 - Train Accuracy:  0.584, Validation Accuracy:  0.602, Loss:  0.689
    Epoch   4 Batch  217/269 - Train Accuracy:  0.571, Validation Accuracy:  0.600, Loss:  0.670
    Epoch   4 Batch  218/269 - Train Accuracy:  0.599, Validation Accuracy:  0.603, Loss:  0.669
    Epoch   4 Batch  219/269 - Train Accuracy:  0.603, Validation Accuracy:  0.603, Loss:  0.674
    Epoch   4 Batch  220/269 - Train Accuracy:  0.605, Validation Accuracy:  0.603, Loss:  0.613
    Epoch   4 Batch  221/269 - Train Accuracy:  0.630, Validation Accuracy:  0.601, Loss:  0.637
    Epoch   4 Batch  222/269 - Train Accuracy:  0.616, Validation Accuracy:  0.610, Loss:  0.626
    Epoch   4 Batch  223/269 - Train Accuracy:  0.596, Validation Accuracy:  0.606, Loss:  0.639
    Epoch   4 Batch  224/269 - Train Accuracy:  0.616, Validation Accuracy:  0.603, Loss:  0.663
    Epoch   4 Batch  225/269 - Train Accuracy:  0.595, Validation Accuracy:  0.599, Loss:  0.652
    Epoch   4 Batch  226/269 - Train Accuracy:  0.605, Validation Accuracy:  0.606, Loss:  0.641
    Epoch   4 Batch  227/269 - Train Accuracy:  0.665, Validation Accuracy:  0.607, Loss:  0.578
    Epoch   4 Batch  228/269 - Train Accuracy:  0.611, Validation Accuracy:  0.603, Loss:  0.640
    Epoch   4 Batch  229/269 - Train Accuracy:  0.599, Validation Accuracy:  0.600, Loss:  0.635
    Epoch   4 Batch  230/269 - Train Accuracy:  0.594, Validation Accuracy:  0.604, Loss:  0.646
    Epoch   4 Batch  231/269 - Train Accuracy:  0.581, Validation Accuracy:  0.606, Loss:  0.683
    Epoch   4 Batch  232/269 - Train Accuracy:  0.573, Validation Accuracy:  0.605, Loss:  0.675
    Epoch   4 Batch  233/269 - Train Accuracy:  0.607, Validation Accuracy:  0.602, Loss:  0.649
    Epoch   4 Batch  234/269 - Train Accuracy:  0.611, Validation Accuracy:  0.609, Loss:  0.634
    Epoch   4 Batch  235/269 - Train Accuracy:  0.611, Validation Accuracy:  0.610, Loss:  0.626
    Epoch   4 Batch  236/269 - Train Accuracy:  0.590, Validation Accuracy:  0.605, Loss:  0.637
    Epoch   4 Batch  237/269 - Train Accuracy:  0.593, Validation Accuracy:  0.605, Loss:  0.641
    Epoch   4 Batch  238/269 - Train Accuracy:  0.619, Validation Accuracy:  0.606, Loss:  0.630
    Epoch   4 Batch  239/269 - Train Accuracy:  0.616, Validation Accuracy:  0.606, Loss:  0.633
    Epoch   4 Batch  240/269 - Train Accuracy:  0.637, Validation Accuracy:  0.609, Loss:  0.580
    Epoch   4 Batch  241/269 - Train Accuracy:  0.614, Validation Accuracy:  0.613, Loss:  0.650
    Epoch   4 Batch  242/269 - Train Accuracy:  0.600, Validation Accuracy:  0.611, Loss:  0.639
    Epoch   4 Batch  243/269 - Train Accuracy:  0.619, Validation Accuracy:  0.609, Loss:  0.616
    Epoch   4 Batch  244/269 - Train Accuracy:  0.604, Validation Accuracy:  0.607, Loss:  0.641
    Epoch   4 Batch  245/269 - Train Accuracy:  0.591, Validation Accuracy:  0.608, Loss:  0.666
    Epoch   4 Batch  246/269 - Train Accuracy:  0.590, Validation Accuracy:  0.613, Loss:  0.641
    Epoch   4 Batch  247/269 - Train Accuracy:  0.615, Validation Accuracy:  0.615, Loss:  0.655
    Epoch   4 Batch  248/269 - Train Accuracy:  0.602, Validation Accuracy:  0.611, Loss:  0.626
    Epoch   4 Batch  249/269 - Train Accuracy:  0.639, Validation Accuracy:  0.609, Loss:  0.599
    Epoch   4 Batch  250/269 - Train Accuracy:  0.592, Validation Accuracy:  0.611, Loss:  0.645
    Epoch   4 Batch  251/269 - Train Accuracy:  0.630, Validation Accuracy:  0.618, Loss:  0.619
    Epoch   4 Batch  252/269 - Train Accuracy:  0.616, Validation Accuracy:  0.613, Loss:  0.639
    Epoch   4 Batch  253/269 - Train Accuracy:  0.599, Validation Accuracy:  0.612, Loss:  0.650
    Epoch   4 Batch  254/269 - Train Accuracy:  0.603, Validation Accuracy:  0.612, Loss:  0.628
    Epoch   4 Batch  255/269 - Train Accuracy:  0.629, Validation Accuracy:  0.612, Loss:  0.613
    Epoch   4 Batch  256/269 - Train Accuracy:  0.590, Validation Accuracy:  0.607, Loss:  0.648
    Epoch   4 Batch  257/269 - Train Accuracy:  0.591, Validation Accuracy:  0.615, Loss:  0.649
    Epoch   4 Batch  258/269 - Train Accuracy:  0.607, Validation Accuracy:  0.618, Loss:  0.641
    Epoch   4 Batch  259/269 - Train Accuracy:  0.629, Validation Accuracy:  0.617, Loss:  0.633
    Epoch   4 Batch  260/269 - Train Accuracy:  0.605, Validation Accuracy:  0.616, Loss:  0.661
    Epoch   4 Batch  261/269 - Train Accuracy:  0.582, Validation Accuracy:  0.620, Loss:  0.671
    Epoch   4 Batch  262/269 - Train Accuracy:  0.626, Validation Accuracy:  0.615, Loss:  0.639
    Epoch   4 Batch  263/269 - Train Accuracy:  0.618, Validation Accuracy:  0.613, Loss:  0.650
    Epoch   4 Batch  264/269 - Train Accuracy:  0.579, Validation Accuracy:  0.612, Loss:  0.664
    Epoch   4 Batch  265/269 - Train Accuracy:  0.597, Validation Accuracy:  0.615, Loss:  0.655
    Epoch   4 Batch  266/269 - Train Accuracy:  0.621, Validation Accuracy:  0.616, Loss:  0.619
    Epoch   4 Batch  267/269 - Train Accuracy:  0.613, Validation Accuracy:  0.613, Loss:  0.637
    Epoch   5 Batch    0/269 - Train Accuracy:  0.585, Validation Accuracy:  0.615, Loss:  0.671
    Epoch   5 Batch    1/269 - Train Accuracy:  0.594, Validation Accuracy:  0.617, Loss:  0.643
    Epoch   5 Batch    2/269 - Train Accuracy:  0.586, Validation Accuracy:  0.613, Loss:  0.647
    Epoch   5 Batch    3/269 - Train Accuracy:  0.598, Validation Accuracy:  0.609, Loss:  0.640
    Epoch   5 Batch    4/269 - Train Accuracy:  0.587, Validation Accuracy:  0.616, Loss:  0.665
    Epoch   5 Batch    5/269 - Train Accuracy:  0.581, Validation Accuracy:  0.619, Loss:  0.651
    Epoch   5 Batch    6/269 - Train Accuracy:  0.610, Validation Accuracy:  0.613, Loss:  0.607
    Epoch   5 Batch    7/269 - Train Accuracy:  0.601, Validation Accuracy:  0.611, Loss:  0.617
    Epoch   5 Batch    8/269 - Train Accuracy:  0.594, Validation Accuracy:  0.618, Loss:  0.668
    Epoch   5 Batch    9/269 - Train Accuracy:  0.601, Validation Accuracy:  0.613, Loss:  0.643
    Epoch   5 Batch   10/269 - Train Accuracy:  0.590, Validation Accuracy:  0.606, Loss:  0.643
    Epoch   5 Batch   11/269 - Train Accuracy:  0.597, Validation Accuracy:  0.608, Loss:  0.640
    Epoch   5 Batch   12/269 - Train Accuracy:  0.583, Validation Accuracy:  0.613, Loss:  0.668
    Epoch   5 Batch   13/269 - Train Accuracy:  0.633, Validation Accuracy:  0.609, Loss:  0.586
    Epoch   5 Batch   14/269 - Train Accuracy:  0.606, Validation Accuracy:  0.605, Loss:  0.623
    Epoch   5 Batch   15/269 - Train Accuracy:  0.601, Validation Accuracy:  0.613, Loss:  0.628
    Epoch   5 Batch   16/269 - Train Accuracy:  0.620, Validation Accuracy:  0.615, Loss:  0.633
    Epoch   5 Batch   17/269 - Train Accuracy:  0.608, Validation Accuracy:  0.607, Loss:  0.620
    Epoch   5 Batch   18/269 - Train Accuracy:  0.575, Validation Accuracy:  0.599, Loss:  0.650
    Epoch   5 Batch   19/269 - Train Accuracy:  0.632, Validation Accuracy:  0.610, Loss:  0.588
    Epoch   5 Batch   20/269 - Train Accuracy:  0.600, Validation Accuracy:  0.612, Loss:  0.646
    Epoch   5 Batch   21/269 - Train Accuracy:  0.596, Validation Accuracy:  0.609, Loss:  0.676
    Epoch   5 Batch   22/269 - Train Accuracy:  0.611, Validation Accuracy:  0.607, Loss:  0.609
    Epoch   5 Batch   23/269 - Train Accuracy:  0.623, Validation Accuracy:  0.617, Loss:  0.617
    Epoch   5 Batch   24/269 - Train Accuracy:  0.615, Validation Accuracy:  0.624, Loss:  0.654
    Epoch   5 Batch   25/269 - Train Accuracy:  0.597, Validation Accuracy:  0.617, Loss:  0.660
    Epoch   5 Batch   26/269 - Train Accuracy:  0.643, Validation Accuracy:  0.617, Loss:  0.577
    Epoch   5 Batch   27/269 - Train Accuracy:  0.616, Validation Accuracy:  0.625, Loss:  0.626
    Epoch   5 Batch   28/269 - Train Accuracy:  0.573, Validation Accuracy:  0.624, Loss:  0.670
    Epoch   5 Batch   29/269 - Train Accuracy:  0.609, Validation Accuracy:  0.620, Loss:  0.646
    Epoch   5 Batch   30/269 - Train Accuracy:  0.621, Validation Accuracy:  0.619, Loss:  0.618
    Epoch   5 Batch   31/269 - Train Accuracy:  0.619, Validation Accuracy:  0.625, Loss:  0.614
    Epoch   5 Batch   32/269 - Train Accuracy:  0.618, Validation Accuracy:  0.628, Loss:  0.611
    Epoch   5 Batch   33/269 - Train Accuracy:  0.630, Validation Accuracy:  0.625, Loss:  0.599
    Epoch   5 Batch   34/269 - Train Accuracy:  0.625, Validation Accuracy:  0.623, Loss:  0.611
    Epoch   5 Batch   35/269 - Train Accuracy:  0.623, Validation Accuracy:  0.624, Loss:  0.635
    Epoch   5 Batch   36/269 - Train Accuracy:  0.614, Validation Accuracy:  0.618, Loss:  0.616
    Epoch   5 Batch   37/269 - Train Accuracy:  0.618, Validation Accuracy:  0.615, Loss:  0.612
    Epoch   5 Batch   38/269 - Train Accuracy:  0.614, Validation Accuracy:  0.619, Loss:  0.617
    Epoch   5 Batch   39/269 - Train Accuracy:  0.619, Validation Accuracy:  0.622, Loss:  0.616
    Epoch   5 Batch   40/269 - Train Accuracy:  0.606, Validation Accuracy:  0.624, Loss:  0.639
    Epoch   5 Batch   41/269 - Train Accuracy:  0.612, Validation Accuracy:  0.624, Loss:  0.627
    Epoch   5 Batch   42/269 - Train Accuracy:  0.645, Validation Accuracy:  0.627, Loss:  0.586
    Epoch   5 Batch   43/269 - Train Accuracy:  0.614, Validation Accuracy:  0.628, Loss:  0.638
    Epoch   5 Batch   44/269 - Train Accuracy:  0.632, Validation Accuracy:  0.630, Loss:  0.621
    Epoch   5 Batch   45/269 - Train Accuracy:  0.597, Validation Accuracy:  0.618, Loss:  0.647
    Epoch   5 Batch   46/269 - Train Accuracy:  0.609, Validation Accuracy:  0.612, Loss:  0.635
    Epoch   5 Batch   47/269 - Train Accuracy:  0.642, Validation Accuracy:  0.612, Loss:  0.573
    Epoch   5 Batch   48/269 - Train Accuracy:  0.619, Validation Accuracy:  0.616, Loss:  0.598
    Epoch   5 Batch   49/269 - Train Accuracy:  0.599, Validation Accuracy:  0.616, Loss:  0.628
    Epoch   5 Batch   50/269 - Train Accuracy:  0.603, Validation Accuracy:  0.613, Loss:  0.640
    Epoch   5 Batch   51/269 - Train Accuracy:  0.600, Validation Accuracy:  0.613, Loss:  0.621
    Epoch   5 Batch   52/269 - Train Accuracy:  0.607, Validation Accuracy:  0.615, Loss:  0.591
    Epoch   5 Batch   53/269 - Train Accuracy:  0.598, Validation Accuracy:  0.615, Loss:  0.647
    Epoch   5 Batch   54/269 - Train Accuracy:  0.598, Validation Accuracy:  0.610, Loss:  0.632
    Epoch   5 Batch   55/269 - Train Accuracy:  0.626, Validation Accuracy:  0.619, Loss:  0.604
    Epoch   5 Batch   56/269 - Train Accuracy:  0.626, Validation Accuracy:  0.616, Loss:  0.608
    Epoch   5 Batch   57/269 - Train Accuracy:  0.624, Validation Accuracy:  0.615, Loss:  0.619
    Epoch   5 Batch   58/269 - Train Accuracy:  0.615, Validation Accuracy:  0.613, Loss:  0.609
    Epoch   5 Batch   59/269 - Train Accuracy:  0.625, Validation Accuracy:  0.619, Loss:  0.588
    Epoch   5 Batch   60/269 - Train Accuracy:  0.635, Validation Accuracy:  0.618, Loss:  0.585
    Epoch   5 Batch   61/269 - Train Accuracy:  0.634, Validation Accuracy:  0.623, Loss:  0.571
    Epoch   5 Batch   62/269 - Train Accuracy:  0.633, Validation Accuracy:  0.620, Loss:  0.588
    Epoch   5 Batch   63/269 - Train Accuracy:  0.609, Validation Accuracy:  0.627, Loss:  0.621
    Epoch   5 Batch   64/269 - Train Accuracy:  0.613, Validation Accuracy:  0.630, Loss:  0.599
    Epoch   5 Batch   65/269 - Train Accuracy:  0.614, Validation Accuracy:  0.621, Loss:  0.607
    Epoch   5 Batch   66/269 - Train Accuracy:  0.622, Validation Accuracy:  0.618, Loss:  0.586
    Epoch   5 Batch   67/269 - Train Accuracy:  0.613, Validation Accuracy:  0.621, Loss:  0.631
    Epoch   5 Batch   68/269 - Train Accuracy:  0.596, Validation Accuracy:  0.619, Loss:  0.617
    Epoch   5 Batch   69/269 - Train Accuracy:  0.588, Validation Accuracy:  0.612, Loss:  0.664
    Epoch   5 Batch   70/269 - Train Accuracy:  0.629, Validation Accuracy:  0.618, Loss:  0.614
    Epoch   5 Batch   71/269 - Train Accuracy:  0.614, Validation Accuracy:  0.620, Loss:  0.629
    Epoch   5 Batch   72/269 - Train Accuracy:  0.615, Validation Accuracy:  0.614, Loss:  0.600
    Epoch   5 Batch   73/269 - Train Accuracy:  0.625, Validation Accuracy:  0.622, Loss:  0.628
    Epoch   5 Batch   74/269 - Train Accuracy:  0.620, Validation Accuracy:  0.617, Loss:  0.610
    Epoch   5 Batch   75/269 - Train Accuracy:  0.612, Validation Accuracy:  0.622, Loss:  0.608
    Epoch   5 Batch   76/269 - Train Accuracy:  0.609, Validation Accuracy:  0.626, Loss:  0.609
    Epoch   5 Batch   77/269 - Train Accuracy:  0.639, Validation Accuracy:  0.628, Loss:  0.602
    Epoch   5 Batch   78/269 - Train Accuracy:  0.634, Validation Accuracy:  0.629, Loss:  0.598
    Epoch   5 Batch   79/269 - Train Accuracy:  0.619, Validation Accuracy:  0.623, Loss:  0.595
    Epoch   5 Batch   80/269 - Train Accuracy:  0.631, Validation Accuracy:  0.622, Loss:  0.605
    Epoch   5 Batch   81/269 - Train Accuracy:  0.629, Validation Accuracy:  0.626, Loss:  0.618
    Epoch   5 Batch   82/269 - Train Accuracy:  0.636, Validation Accuracy:  0.630, Loss:  0.580
    Epoch   5 Batch   83/269 - Train Accuracy:  0.634, Validation Accuracy:  0.629, Loss:  0.619
    Epoch   5 Batch   84/269 - Train Accuracy:  0.626, Validation Accuracy:  0.622, Loss:  0.592
    Epoch   5 Batch   85/269 - Train Accuracy:  0.626, Validation Accuracy:  0.627, Loss:  0.609
    Epoch   5 Batch   86/269 - Train Accuracy:  0.599, Validation Accuracy:  0.623, Loss:  0.599
    Epoch   5 Batch   87/269 - Train Accuracy:  0.595, Validation Accuracy:  0.619, Loss:  0.638
    Epoch   5 Batch   88/269 - Train Accuracy:  0.619, Validation Accuracy:  0.620, Loss:  0.602
    Epoch   5 Batch   89/269 - Train Accuracy:  0.638, Validation Accuracy:  0.622, Loss:  0.606
    Epoch   5 Batch   90/269 - Train Accuracy:  0.582, Validation Accuracy:  0.621, Loss:  0.639
    Epoch   5 Batch   91/269 - Train Accuracy:  0.612, Validation Accuracy:  0.616, Loss:  0.585
    Epoch   5 Batch   92/269 - Train Accuracy:  0.617, Validation Accuracy:  0.623, Loss:  0.590
    Epoch   5 Batch   93/269 - Train Accuracy:  0.639, Validation Accuracy:  0.623, Loss:  0.575
    Epoch   5 Batch   94/269 - Train Accuracy:  0.607, Validation Accuracy:  0.621, Loss:  0.615
    Epoch   5 Batch   95/269 - Train Accuracy:  0.619, Validation Accuracy:  0.626, Loss:  0.613
    Epoch   5 Batch   96/269 - Train Accuracy:  0.634, Validation Accuracy:  0.626, Loss:  0.595
    Epoch   5 Batch   97/269 - Train Accuracy:  0.613, Validation Accuracy:  0.620, Loss:  0.595
    Epoch   5 Batch   98/269 - Train Accuracy:  0.623, Validation Accuracy:  0.625, Loss:  0.599
    Epoch   5 Batch   99/269 - Train Accuracy:  0.618, Validation Accuracy:  0.625, Loss:  0.618
    Epoch   5 Batch  100/269 - Train Accuracy:  0.626, Validation Accuracy:  0.622, Loss:  0.591
    Epoch   5 Batch  101/269 - Train Accuracy:  0.596, Validation Accuracy:  0.623, Loss:  0.635
    Epoch   5 Batch  102/269 - Train Accuracy:  0.626, Validation Accuracy:  0.627, Loss:  0.593
    Epoch   5 Batch  103/269 - Train Accuracy:  0.622, Validation Accuracy:  0.631, Loss:  0.584
    Epoch   5 Batch  104/269 - Train Accuracy:  0.613, Validation Accuracy:  0.630, Loss:  0.598
    Epoch   5 Batch  105/269 - Train Accuracy:  0.626, Validation Accuracy:  0.632, Loss:  0.603
    Epoch   5 Batch  106/269 - Train Accuracy:  0.621, Validation Accuracy:  0.631, Loss:  0.586
    Epoch   5 Batch  107/269 - Train Accuracy:  0.588, Validation Accuracy:  0.624, Loss:  0.625
    Epoch   5 Batch  108/269 - Train Accuracy:  0.626, Validation Accuracy:  0.619, Loss:  0.591
    Epoch   5 Batch  109/269 - Train Accuracy:  0.604, Validation Accuracy:  0.623, Loss:  0.598
    Epoch   5 Batch  110/269 - Train Accuracy:  0.618, Validation Accuracy:  0.619, Loss:  0.590
    Epoch   5 Batch  111/269 - Train Accuracy:  0.591, Validation Accuracy:  0.620, Loss:  0.625
    Epoch   5 Batch  112/269 - Train Accuracy:  0.633, Validation Accuracy:  0.627, Loss:  0.599
    Epoch   5 Batch  113/269 - Train Accuracy:  0.626, Validation Accuracy:  0.625, Loss:  0.565
    Epoch   5 Batch  114/269 - Train Accuracy:  0.616, Validation Accuracy:  0.621, Loss:  0.586
    Epoch   5 Batch  115/269 - Train Accuracy:  0.610, Validation Accuracy:  0.622, Loss:  0.627
    Epoch   5 Batch  116/269 - Train Accuracy:  0.614, Validation Accuracy:  0.623, Loss:  0.599
    Epoch   5 Batch  117/269 - Train Accuracy:  0.615, Validation Accuracy:  0.622, Loss:  0.589
    Epoch   5 Batch  118/269 - Train Accuracy:  0.637, Validation Accuracy:  0.630, Loss:  0.575
    Epoch   5 Batch  119/269 - Train Accuracy:  0.615, Validation Accuracy:  0.630, Loss:  0.617
    Epoch   5 Batch  120/269 - Train Accuracy:  0.618, Validation Accuracy:  0.631, Loss:  0.607
    Epoch   5 Batch  121/269 - Train Accuracy:  0.626, Validation Accuracy:  0.627, Loss:  0.584
    Epoch   5 Batch  122/269 - Train Accuracy:  0.628, Validation Accuracy:  0.627, Loss:  0.584
    Epoch   5 Batch  123/269 - Train Accuracy:  0.606, Validation Accuracy:  0.630, Loss:  0.618
    Epoch   5 Batch  124/269 - Train Accuracy:  0.623, Validation Accuracy:  0.633, Loss:  0.577
    Epoch   5 Batch  125/269 - Train Accuracy:  0.633, Validation Accuracy:  0.633, Loss:  0.578
    Epoch   5 Batch  126/269 - Train Accuracy:  0.641, Validation Accuracy:  0.637, Loss:  0.586
    Epoch   5 Batch  127/269 - Train Accuracy:  0.618, Validation Accuracy:  0.635, Loss:  0.606
    Epoch   5 Batch  128/269 - Train Accuracy:  0.646, Validation Accuracy:  0.634, Loss:  0.591
    Epoch   5 Batch  129/269 - Train Accuracy:  0.630, Validation Accuracy:  0.633, Loss:  0.591
    Epoch   5 Batch  130/269 - Train Accuracy:  0.613, Validation Accuracy:  0.634, Loss:  0.612
    Epoch   5 Batch  131/269 - Train Accuracy:  0.615, Validation Accuracy:  0.630, Loss:  0.605
    Epoch   5 Batch  132/269 - Train Accuracy:  0.627, Validation Accuracy:  0.628, Loss:  0.589
    Epoch   5 Batch  133/269 - Train Accuracy:  0.621, Validation Accuracy:  0.624, Loss:  0.573
    Epoch   5 Batch  134/269 - Train Accuracy:  0.600, Validation Accuracy:  0.623, Loss:  0.602
    Epoch   5 Batch  135/269 - Train Accuracy:  0.594, Validation Accuracy:  0.625, Loss:  0.632
    Epoch   5 Batch  136/269 - Train Accuracy:  0.589, Validation Accuracy:  0.622, Loss:  0.630
    Epoch   5 Batch  137/269 - Train Accuracy:  0.611, Validation Accuracy:  0.625, Loss:  0.614
    Epoch   5 Batch  138/269 - Train Accuracy:  0.608, Validation Accuracy:  0.620, Loss:  0.605
    Epoch   5 Batch  139/269 - Train Accuracy:  0.636, Validation Accuracy:  0.621, Loss:  0.569
    Epoch   5 Batch  140/269 - Train Accuracy:  0.632, Validation Accuracy:  0.627, Loss:  0.604
    Epoch   5 Batch  141/269 - Train Accuracy:  0.626, Validation Accuracy:  0.625, Loss:  0.598
    Epoch   5 Batch  142/269 - Train Accuracy:  0.621, Validation Accuracy:  0.625, Loss:  0.572
    Epoch   5 Batch  143/269 - Train Accuracy:  0.626, Validation Accuracy:  0.626, Loss:  0.585
    Epoch   5 Batch  144/269 - Train Accuracy:  0.640, Validation Accuracy:  0.629, Loss:  0.558
    Epoch   5 Batch  145/269 - Train Accuracy:  0.643, Validation Accuracy:  0.634, Loss:  0.579
    Epoch   5 Batch  146/269 - Train Accuracy:  0.629, Validation Accuracy:  0.630, Loss:  0.564
    Epoch   5 Batch  147/269 - Train Accuracy:  0.636, Validation Accuracy:  0.627, Loss:  0.561
    Epoch   5 Batch  148/269 - Train Accuracy:  0.617, Validation Accuracy:  0.630, Loss:  0.589
    Epoch   5 Batch  149/269 - Train Accuracy:  0.632, Validation Accuracy:  0.626, Loss:  0.595
    Epoch   5 Batch  150/269 - Train Accuracy:  0.641, Validation Accuracy:  0.627, Loss:  0.576
    Epoch   5 Batch  151/269 - Train Accuracy:  0.654, Validation Accuracy:  0.627, Loss:  0.558
    Epoch   5 Batch  152/269 - Train Accuracy:  0.627, Validation Accuracy:  0.630, Loss:  0.584
    Epoch   5 Batch  153/269 - Train Accuracy:  0.641, Validation Accuracy:  0.630, Loss:  0.571
    Epoch   5 Batch  154/269 - Train Accuracy:  0.616, Validation Accuracy:  0.631, Loss:  0.595
    Epoch   5 Batch  155/269 - Train Accuracy:  0.661, Validation Accuracy:  0.629, Loss:  0.554
    Epoch   5 Batch  156/269 - Train Accuracy:  0.615, Validation Accuracy:  0.628, Loss:  0.609
    Epoch   5 Batch  157/269 - Train Accuracy:  0.634, Validation Accuracy:  0.631, Loss:  0.576
    Epoch   5 Batch  158/269 - Train Accuracy:  0.625, Validation Accuracy:  0.630, Loss:  0.580
    Epoch   5 Batch  159/269 - Train Accuracy:  0.629, Validation Accuracy:  0.630, Loss:  0.577
    Epoch   5 Batch  160/269 - Train Accuracy:  0.637, Validation Accuracy:  0.632, Loss:  0.569
    Epoch   5 Batch  161/269 - Train Accuracy:  0.632, Validation Accuracy:  0.636, Loss:  0.580
    Epoch   5 Batch  162/269 - Train Accuracy:  0.634, Validation Accuracy:  0.635, Loss:  0.575
    Epoch   5 Batch  163/269 - Train Accuracy:  0.652, Validation Accuracy:  0.637, Loss:  0.575
    Epoch   5 Batch  164/269 - Train Accuracy:  0.641, Validation Accuracy:  0.639, Loss:  0.569
    Epoch   5 Batch  165/269 - Train Accuracy:  0.620, Validation Accuracy:  0.640, Loss:  0.589
    Epoch   5 Batch  166/269 - Train Accuracy:  0.652, Validation Accuracy:  0.638, Loss:  0.543
    Epoch   5 Batch  167/269 - Train Accuracy:  0.634, Validation Accuracy:  0.636, Loss:  0.579
    Epoch   5 Batch  168/269 - Train Accuracy:  0.628, Validation Accuracy:  0.636, Loss:  0.583
    Epoch   5 Batch  169/269 - Train Accuracy:  0.636, Validation Accuracy:  0.638, Loss:  0.576
    Epoch   5 Batch  170/269 - Train Accuracy:  0.635, Validation Accuracy:  0.640, Loss:  0.565
    Epoch   5 Batch  171/269 - Train Accuracy:  0.637, Validation Accuracy:  0.638, Loss:  0.598
    Epoch   5 Batch  172/269 - Train Accuracy:  0.642, Validation Accuracy:  0.645, Loss:  0.587
    Epoch   5 Batch  173/269 - Train Accuracy:  0.634, Validation Accuracy:  0.637, Loss:  0.561
    Epoch   5 Batch  174/269 - Train Accuracy:  0.621, Validation Accuracy:  0.640, Loss:  0.578
    Epoch   5 Batch  175/269 - Train Accuracy:  0.645, Validation Accuracy:  0.642, Loss:  0.587
    Epoch   5 Batch  176/269 - Train Accuracy:  0.624, Validation Accuracy:  0.638, Loss:  0.607
    Epoch   5 Batch  177/269 - Train Accuracy:  0.636, Validation Accuracy:  0.640, Loss:  0.562
    Epoch   5 Batch  178/269 - Train Accuracy:  0.624, Validation Accuracy:  0.637, Loss:  0.595
    Epoch   5 Batch  179/269 - Train Accuracy:  0.642, Validation Accuracy:  0.632, Loss:  0.568
    Epoch   5 Batch  180/269 - Train Accuracy:  0.636, Validation Accuracy:  0.631, Loss:  0.562
    Epoch   5 Batch  181/269 - Train Accuracy:  0.617, Validation Accuracy:  0.631, Loss:  0.570
    Epoch   5 Batch  182/269 - Train Accuracy:  0.639, Validation Accuracy:  0.627, Loss:  0.572
    Epoch   5 Batch  183/269 - Train Accuracy:  0.688, Validation Accuracy:  0.630, Loss:  0.498
    Epoch   5 Batch  184/269 - Train Accuracy:  0.616, Validation Accuracy:  0.632, Loss:  0.587
    Epoch   5 Batch  185/269 - Train Accuracy:  0.637, Validation Accuracy:  0.630, Loss:  0.567
    Epoch   5 Batch  186/269 - Train Accuracy:  0.611, Validation Accuracy:  0.632, Loss:  0.590
    Epoch   5 Batch  187/269 - Train Accuracy:  0.646, Validation Accuracy:  0.634, Loss:  0.552
    Epoch   5 Batch  188/269 - Train Accuracy:  0.649, Validation Accuracy:  0.636, Loss:  0.550
    Epoch   5 Batch  189/269 - Train Accuracy:  0.631, Validation Accuracy:  0.638, Loss:  0.557
    Epoch   5 Batch  190/269 - Train Accuracy:  0.621, Validation Accuracy:  0.640, Loss:  0.556
    Epoch   5 Batch  191/269 - Train Accuracy:  0.657, Validation Accuracy:  0.646, Loss:  0.557
    Epoch   5 Batch  192/269 - Train Accuracy:  0.653, Validation Accuracy:  0.641, Loss:  0.564
    Epoch   5 Batch  193/269 - Train Accuracy:  0.655, Validation Accuracy:  0.645, Loss:  0.559
    Epoch   5 Batch  194/269 - Train Accuracy:  0.656, Validation Accuracy:  0.645, Loss:  0.568
    Epoch   5 Batch  195/269 - Train Accuracy:  0.634, Validation Accuracy:  0.637, Loss:  0.564
    Epoch   5 Batch  196/269 - Train Accuracy:  0.623, Validation Accuracy:  0.643, Loss:  0.563
    Epoch   5 Batch  197/269 - Train Accuracy:  0.617, Validation Accuracy:  0.646, Loss:  0.598
    Epoch   5 Batch  198/269 - Train Accuracy:  0.625, Validation Accuracy:  0.648, Loss:  0.596
    Epoch   5 Batch  199/269 - Train Accuracy:  0.627, Validation Accuracy:  0.645, Loss:  0.583
    Epoch   5 Batch  200/269 - Train Accuracy:  0.634, Validation Accuracy:  0.641, Loss:  0.582
    Epoch   5 Batch  201/269 - Train Accuracy:  0.640, Validation Accuracy:  0.645, Loss:  0.562
    Epoch   5 Batch  202/269 - Train Accuracy:  0.628, Validation Accuracy:  0.647, Loss:  0.559
    Epoch   5 Batch  203/269 - Train Accuracy:  0.618, Validation Accuracy:  0.647, Loss:  0.598
    Epoch   5 Batch  204/269 - Train Accuracy:  0.633, Validation Accuracy:  0.646, Loss:  0.584
    Epoch   5 Batch  205/269 - Train Accuracy:  0.636, Validation Accuracy:  0.641, Loss:  0.554
    Epoch   5 Batch  206/269 - Train Accuracy:  0.628, Validation Accuracy:  0.651, Loss:  0.587
    Epoch   5 Batch  207/269 - Train Accuracy:  0.671, Validation Accuracy:  0.652, Loss:  0.540
    Epoch   5 Batch  208/269 - Train Accuracy:  0.626, Validation Accuracy:  0.642, Loss:  0.591
    Epoch   5 Batch  209/269 - Train Accuracy:  0.637, Validation Accuracy:  0.646, Loss:  0.570
    Epoch   5 Batch  210/269 - Train Accuracy:  0.644, Validation Accuracy:  0.644, Loss:  0.552
    Epoch   5 Batch  211/269 - Train Accuracy:  0.649, Validation Accuracy:  0.641, Loss:  0.564
    Epoch   5 Batch  212/269 - Train Accuracy:  0.652, Validation Accuracy:  0.641, Loss:  0.558
    Epoch   5 Batch  213/269 - Train Accuracy:  0.643, Validation Accuracy:  0.638, Loss:  0.555
    Epoch   5 Batch  214/269 - Train Accuracy:  0.649, Validation Accuracy:  0.638, Loss:  0.559
    Epoch   5 Batch  215/269 - Train Accuracy:  0.658, Validation Accuracy:  0.634, Loss:  0.531
    Epoch   5 Batch  216/269 - Train Accuracy:  0.613, Validation Accuracy:  0.638, Loss:  0.603
    Epoch   5 Batch  217/269 - Train Accuracy:  0.617, Validation Accuracy:  0.642, Loss:  0.574
    Epoch   5 Batch  218/269 - Train Accuracy:  0.629, Validation Accuracy:  0.638, Loss:  0.579
    Epoch   5 Batch  219/269 - Train Accuracy:  0.649, Validation Accuracy:  0.646, Loss:  0.585
    Epoch   5 Batch  220/269 - Train Accuracy:  0.651, Validation Accuracy:  0.643, Loss:  0.528
    Epoch   5 Batch  221/269 - Train Accuracy:  0.671, Validation Accuracy:  0.639, Loss:  0.551
    Epoch   5 Batch  222/269 - Train Accuracy:  0.656, Validation Accuracy:  0.648, Loss:  0.538
    Epoch   5 Batch  223/269 - Train Accuracy:  0.635, Validation Accuracy:  0.645, Loss:  0.548
    Epoch   5 Batch  224/269 - Train Accuracy:  0.656, Validation Accuracy:  0.647, Loss:  0.566
    Epoch   5 Batch  225/269 - Train Accuracy:  0.632, Validation Accuracy:  0.643, Loss:  0.561
    Epoch   5 Batch  226/269 - Train Accuracy:  0.647, Validation Accuracy:  0.648, Loss:  0.552
    Epoch   5 Batch  227/269 - Train Accuracy:  0.692, Validation Accuracy:  0.645, Loss:  0.498
    Epoch   5 Batch  228/269 - Train Accuracy:  0.641, Validation Accuracy:  0.635, Loss:  0.553
    Epoch   5 Batch  229/269 - Train Accuracy:  0.640, Validation Accuracy:  0.638, Loss:  0.546
    Epoch   5 Batch  230/269 - Train Accuracy:  0.635, Validation Accuracy:  0.639, Loss:  0.554
    Epoch   5 Batch  231/269 - Train Accuracy:  0.622, Validation Accuracy:  0.646, Loss:  0.585
    Epoch   5 Batch  232/269 - Train Accuracy:  0.613, Validation Accuracy:  0.646, Loss:  0.578
    Epoch   5 Batch  233/269 - Train Accuracy:  0.653, Validation Accuracy:  0.643, Loss:  0.557
    Epoch   5 Batch  234/269 - Train Accuracy:  0.644, Validation Accuracy:  0.643, Loss:  0.547
    Epoch   5 Batch  235/269 - Train Accuracy:  0.649, Validation Accuracy:  0.642, Loss:  0.536
    Epoch   5 Batch  236/269 - Train Accuracy:  0.623, Validation Accuracy:  0.648, Loss:  0.549
    Epoch   5 Batch  237/269 - Train Accuracy:  0.637, Validation Accuracy:  0.647, Loss:  0.549
    Epoch   5 Batch  238/269 - Train Accuracy:  0.658, Validation Accuracy:  0.653, Loss:  0.544
    Epoch   5 Batch  239/269 - Train Accuracy:  0.659, Validation Accuracy:  0.658, Loss:  0.544
    Epoch   5 Batch  240/269 - Train Accuracy:  0.670, Validation Accuracy:  0.655, Loss:  0.499
    Epoch   5 Batch  241/269 - Train Accuracy:  0.642, Validation Accuracy:  0.654, Loss:  0.558
    Epoch   5 Batch  242/269 - Train Accuracy:  0.645, Validation Accuracy:  0.651, Loss:  0.546
    Epoch   5 Batch  243/269 - Train Accuracy:  0.661, Validation Accuracy:  0.650, Loss:  0.536
    Epoch   5 Batch  244/269 - Train Accuracy:  0.645, Validation Accuracy:  0.645, Loss:  0.551
    Epoch   5 Batch  245/269 - Train Accuracy:  0.638, Validation Accuracy:  0.651, Loss:  0.581
    Epoch   5 Batch  246/269 - Train Accuracy:  0.635, Validation Accuracy:  0.651, Loss:  0.556
    Epoch   5 Batch  247/269 - Train Accuracy:  0.644, Validation Accuracy:  0.648, Loss:  0.563
    Epoch   5 Batch  248/269 - Train Accuracy:  0.649, Validation Accuracy:  0.651, Loss:  0.552
    Epoch   5 Batch  249/269 - Train Accuracy:  0.671, Validation Accuracy:  0.647, Loss:  0.520
    Epoch   5 Batch  250/269 - Train Accuracy:  0.627, Validation Accuracy:  0.646, Loss:  0.556
    Epoch   5 Batch  251/269 - Train Accuracy:  0.671, Validation Accuracy:  0.654, Loss:  0.535
    Epoch   5 Batch  252/269 - Train Accuracy:  0.644, Validation Accuracy:  0.653, Loss:  0.554
    Epoch   5 Batch  253/269 - Train Accuracy:  0.644, Validation Accuracy:  0.654, Loss:  0.564
    Epoch   5 Batch  254/269 - Train Accuracy:  0.649, Validation Accuracy:  0.659, Loss:  0.545
    Epoch   5 Batch  255/269 - Train Accuracy:  0.679, Validation Accuracy:  0.666, Loss:  0.522
    Epoch   5 Batch  256/269 - Train Accuracy:  0.640, Validation Accuracy:  0.662, Loss:  0.556
    Epoch   5 Batch  257/269 - Train Accuracy:  0.634, Validation Accuracy:  0.656, Loss:  0.558
    Epoch   5 Batch  258/269 - Train Accuracy:  0.641, Validation Accuracy:  0.657, Loss:  0.548
    Epoch   5 Batch  259/269 - Train Accuracy:  0.673, Validation Accuracy:  0.664, Loss:  0.549
    Epoch   5 Batch  260/269 - Train Accuracy:  0.645, Validation Accuracy:  0.667, Loss:  0.572
    Epoch   5 Batch  261/269 - Train Accuracy:  0.629, Validation Accuracy:  0.668, Loss:  0.572
    Epoch   5 Batch  262/269 - Train Accuracy:  0.665, Validation Accuracy:  0.664, Loss:  0.545
    Epoch   5 Batch  263/269 - Train Accuracy:  0.654, Validation Accuracy:  0.663, Loss:  0.556
    Epoch   5 Batch  264/269 - Train Accuracy:  0.627, Validation Accuracy:  0.652, Loss:  0.569
    Epoch   5 Batch  265/269 - Train Accuracy:  0.633, Validation Accuracy:  0.650, Loss:  0.564
    Epoch   5 Batch  266/269 - Train Accuracy:  0.656, Validation Accuracy:  0.652, Loss:  0.527
    Epoch   5 Batch  267/269 - Train Accuracy:  0.652, Validation Accuracy:  0.655, Loss:  0.547
    Epoch   6 Batch    0/269 - Train Accuracy:  0.635, Validation Accuracy:  0.657, Loss:  0.571
    Epoch   6 Batch    1/269 - Train Accuracy:  0.634, Validation Accuracy:  0.647, Loss:  0.551
    Epoch   6 Batch    2/269 - Train Accuracy:  0.632, Validation Accuracy:  0.651, Loss:  0.552
    Epoch   6 Batch    3/269 - Train Accuracy:  0.645, Validation Accuracy:  0.654, Loss:  0.554
    Epoch   6 Batch    4/269 - Train Accuracy:  0.633, Validation Accuracy:  0.660, Loss:  0.572
    Epoch   6 Batch    5/269 - Train Accuracy:  0.625, Validation Accuracy:  0.649, Loss:  0.562
    Epoch   6 Batch    6/269 - Train Accuracy:  0.652, Validation Accuracy:  0.641, Loss:  0.526
    Epoch   6 Batch    7/269 - Train Accuracy:  0.660, Validation Accuracy:  0.648, Loss:  0.537
    Epoch   6 Batch    8/269 - Train Accuracy:  0.635, Validation Accuracy:  0.653, Loss:  0.571
    Epoch   6 Batch    9/269 - Train Accuracy:  0.648, Validation Accuracy:  0.660, Loss:  0.547
    Epoch   6 Batch   10/269 - Train Accuracy:  0.641, Validation Accuracy:  0.662, Loss:  0.559
    Epoch   6 Batch   11/269 - Train Accuracy:  0.639, Validation Accuracy:  0.658, Loss:  0.556
    Epoch   6 Batch   12/269 - Train Accuracy:  0.620, Validation Accuracy:  0.659, Loss:  0.573
    Epoch   6 Batch   13/269 - Train Accuracy:  0.679, Validation Accuracy:  0.656, Loss:  0.498
    Epoch   6 Batch   14/269 - Train Accuracy:  0.655, Validation Accuracy:  0.656, Loss:  0.539
    Epoch   6 Batch   15/269 - Train Accuracy:  0.641, Validation Accuracy:  0.651, Loss:  0.531
    Epoch   6 Batch   16/269 - Train Accuracy:  0.650, Validation Accuracy:  0.650, Loss:  0.543
    Epoch   6 Batch   17/269 - Train Accuracy:  0.644, Validation Accuracy:  0.651, Loss:  0.528
    Epoch   6 Batch   18/269 - Train Accuracy:  0.630, Validation Accuracy:  0.655, Loss:  0.557
    Epoch   6 Batch   19/269 - Train Accuracy:  0.671, Validation Accuracy:  0.649, Loss:  0.502
    Epoch   6 Batch   20/269 - Train Accuracy:  0.640, Validation Accuracy:  0.657, Loss:  0.558
    Epoch   6 Batch   21/269 - Train Accuracy:  0.642, Validation Accuracy:  0.653, Loss:  0.571
    Epoch   6 Batch   22/269 - Train Accuracy:  0.654, Validation Accuracy:  0.656, Loss:  0.520
    Epoch   6 Batch   23/269 - Train Accuracy:  0.657, Validation Accuracy:  0.657, Loss:  0.527
    Epoch   6 Batch   24/269 - Train Accuracy:  0.652, Validation Accuracy:  0.664, Loss:  0.563
    Epoch   6 Batch   25/269 - Train Accuracy:  0.639, Validation Accuracy:  0.666, Loss:  0.560
    Epoch   6 Batch   26/269 - Train Accuracy:  0.680, Validation Accuracy:  0.666, Loss:  0.496
    Epoch   6 Batch   27/269 - Train Accuracy:  0.646, Validation Accuracy:  0.662, Loss:  0.529
    Epoch   6 Batch   28/269 - Train Accuracy:  0.618, Validation Accuracy:  0.662, Loss:  0.579
    Epoch   6 Batch   29/269 - Train Accuracy:  0.645, Validation Accuracy:  0.663, Loss:  0.555
    Epoch   6 Batch   30/269 - Train Accuracy:  0.659, Validation Accuracy:  0.663, Loss:  0.531
    Epoch   6 Batch   31/269 - Train Accuracy:  0.657, Validation Accuracy:  0.666, Loss:  0.520
    Epoch   6 Batch   32/269 - Train Accuracy:  0.653, Validation Accuracy:  0.668, Loss:  0.524
    Epoch   6 Batch   33/269 - Train Accuracy:  0.676, Validation Accuracy:  0.663, Loss:  0.509
    Epoch   6 Batch   34/269 - Train Accuracy:  0.662, Validation Accuracy:  0.667, Loss:  0.527
    Epoch   6 Batch   35/269 - Train Accuracy:  0.666, Validation Accuracy:  0.668, Loss:  0.552
    Epoch   6 Batch   36/269 - Train Accuracy:  0.657, Validation Accuracy:  0.666, Loss:  0.529
    Epoch   6 Batch   37/269 - Train Accuracy:  0.661, Validation Accuracy:  0.665, Loss:  0.528
    Epoch   6 Batch   38/269 - Train Accuracy:  0.659, Validation Accuracy:  0.664, Loss:  0.532
    Epoch   6 Batch   39/269 - Train Accuracy:  0.660, Validation Accuracy:  0.664, Loss:  0.529
    Epoch   6 Batch   40/269 - Train Accuracy:  0.658, Validation Accuracy:  0.671, Loss:  0.553
    Epoch   6 Batch   41/269 - Train Accuracy:  0.667, Validation Accuracy:  0.672, Loss:  0.537
    Epoch   6 Batch   42/269 - Train Accuracy:  0.680, Validation Accuracy:  0.673, Loss:  0.501
    Epoch   6 Batch   43/269 - Train Accuracy:  0.654, Validation Accuracy:  0.670, Loss:  0.542
    Epoch   6 Batch   44/269 - Train Accuracy:  0.672, Validation Accuracy:  0.668, Loss:  0.529
    Epoch   6 Batch   45/269 - Train Accuracy:  0.638, Validation Accuracy:  0.660, Loss:  0.549
    Epoch   6 Batch   46/269 - Train Accuracy:  0.653, Validation Accuracy:  0.660, Loss:  0.543
    Epoch   6 Batch   47/269 - Train Accuracy:  0.675, Validation Accuracy:  0.654, Loss:  0.494
    Epoch   6 Batch   48/269 - Train Accuracy:  0.665, Validation Accuracy:  0.667, Loss:  0.518
    Epoch   6 Batch   49/269 - Train Accuracy:  0.647, Validation Accuracy:  0.664, Loss:  0.530
    Epoch   6 Batch   50/269 - Train Accuracy:  0.648, Validation Accuracy:  0.659, Loss:  0.550
    Epoch   6 Batch   51/269 - Train Accuracy:  0.636, Validation Accuracy:  0.656, Loss:  0.530
    Epoch   6 Batch   52/269 - Train Accuracy:  0.648, Validation Accuracy:  0.657, Loss:  0.506
    Epoch   6 Batch   53/269 - Train Accuracy:  0.639, Validation Accuracy:  0.657, Loss:  0.551
    Epoch   6 Batch   54/269 - Train Accuracy:  0.651, Validation Accuracy:  0.663, Loss:  0.546
    Epoch   6 Batch   55/269 - Train Accuracy:  0.686, Validation Accuracy:  0.666, Loss:  0.515
    Epoch   6 Batch   56/269 - Train Accuracy:  0.672, Validation Accuracy:  0.664, Loss:  0.525
    Epoch   6 Batch   57/269 - Train Accuracy:  0.659, Validation Accuracy:  0.656, Loss:  0.536
    Epoch   6 Batch   58/269 - Train Accuracy:  0.650, Validation Accuracy:  0.656, Loss:  0.521
    Epoch   6 Batch   59/269 - Train Accuracy:  0.668, Validation Accuracy:  0.658, Loss:  0.500
    Epoch   6 Batch   60/269 - Train Accuracy:  0.664, Validation Accuracy:  0.662, Loss:  0.504
    Epoch   6 Batch   61/269 - Train Accuracy:  0.676, Validation Accuracy:  0.665, Loss:  0.488
    Epoch   6 Batch   62/269 - Train Accuracy:  0.664, Validation Accuracy:  0.667, Loss:  0.504
    Epoch   6 Batch   63/269 - Train Accuracy:  0.658, Validation Accuracy:  0.664, Loss:  0.535
    Epoch   6 Batch   64/269 - Train Accuracy:  0.645, Validation Accuracy:  0.661, Loss:  0.511
    Epoch   6 Batch   65/269 - Train Accuracy:  0.655, Validation Accuracy:  0.662, Loss:  0.520
    Epoch   6 Batch   66/269 - Train Accuracy:  0.653, Validation Accuracy:  0.657, Loss:  0.503
    Epoch   6 Batch   67/269 - Train Accuracy:  0.652, Validation Accuracy:  0.668, Loss:  0.537
    Epoch   6 Batch   68/269 - Train Accuracy:  0.641, Validation Accuracy:  0.663, Loss:  0.522
    Epoch   6 Batch   69/269 - Train Accuracy:  0.630, Validation Accuracy:  0.668, Loss:  0.570
    Epoch   6 Batch   70/269 - Train Accuracy:  0.681, Validation Accuracy:  0.661, Loss:  0.526
    Epoch   6 Batch   71/269 - Train Accuracy:  0.657, Validation Accuracy:  0.661, Loss:  0.548
    Epoch   6 Batch   72/269 - Train Accuracy:  0.664, Validation Accuracy:  0.666, Loss:  0.511
    Epoch   6 Batch   73/269 - Train Accuracy:  0.657, Validation Accuracy:  0.668, Loss:  0.538
    Epoch   6 Batch   74/269 - Train Accuracy:  0.651, Validation Accuracy:  0.661, Loss:  0.527
    Epoch   6 Batch   75/269 - Train Accuracy:  0.659, Validation Accuracy:  0.661, Loss:  0.526
    Epoch   6 Batch   76/269 - Train Accuracy:  0.649, Validation Accuracy:  0.662, Loss:  0.521
    Epoch   6 Batch   77/269 - Train Accuracy:  0.668, Validation Accuracy:  0.658, Loss:  0.516
    Epoch   6 Batch   78/269 - Train Accuracy:  0.661, Validation Accuracy:  0.657, Loss:  0.514
    Epoch   6 Batch   79/269 - Train Accuracy:  0.662, Validation Accuracy:  0.665, Loss:  0.510
    Epoch   6 Batch   80/269 - Train Accuracy:  0.675, Validation Accuracy:  0.667, Loss:  0.515
    Epoch   6 Batch   81/269 - Train Accuracy:  0.672, Validation Accuracy:  0.669, Loss:  0.528
    Epoch   6 Batch   82/269 - Train Accuracy:  0.674, Validation Accuracy:  0.667, Loss:  0.495
    Epoch   6 Batch   83/269 - Train Accuracy:  0.665, Validation Accuracy:  0.668, Loss:  0.533
    Epoch   6 Batch   84/269 - Train Accuracy:  0.664, Validation Accuracy:  0.666, Loss:  0.510
    Epoch   6 Batch   85/269 - Train Accuracy:  0.662, Validation Accuracy:  0.668, Loss:  0.520
    Epoch   6 Batch   86/269 - Train Accuracy:  0.631, Validation Accuracy:  0.665, Loss:  0.511
    Epoch   6 Batch   87/269 - Train Accuracy:  0.636, Validation Accuracy:  0.664, Loss:  0.549
    Epoch   6 Batch   88/269 - Train Accuracy:  0.662, Validation Accuracy:  0.667, Loss:  0.519
    Epoch   6 Batch   89/269 - Train Accuracy:  0.676, Validation Accuracy:  0.670, Loss:  0.515
    Epoch   6 Batch   90/269 - Train Accuracy:  0.629, Validation Accuracy:  0.670, Loss:  0.549
    Epoch   6 Batch   91/269 - Train Accuracy:  0.663, Validation Accuracy:  0.674, Loss:  0.500
    Epoch   6 Batch   92/269 - Train Accuracy:  0.660, Validation Accuracy:  0.676, Loss:  0.502
    Epoch   6 Batch   93/269 - Train Accuracy:  0.676, Validation Accuracy:  0.677, Loss:  0.494
    Epoch   6 Batch   94/269 - Train Accuracy:  0.658, Validation Accuracy:  0.675, Loss:  0.522
    Epoch   6 Batch   95/269 - Train Accuracy:  0.665, Validation Accuracy:  0.673, Loss:  0.514
    Epoch   6 Batch   96/269 - Train Accuracy:  0.666, Validation Accuracy:  0.672, Loss:  0.505
    Epoch   6 Batch   97/269 - Train Accuracy:  0.652, Validation Accuracy:  0.675, Loss:  0.507
    Epoch   6 Batch   98/269 - Train Accuracy:  0.672, Validation Accuracy:  0.675, Loss:  0.512
    Epoch   6 Batch   99/269 - Train Accuracy:  0.658, Validation Accuracy:  0.674, Loss:  0.533
    Epoch   6 Batch  100/269 - Train Accuracy:  0.685, Validation Accuracy:  0.676, Loss:  0.500
    Epoch   6 Batch  101/269 - Train Accuracy:  0.644, Validation Accuracy:  0.676, Loss:  0.545
    Epoch   6 Batch  102/269 - Train Accuracy:  0.666, Validation Accuracy:  0.676, Loss:  0.510
    Epoch   6 Batch  103/269 - Train Accuracy:  0.661, Validation Accuracy:  0.673, Loss:  0.503
    Epoch   6 Batch  104/269 - Train Accuracy:  0.661, Validation Accuracy:  0.674, Loss:  0.507
    Epoch   6 Batch  105/269 - Train Accuracy:  0.659, Validation Accuracy:  0.678, Loss:  0.515
    Epoch   6 Batch  106/269 - Train Accuracy:  0.657, Validation Accuracy:  0.675, Loss:  0.509
    Epoch   6 Batch  107/269 - Train Accuracy:  0.629, Validation Accuracy:  0.673, Loss:  0.531
    Epoch   6 Batch  108/269 - Train Accuracy:  0.665, Validation Accuracy:  0.671, Loss:  0.507
    Epoch   6 Batch  109/269 - Train Accuracy:  0.628, Validation Accuracy:  0.661, Loss:  0.521
    Epoch   6 Batch  110/269 - Train Accuracy:  0.644, Validation Accuracy:  0.663, Loss:  0.505
    Epoch   6 Batch  111/269 - Train Accuracy:  0.627, Validation Accuracy:  0.671, Loss:  0.542
    Epoch   6 Batch  112/269 - Train Accuracy:  0.661, Validation Accuracy:  0.668, Loss:  0.521
    Epoch   6 Batch  113/269 - Train Accuracy:  0.662, Validation Accuracy:  0.667, Loss:  0.489
    Epoch   6 Batch  114/269 - Train Accuracy:  0.653, Validation Accuracy:  0.666, Loss:  0.510
    Epoch   6 Batch  115/269 - Train Accuracy:  0.645, Validation Accuracy:  0.664, Loss:  0.535
    Epoch   6 Batch  116/269 - Train Accuracy:  0.651, Validation Accuracy:  0.665, Loss:  0.518
    Epoch   6 Batch  117/269 - Train Accuracy:  0.653, Validation Accuracy:  0.669, Loss:  0.504
    Epoch   6 Batch  118/269 - Train Accuracy:  0.688, Validation Accuracy:  0.669, Loss:  0.498
    Epoch   6 Batch  119/269 - Train Accuracy:  0.654, Validation Accuracy:  0.671, Loss:  0.531
    Epoch   6 Batch  120/269 - Train Accuracy:  0.661, Validation Accuracy:  0.671, Loss:  0.520
    Epoch   6 Batch  121/269 - Train Accuracy:  0.666, Validation Accuracy:  0.672, Loss:  0.498
    Epoch   6 Batch  122/269 - Train Accuracy:  0.665, Validation Accuracy:  0.673, Loss:  0.501
    Epoch   6 Batch  123/269 - Train Accuracy:  0.648, Validation Accuracy:  0.678, Loss:  0.524
    Epoch   6 Batch  124/269 - Train Accuracy:  0.667, Validation Accuracy:  0.679, Loss:  0.496
    Epoch   6 Batch  125/269 - Train Accuracy:  0.672, Validation Accuracy:  0.680, Loss:  0.497
    Epoch   6 Batch  126/269 - Train Accuracy:  0.675, Validation Accuracy:  0.677, Loss:  0.507
    Epoch   6 Batch  127/269 - Train Accuracy:  0.658, Validation Accuracy:  0.672, Loss:  0.522
    Epoch   6 Batch  128/269 - Train Accuracy:  0.683, Validation Accuracy:  0.675, Loss:  0.509
    Epoch   6 Batch  129/269 - Train Accuracy:  0.662, Validation Accuracy:  0.678, Loss:  0.506
    Epoch   6 Batch  130/269 - Train Accuracy:  0.643, Validation Accuracy:  0.678, Loss:  0.531
    Epoch   6 Batch  131/269 - Train Accuracy:  0.651, Validation Accuracy:  0.680, Loss:  0.520
    Epoch   6 Batch  132/269 - Train Accuracy:  0.658, Validation Accuracy:  0.672, Loss:  0.508
    Epoch   6 Batch  133/269 - Train Accuracy:  0.673, Validation Accuracy:  0.674, Loss:  0.493
    Epoch   6 Batch  134/269 - Train Accuracy:  0.646, Validation Accuracy:  0.671, Loss:  0.519
    Epoch   6 Batch  135/269 - Train Accuracy:  0.638, Validation Accuracy:  0.674, Loss:  0.542
    Epoch   6 Batch  136/269 - Train Accuracy:  0.629, Validation Accuracy:  0.669, Loss:  0.539
    Epoch   6 Batch  137/269 - Train Accuracy:  0.647, Validation Accuracy:  0.665, Loss:  0.534
    Epoch   6 Batch  138/269 - Train Accuracy:  0.659, Validation Accuracy:  0.667, Loss:  0.517
    Epoch   6 Batch  139/269 - Train Accuracy:  0.683, Validation Accuracy:  0.670, Loss:  0.494
    Epoch   6 Batch  140/269 - Train Accuracy:  0.671, Validation Accuracy:  0.668, Loss:  0.522
    Epoch   6 Batch  141/269 - Train Accuracy:  0.662, Validation Accuracy:  0.671, Loss:  0.520
    Epoch   6 Batch  142/269 - Train Accuracy:  0.658, Validation Accuracy:  0.671, Loss:  0.488
    Epoch   6 Batch  143/269 - Train Accuracy:  0.667, Validation Accuracy:  0.667, Loss:  0.498
    Epoch   6 Batch  144/269 - Train Accuracy:  0.665, Validation Accuracy:  0.674, Loss:  0.491
    Epoch   6 Batch  145/269 - Train Accuracy:  0.669, Validation Accuracy:  0.676, Loss:  0.499
    Epoch   6 Batch  146/269 - Train Accuracy:  0.656, Validation Accuracy:  0.671, Loss:  0.490
    Epoch   6 Batch  147/269 - Train Accuracy:  0.672, Validation Accuracy:  0.678, Loss:  0.488
    Epoch   6 Batch  148/269 - Train Accuracy:  0.655, Validation Accuracy:  0.673, Loss:  0.513
    Epoch   6 Batch  149/269 - Train Accuracy:  0.656, Validation Accuracy:  0.668, Loss:  0.517
    Epoch   6 Batch  150/269 - Train Accuracy:  0.676, Validation Accuracy:  0.673, Loss:  0.499
    Epoch   6 Batch  151/269 - Train Accuracy:  0.697, Validation Accuracy:  0.673, Loss:  0.486
    Epoch   6 Batch  152/269 - Train Accuracy:  0.661, Validation Accuracy:  0.670, Loss:  0.501
    Epoch   6 Batch  153/269 - Train Accuracy:  0.673, Validation Accuracy:  0.673, Loss:  0.492
    Epoch   6 Batch  154/269 - Train Accuracy:  0.650, Validation Accuracy:  0.678, Loss:  0.512
    Epoch   6 Batch  155/269 - Train Accuracy:  0.695, Validation Accuracy:  0.680, Loss:  0.479
    Epoch   6 Batch  156/269 - Train Accuracy:  0.659, Validation Accuracy:  0.677, Loss:  0.521
    Epoch   6 Batch  157/269 - Train Accuracy:  0.662, Validation Accuracy:  0.677, Loss:  0.496
    Epoch   6 Batch  158/269 - Train Accuracy:  0.667, Validation Accuracy:  0.678, Loss:  0.507
    Epoch   6 Batch  159/269 - Train Accuracy:  0.662, Validation Accuracy:  0.678, Loss:  0.504
    Epoch   6 Batch  160/269 - Train Accuracy:  0.673, Validation Accuracy:  0.678, Loss:  0.494
    Epoch   6 Batch  161/269 - Train Accuracy:  0.666, Validation Accuracy:  0.679, Loss:  0.500
    Epoch   6 Batch  162/269 - Train Accuracy:  0.671, Validation Accuracy:  0.679, Loss:  0.501
    Epoch   6 Batch  163/269 - Train Accuracy:  0.685, Validation Accuracy:  0.682, Loss:  0.496
    Epoch   6 Batch  164/269 - Train Accuracy:  0.667, Validation Accuracy:  0.679, Loss:  0.493
    Epoch   6 Batch  165/269 - Train Accuracy:  0.650, Validation Accuracy:  0.677, Loss:  0.510
    Epoch   6 Batch  166/269 - Train Accuracy:  0.685, Validation Accuracy:  0.681, Loss:  0.472
    Epoch   6 Batch  167/269 - Train Accuracy:  0.666, Validation Accuracy:  0.684, Loss:  0.499
    Epoch   6 Batch  168/269 - Train Accuracy:  0.660, Validation Accuracy:  0.683, Loss:  0.503
    Epoch   6 Batch  169/269 - Train Accuracy:  0.669, Validation Accuracy:  0.683, Loss:  0.503
    Epoch   6 Batch  170/269 - Train Accuracy:  0.665, Validation Accuracy:  0.675, Loss:  0.488
    Epoch   6 Batch  171/269 - Train Accuracy:  0.668, Validation Accuracy:  0.675, Loss:  0.521
    Epoch   6 Batch  172/269 - Train Accuracy:  0.670, Validation Accuracy:  0.673, Loss:  0.504
    Epoch   6 Batch  173/269 - Train Accuracy:  0.669, Validation Accuracy:  0.682, Loss:  0.486
    Epoch   6 Batch  174/269 - Train Accuracy:  0.653, Validation Accuracy:  0.680, Loss:  0.503
    Epoch   6 Batch  175/269 - Train Accuracy:  0.672, Validation Accuracy:  0.676, Loss:  0.509
    Epoch   6 Batch  176/269 - Train Accuracy:  0.646, Validation Accuracy:  0.678, Loss:  0.530
    Epoch   6 Batch  177/269 - Train Accuracy:  0.667, Validation Accuracy:  0.677, Loss:  0.484
    Epoch   6 Batch  178/269 - Train Accuracy:  0.662, Validation Accuracy:  0.680, Loss:  0.516
    Epoch   6 Batch  179/269 - Train Accuracy:  0.673, Validation Accuracy:  0.683, Loss:  0.491
    Epoch   6 Batch  180/269 - Train Accuracy:  0.681, Validation Accuracy:  0.686, Loss:  0.484
    Epoch   6 Batch  181/269 - Train Accuracy:  0.658, Validation Accuracy:  0.683, Loss:  0.496
    Epoch   6 Batch  182/269 - Train Accuracy:  0.671, Validation Accuracy:  0.676, Loss:  0.497
    Epoch   6 Batch  183/269 - Train Accuracy:  0.722, Validation Accuracy:  0.669, Loss:  0.431
    Epoch   6 Batch  184/269 - Train Accuracy:  0.648, Validation Accuracy:  0.673, Loss:  0.515
    Epoch   6 Batch  185/269 - Train Accuracy:  0.675, Validation Accuracy:  0.677, Loss:  0.486
    Epoch   6 Batch  186/269 - Train Accuracy:  0.656, Validation Accuracy:  0.682, Loss:  0.507
    Epoch   6 Batch  187/269 - Train Accuracy:  0.688, Validation Accuracy:  0.683, Loss:  0.481
    Epoch   6 Batch  188/269 - Train Accuracy:  0.689, Validation Accuracy:  0.682, Loss:  0.479
    Epoch   6 Batch  189/269 - Train Accuracy:  0.670, Validation Accuracy:  0.676, Loss:  0.488
    Epoch   6 Batch  190/269 - Train Accuracy:  0.662, Validation Accuracy:  0.676, Loss:  0.482
    Epoch   6 Batch  191/269 - Train Accuracy:  0.689, Validation Accuracy:  0.679, Loss:  0.482
    Epoch   6 Batch  192/269 - Train Accuracy:  0.677, Validation Accuracy:  0.676, Loss:  0.492
    Epoch   6 Batch  193/269 - Train Accuracy:  0.687, Validation Accuracy:  0.682, Loss:  0.482
    Epoch   6 Batch  194/269 - Train Accuracy:  0.683, Validation Accuracy:  0.676, Loss:  0.491
    Epoch   6 Batch  195/269 - Train Accuracy:  0.666, Validation Accuracy:  0.672, Loss:  0.489
    Epoch   6 Batch  196/269 - Train Accuracy:  0.659, Validation Accuracy:  0.685, Loss:  0.484
    Epoch   6 Batch  197/269 - Train Accuracy:  0.646, Validation Accuracy:  0.685, Loss:  0.514
    Epoch   6 Batch  198/269 - Train Accuracy:  0.655, Validation Accuracy:  0.684, Loss:  0.511
    Epoch   6 Batch  199/269 - Train Accuracy:  0.656, Validation Accuracy:  0.681, Loss:  0.492
    Epoch   6 Batch  200/269 - Train Accuracy:  0.671, Validation Accuracy:  0.681, Loss:  0.509
    Epoch   6 Batch  201/269 - Train Accuracy:  0.668, Validation Accuracy:  0.678, Loss:  0.490
    Epoch   6 Batch  202/269 - Train Accuracy:  0.663, Validation Accuracy:  0.687, Loss:  0.488
    Epoch   6 Batch  203/269 - Train Accuracy:  0.658, Validation Accuracy:  0.686, Loss:  0.520
    Epoch   6 Batch  204/269 - Train Accuracy:  0.654, Validation Accuracy:  0.688, Loss:  0.514
    Epoch   6 Batch  205/269 - Train Accuracy:  0.665, Validation Accuracy:  0.685, Loss:  0.485
    Epoch   6 Batch  206/269 - Train Accuracy:  0.659, Validation Accuracy:  0.681, Loss:  0.512
    Epoch   6 Batch  207/269 - Train Accuracy:  0.692, Validation Accuracy:  0.673, Loss:  0.465
    Epoch   6 Batch  208/269 - Train Accuracy:  0.656, Validation Accuracy:  0.679, Loss:  0.511
    Epoch   6 Batch  209/269 - Train Accuracy:  0.662, Validation Accuracy:  0.683, Loss:  0.499
    Epoch   6 Batch  210/269 - Train Accuracy:  0.679, Validation Accuracy:  0.691, Loss:  0.485
    Epoch   6 Batch  211/269 - Train Accuracy:  0.671, Validation Accuracy:  0.685, Loss:  0.490
    Epoch   6 Batch  212/269 - Train Accuracy:  0.684, Validation Accuracy:  0.683, Loss:  0.482
    Epoch   6 Batch  213/269 - Train Accuracy:  0.670, Validation Accuracy:  0.681, Loss:  0.485
    Epoch   6 Batch  214/269 - Train Accuracy:  0.681, Validation Accuracy:  0.673, Loss:  0.485
    Epoch   6 Batch  215/269 - Train Accuracy:  0.695, Validation Accuracy:  0.674, Loss:  0.464
    Epoch   6 Batch  216/269 - Train Accuracy:  0.654, Validation Accuracy:  0.680, Loss:  0.522
    Epoch   6 Batch  217/269 - Train Accuracy:  0.650, Validation Accuracy:  0.686, Loss:  0.504
    Epoch   6 Batch  218/269 - Train Accuracy:  0.668, Validation Accuracy:  0.691, Loss:  0.501
    Epoch   6 Batch  219/269 - Train Accuracy:  0.682, Validation Accuracy:  0.690, Loss:  0.502
    Epoch   6 Batch  220/269 - Train Accuracy:  0.681, Validation Accuracy:  0.684, Loss:  0.462
    Epoch   6 Batch  221/269 - Train Accuracy:  0.704, Validation Accuracy:  0.685, Loss:  0.484
    Epoch   6 Batch  222/269 - Train Accuracy:  0.685, Validation Accuracy:  0.682, Loss:  0.477
    Epoch   6 Batch  223/269 - Train Accuracy:  0.669, Validation Accuracy:  0.682, Loss:  0.479
    Epoch   6 Batch  224/269 - Train Accuracy:  0.682, Validation Accuracy:  0.677, Loss:  0.501
    Epoch   6 Batch  225/269 - Train Accuracy:  0.669, Validation Accuracy:  0.687, Loss:  0.492
    Epoch   6 Batch  226/269 - Train Accuracy:  0.674, Validation Accuracy:  0.688, Loss:  0.478
    Epoch   6 Batch  227/269 - Train Accuracy:  0.728, Validation Accuracy:  0.689, Loss:  0.445
    Epoch   6 Batch  228/269 - Train Accuracy:  0.671, Validation Accuracy:  0.681, Loss:  0.487
    Epoch   6 Batch  229/269 - Train Accuracy:  0.679, Validation Accuracy:  0.682, Loss:  0.482
    Epoch   6 Batch  230/269 - Train Accuracy:  0.661, Validation Accuracy:  0.677, Loss:  0.483
    Epoch   6 Batch  231/269 - Train Accuracy:  0.652, Validation Accuracy:  0.680, Loss:  0.510
    Epoch   6 Batch  232/269 - Train Accuracy:  0.645, Validation Accuracy:  0.679, Loss:  0.511
    Epoch   6 Batch  233/269 - Train Accuracy:  0.686, Validation Accuracy:  0.689, Loss:  0.500
    Epoch   6 Batch  234/269 - Train Accuracy:  0.679, Validation Accuracy:  0.687, Loss:  0.478
    Epoch   6 Batch  235/269 - Train Accuracy:  0.686, Validation Accuracy:  0.690, Loss:  0.475
    Epoch   6 Batch  236/269 - Train Accuracy:  0.660, Validation Accuracy:  0.683, Loss:  0.477
    Epoch   6 Batch  237/269 - Train Accuracy:  0.664, Validation Accuracy:  0.691, Loss:  0.483
    Epoch   6 Batch  238/269 - Train Accuracy:  0.687, Validation Accuracy:  0.691, Loss:  0.474
    Epoch   6 Batch  239/269 - Train Accuracy:  0.689, Validation Accuracy:  0.688, Loss:  0.478
    Epoch   6 Batch  240/269 - Train Accuracy:  0.693, Validation Accuracy:  0.682, Loss:  0.438
    Epoch   6 Batch  241/269 - Train Accuracy:  0.673, Validation Accuracy:  0.683, Loss:  0.491
    Epoch   6 Batch  242/269 - Train Accuracy:  0.668, Validation Accuracy:  0.684, Loss:  0.475
    Epoch   6 Batch  243/269 - Train Accuracy:  0.686, Validation Accuracy:  0.684, Loss:  0.468
    Epoch   6 Batch  244/269 - Train Accuracy:  0.676, Validation Accuracy:  0.683, Loss:  0.478
    Epoch   6 Batch  245/269 - Train Accuracy:  0.672, Validation Accuracy:  0.688, Loss:  0.500
    Epoch   6 Batch  246/269 - Train Accuracy:  0.659, Validation Accuracy:  0.689, Loss:  0.480
    Epoch   6 Batch  247/269 - Train Accuracy:  0.681, Validation Accuracy:  0.689, Loss:  0.487
    Epoch   6 Batch  248/269 - Train Accuracy:  0.675, Validation Accuracy:  0.684, Loss:  0.470
    Epoch   6 Batch  249/269 - Train Accuracy:  0.694, Validation Accuracy:  0.684, Loss:  0.452
    Epoch   6 Batch  250/269 - Train Accuracy:  0.666, Validation Accuracy:  0.687, Loss:  0.483
    Epoch   6 Batch  251/269 - Train Accuracy:  0.698, Validation Accuracy:  0.689, Loss:  0.462
    Epoch   6 Batch  252/269 - Train Accuracy:  0.679, Validation Accuracy:  0.692, Loss:  0.481
    Epoch   6 Batch  253/269 - Train Accuracy:  0.673, Validation Accuracy:  0.691, Loss:  0.488
    Epoch   6 Batch  254/269 - Train Accuracy:  0.682, Validation Accuracy:  0.689, Loss:  0.471
    Epoch   6 Batch  255/269 - Train Accuracy:  0.701, Validation Accuracy:  0.685, Loss:  0.457
    Epoch   6 Batch  256/269 - Train Accuracy:  0.665, Validation Accuracy:  0.687, Loss:  0.485
    Epoch   6 Batch  257/269 - Train Accuracy:  0.658, Validation Accuracy:  0.686, Loss:  0.487
    Epoch   6 Batch  258/269 - Train Accuracy:  0.666, Validation Accuracy:  0.685, Loss:  0.482
    Epoch   6 Batch  259/269 - Train Accuracy:  0.691, Validation Accuracy:  0.687, Loss:  0.477
    Epoch   6 Batch  260/269 - Train Accuracy:  0.659, Validation Accuracy:  0.690, Loss:  0.499
    Epoch   6 Batch  261/269 - Train Accuracy:  0.654, Validation Accuracy:  0.691, Loss:  0.506
    Epoch   6 Batch  262/269 - Train Accuracy:  0.692, Validation Accuracy:  0.690, Loss:  0.476
    Epoch   6 Batch  263/269 - Train Accuracy:  0.681, Validation Accuracy:  0.689, Loss:  0.487
    Epoch   6 Batch  264/269 - Train Accuracy:  0.667, Validation Accuracy:  0.687, Loss:  0.500
    Epoch   6 Batch  265/269 - Train Accuracy:  0.666, Validation Accuracy:  0.690, Loss:  0.491
    Epoch   6 Batch  266/269 - Train Accuracy:  0.688, Validation Accuracy:  0.688, Loss:  0.467
    Epoch   6 Batch  267/269 - Train Accuracy:  0.685, Validation Accuracy:  0.684, Loss:  0.486
    Epoch   7 Batch    0/269 - Train Accuracy:  0.668, Validation Accuracy:  0.693, Loss:  0.510
    Epoch   7 Batch    1/269 - Train Accuracy:  0.665, Validation Accuracy:  0.688, Loss:  0.480
    Epoch   7 Batch    2/269 - Train Accuracy:  0.657, Validation Accuracy:  0.686, Loss:  0.489
    Epoch   7 Batch    3/269 - Train Accuracy:  0.674, Validation Accuracy:  0.684, Loss:  0.490
    Epoch   7 Batch    4/269 - Train Accuracy:  0.655, Validation Accuracy:  0.691, Loss:  0.502
    Epoch   7 Batch    5/269 - Train Accuracy:  0.650, Validation Accuracy:  0.685, Loss:  0.493
    Epoch   7 Batch    6/269 - Train Accuracy:  0.688, Validation Accuracy:  0.685, Loss:  0.464
    Epoch   7 Batch    7/269 - Train Accuracy:  0.681, Validation Accuracy:  0.684, Loss:  0.466
    Epoch   7 Batch    8/269 - Train Accuracy:  0.653, Validation Accuracy:  0.683, Loss:  0.496
    Epoch   7 Batch    9/269 - Train Accuracy:  0.664, Validation Accuracy:  0.684, Loss:  0.487
    Epoch   7 Batch   10/269 - Train Accuracy:  0.677, Validation Accuracy:  0.687, Loss:  0.485
    Epoch   7 Batch   11/269 - Train Accuracy:  0.661, Validation Accuracy:  0.682, Loss:  0.482
    Epoch   7 Batch   12/269 - Train Accuracy:  0.656, Validation Accuracy:  0.689, Loss:  0.501
    Epoch   7 Batch   13/269 - Train Accuracy:  0.700, Validation Accuracy:  0.690, Loss:  0.441
    Epoch   7 Batch   14/269 - Train Accuracy:  0.677, Validation Accuracy:  0.690, Loss:  0.471
    Epoch   7 Batch   15/269 - Train Accuracy:  0.662, Validation Accuracy:  0.687, Loss:  0.463
    Epoch   7 Batch   16/269 - Train Accuracy:  0.676, Validation Accuracy:  0.685, Loss:  0.475
    Epoch   7 Batch   17/269 - Train Accuracy:  0.678, Validation Accuracy:  0.684, Loss:  0.459
    Epoch   7 Batch   18/269 - Train Accuracy:  0.663, Validation Accuracy:  0.685, Loss:  0.489
    Epoch   7 Batch   19/269 - Train Accuracy:  0.701, Validation Accuracy:  0.688, Loss:  0.438
    Epoch   7 Batch   20/269 - Train Accuracy:  0.679, Validation Accuracy:  0.687, Loss:  0.491
    Epoch   7 Batch   21/269 - Train Accuracy:  0.675, Validation Accuracy:  0.690, Loss:  0.503
    Epoch   7 Batch   22/269 - Train Accuracy:  0.686, Validation Accuracy:  0.693, Loss:  0.451
    Epoch   7 Batch   23/269 - Train Accuracy:  0.683, Validation Accuracy:  0.693, Loss:  0.462
    Epoch   7 Batch   24/269 - Train Accuracy:  0.677, Validation Accuracy:  0.693, Loss:  0.483
    Epoch   7 Batch   25/269 - Train Accuracy:  0.657, Validation Accuracy:  0.694, Loss:  0.499
    Epoch   7 Batch   26/269 - Train Accuracy:  0.699, Validation Accuracy:  0.693, Loss:  0.432
    Epoch   7 Batch   27/269 - Train Accuracy:  0.674, Validation Accuracy:  0.694, Loss:  0.466
    Epoch   7 Batch   28/269 - Train Accuracy:  0.644, Validation Accuracy:  0.690, Loss:  0.506
    Epoch   7 Batch   29/269 - Train Accuracy:  0.671, Validation Accuracy:  0.692, Loss:  0.484
    Epoch   7 Batch   30/269 - Train Accuracy:  0.684, Validation Accuracy:  0.694, Loss:  0.466
    Epoch   7 Batch   31/269 - Train Accuracy:  0.684, Validation Accuracy:  0.698, Loss:  0.463
    Epoch   7 Batch   32/269 - Train Accuracy:  0.680, Validation Accuracy:  0.694, Loss:  0.459
    Epoch   7 Batch   33/269 - Train Accuracy:  0.700, Validation Accuracy:  0.692, Loss:  0.454
    Epoch   7 Batch   34/269 - Train Accuracy:  0.690, Validation Accuracy:  0.694, Loss:  0.460
    Epoch   7 Batch   35/269 - Train Accuracy:  0.686, Validation Accuracy:  0.694, Loss:  0.485
    Epoch   7 Batch   36/269 - Train Accuracy:  0.673, Validation Accuracy:  0.694, Loss:  0.465
    Epoch   7 Batch   37/269 - Train Accuracy:  0.688, Validation Accuracy:  0.694, Loss:  0.460
    Epoch   7 Batch   38/269 - Train Accuracy:  0.687, Validation Accuracy:  0.697, Loss:  0.457
    Epoch   7 Batch   39/269 - Train Accuracy:  0.682, Validation Accuracy:  0.696, Loss:  0.468
    Epoch   7 Batch   40/269 - Train Accuracy:  0.677, Validation Accuracy:  0.695, Loss:  0.487
    Epoch   7 Batch   41/269 - Train Accuracy:  0.681, Validation Accuracy:  0.695, Loss:  0.472
    Epoch   7 Batch   42/269 - Train Accuracy:  0.692, Validation Accuracy:  0.697, Loss:  0.438
    Epoch   7 Batch   43/269 - Train Accuracy:  0.667, Validation Accuracy:  0.696, Loss:  0.475
    Epoch   7 Batch   44/269 - Train Accuracy:  0.694, Validation Accuracy:  0.695, Loss:  0.468
    Epoch   7 Batch   45/269 - Train Accuracy:  0.665, Validation Accuracy:  0.691, Loss:  0.483
    Epoch   7 Batch   46/269 - Train Accuracy:  0.683, Validation Accuracy:  0.695, Loss:  0.481
    Epoch   7 Batch   47/269 - Train Accuracy:  0.709, Validation Accuracy:  0.697, Loss:  0.436
    Epoch   7 Batch   48/269 - Train Accuracy:  0.699, Validation Accuracy:  0.701, Loss:  0.450
    Epoch   7 Batch   49/269 - Train Accuracy:  0.675, Validation Accuracy:  0.693, Loss:  0.466
    Epoch   7 Batch   50/269 - Train Accuracy:  0.681, Validation Accuracy:  0.689, Loss:  0.482
    Epoch   7 Batch   51/269 - Train Accuracy:  0.673, Validation Accuracy:  0.694, Loss:  0.466
    Epoch   7 Batch   52/269 - Train Accuracy:  0.678, Validation Accuracy:  0.695, Loss:  0.444
    Epoch   7 Batch   53/269 - Train Accuracy:  0.662, Validation Accuracy:  0.689, Loss:  0.488
    Epoch   7 Batch   54/269 - Train Accuracy:  0.680, Validation Accuracy:  0.693, Loss:  0.481
    Epoch   7 Batch   55/269 - Train Accuracy:  0.704, Validation Accuracy:  0.698, Loss:  0.454
    Epoch   7 Batch   56/269 - Train Accuracy:  0.702, Validation Accuracy:  0.697, Loss:  0.466
    Epoch   7 Batch   57/269 - Train Accuracy:  0.685, Validation Accuracy:  0.695, Loss:  0.467
    Epoch   7 Batch   58/269 - Train Accuracy:  0.685, Validation Accuracy:  0.697, Loss:  0.455
    Epoch   7 Batch   59/269 - Train Accuracy:  0.698, Validation Accuracy:  0.695, Loss:  0.437
    Epoch   7 Batch   60/269 - Train Accuracy:  0.682, Validation Accuracy:  0.689, Loss:  0.440
    Epoch   7 Batch   61/269 - Train Accuracy:  0.695, Validation Accuracy:  0.693, Loss:  0.428
    Epoch   7 Batch   62/269 - Train Accuracy:  0.688, Validation Accuracy:  0.688, Loss:  0.447
    Epoch   7 Batch   63/269 - Train Accuracy:  0.680, Validation Accuracy:  0.696, Loss:  0.462
    Epoch   7 Batch   64/269 - Train Accuracy:  0.673, Validation Accuracy:  0.698, Loss:  0.449
    Epoch   7 Batch   65/269 - Train Accuracy:  0.666, Validation Accuracy:  0.689, Loss:  0.463
    Epoch   7 Batch   66/269 - Train Accuracy:  0.677, Validation Accuracy:  0.694, Loss:  0.445
    Epoch   7 Batch   67/269 - Train Accuracy:  0.678, Validation Accuracy:  0.696, Loss:  0.470
    Epoch   7 Batch   68/269 - Train Accuracy:  0.668, Validation Accuracy:  0.695, Loss:  0.466
    Epoch   7 Batch   69/269 - Train Accuracy:  0.659, Validation Accuracy:  0.697, Loss:  0.504
    Epoch   7 Batch   70/269 - Train Accuracy:  0.708, Validation Accuracy:  0.698, Loss:  0.461
    Epoch   7 Batch   71/269 - Train Accuracy:  0.688, Validation Accuracy:  0.694, Loss:  0.477
    Epoch   7 Batch   72/269 - Train Accuracy:  0.692, Validation Accuracy:  0.688, Loss:  0.454
    Epoch   7 Batch   73/269 - Train Accuracy:  0.684, Validation Accuracy:  0.691, Loss:  0.474
    Epoch   7 Batch   74/269 - Train Accuracy:  0.683, Validation Accuracy:  0.694, Loss:  0.460
    Epoch   7 Batch   75/269 - Train Accuracy:  0.687, Validation Accuracy:  0.690, Loss:  0.459
    Epoch   7 Batch   76/269 - Train Accuracy:  0.675, Validation Accuracy:  0.693, Loss:  0.461
    Epoch   7 Batch   77/269 - Train Accuracy:  0.704, Validation Accuracy:  0.688, Loss:  0.455
    Epoch   7 Batch   78/269 - Train Accuracy:  0.689, Validation Accuracy:  0.693, Loss:  0.453
    Epoch   7 Batch   79/269 - Train Accuracy:  0.686, Validation Accuracy:  0.694, Loss:  0.455
    Epoch   7 Batch   80/269 - Train Accuracy:  0.703, Validation Accuracy:  0.698, Loss:  0.452
    Epoch   7 Batch   81/269 - Train Accuracy:  0.690, Validation Accuracy:  0.694, Loss:  0.460
    Epoch   7 Batch   82/269 - Train Accuracy:  0.698, Validation Accuracy:  0.699, Loss:  0.437
    Epoch   7 Batch   83/269 - Train Accuracy:  0.696, Validation Accuracy:  0.696, Loss:  0.470
    Epoch   7 Batch   84/269 - Train Accuracy:  0.693, Validation Accuracy:  0.689, Loss:  0.447
    Epoch   7 Batch   85/269 - Train Accuracy:  0.679, Validation Accuracy:  0.693, Loss:  0.458
    Epoch   7 Batch   86/269 - Train Accuracy:  0.672, Validation Accuracy:  0.695, Loss:  0.450
    Epoch   7 Batch   87/269 - Train Accuracy:  0.667, Validation Accuracy:  0.696, Loss:  0.480
    Epoch   7 Batch   88/269 - Train Accuracy:  0.687, Validation Accuracy:  0.699, Loss:  0.456
    Epoch   7 Batch   89/269 - Train Accuracy:  0.696, Validation Accuracy:  0.697, Loss:  0.451
    Epoch   7 Batch   90/269 - Train Accuracy:  0.652, Validation Accuracy:  0.698, Loss:  0.481
    Epoch   7 Batch   91/269 - Train Accuracy:  0.684, Validation Accuracy:  0.699, Loss:  0.438
    Epoch   7 Batch   92/269 - Train Accuracy:  0.681, Validation Accuracy:  0.697, Loss:  0.440
    Epoch   7 Batch   93/269 - Train Accuracy:  0.694, Validation Accuracy:  0.697, Loss:  0.440
    Epoch   7 Batch   94/269 - Train Accuracy:  0.676, Validation Accuracy:  0.693, Loss:  0.462
    Epoch   7 Batch   95/269 - Train Accuracy:  0.685, Validation Accuracy:  0.692, Loss:  0.455
    Epoch   7 Batch   96/269 - Train Accuracy:  0.683, Validation Accuracy:  0.693, Loss:  0.444
    Epoch   7 Batch   97/269 - Train Accuracy:  0.678, Validation Accuracy:  0.695, Loss:  0.442
    Epoch   7 Batch   98/269 - Train Accuracy:  0.693, Validation Accuracy:  0.696, Loss:  0.456
    Epoch   7 Batch   99/269 - Train Accuracy:  0.673, Validation Accuracy:  0.695, Loss:  0.463
    Epoch   7 Batch  100/269 - Train Accuracy:  0.710, Validation Accuracy:  0.696, Loss:  0.445
    Epoch   7 Batch  101/269 - Train Accuracy:  0.658, Validation Accuracy:  0.693, Loss:  0.479
    Epoch   7 Batch  102/269 - Train Accuracy:  0.690, Validation Accuracy:  0.699, Loss:  0.451
    Epoch   7 Batch  103/269 - Train Accuracy:  0.684, Validation Accuracy:  0.696, Loss:  0.447
    Epoch   7 Batch  104/269 - Train Accuracy:  0.678, Validation Accuracy:  0.696, Loss:  0.445
    Epoch   7 Batch  105/269 - Train Accuracy:  0.684, Validation Accuracy:  0.701, Loss:  0.455
    Epoch   7 Batch  106/269 - Train Accuracy:  0.680, Validation Accuracy:  0.702, Loss:  0.446
    Epoch   7 Batch  107/269 - Train Accuracy:  0.650, Validation Accuracy:  0.696, Loss:  0.471
    Epoch   7 Batch  108/269 - Train Accuracy:  0.693, Validation Accuracy:  0.698, Loss:  0.447
    Epoch   7 Batch  109/269 - Train Accuracy:  0.661, Validation Accuracy:  0.694, Loss:  0.454
    Epoch   7 Batch  110/269 - Train Accuracy:  0.672, Validation Accuracy:  0.697, Loss:  0.449
    Epoch   7 Batch  111/269 - Train Accuracy:  0.668, Validation Accuracy:  0.699, Loss:  0.477
    Epoch   7 Batch  112/269 - Train Accuracy:  0.696, Validation Accuracy:  0.701, Loss:  0.455
    Epoch   7 Batch  113/269 - Train Accuracy:  0.698, Validation Accuracy:  0.696, Loss:  0.432
    Epoch   7 Batch  114/269 - Train Accuracy:  0.688, Validation Accuracy:  0.696, Loss:  0.448
    Epoch   7 Batch  115/269 - Train Accuracy:  0.677, Validation Accuracy:  0.693, Loss:  0.472
    Epoch   7 Batch  116/269 - Train Accuracy:  0.690, Validation Accuracy:  0.696, Loss:  0.451
    Epoch   7 Batch  117/269 - Train Accuracy:  0.678, Validation Accuracy:  0.697, Loss:  0.449
    Epoch   7 Batch  118/269 - Train Accuracy:  0.717, Validation Accuracy:  0.699, Loss:  0.437
    Epoch   7 Batch  119/269 - Train Accuracy:  0.679, Validation Accuracy:  0.703, Loss:  0.470
    Epoch   7 Batch  120/269 - Train Accuracy:  0.688, Validation Accuracy:  0.698, Loss:  0.456
    Epoch   7 Batch  121/269 - Train Accuracy:  0.684, Validation Accuracy:  0.697, Loss:  0.441
    Epoch   7 Batch  122/269 - Train Accuracy:  0.694, Validation Accuracy:  0.697, Loss:  0.445
    Epoch   7 Batch  123/269 - Train Accuracy:  0.671, Validation Accuracy:  0.698, Loss:  0.465
    Epoch   7 Batch  124/269 - Train Accuracy:  0.682, Validation Accuracy:  0.703, Loss:  0.439
    Epoch   7 Batch  125/269 - Train Accuracy:  0.692, Validation Accuracy:  0.702, Loss:  0.436
    Epoch   7 Batch  126/269 - Train Accuracy:  0.695, Validation Accuracy:  0.700, Loss:  0.445
    Epoch   7 Batch  127/269 - Train Accuracy:  0.674, Validation Accuracy:  0.699, Loss:  0.460
    Epoch   7 Batch  128/269 - Train Accuracy:  0.699, Validation Accuracy:  0.700, Loss:  0.452
    Epoch   7 Batch  129/269 - Train Accuracy:  0.683, Validation Accuracy:  0.700, Loss:  0.440
    Epoch   7 Batch  130/269 - Train Accuracy:  0.680, Validation Accuracy:  0.699, Loss:  0.462
    Epoch   7 Batch  131/269 - Train Accuracy:  0.677, Validation Accuracy:  0.696, Loss:  0.459
    Epoch   7 Batch  132/269 - Train Accuracy:  0.681, Validation Accuracy:  0.693, Loss:  0.449
    Epoch   7 Batch  133/269 - Train Accuracy:  0.698, Validation Accuracy:  0.696, Loss:  0.430
    Epoch   7 Batch  134/269 - Train Accuracy:  0.677, Validation Accuracy:  0.698, Loss:  0.456
    Epoch   7 Batch  135/269 - Train Accuracy:  0.670, Validation Accuracy:  0.700, Loss:  0.473
    Epoch   7 Batch  136/269 - Train Accuracy:  0.664, Validation Accuracy:  0.704, Loss:  0.475
    Epoch   7 Batch  137/269 - Train Accuracy:  0.686, Validation Accuracy:  0.705, Loss:  0.471
    Epoch   7 Batch  138/269 - Train Accuracy:  0.692, Validation Accuracy:  0.700, Loss:  0.455
    Epoch   7 Batch  139/269 - Train Accuracy:  0.710, Validation Accuracy:  0.696, Loss:  0.432
    Epoch   7 Batch  140/269 - Train Accuracy:  0.701, Validation Accuracy:  0.697, Loss:  0.457
    Epoch   7 Batch  141/269 - Train Accuracy:  0.683, Validation Accuracy:  0.696, Loss:  0.463
    Epoch   7 Batch  142/269 - Train Accuracy:  0.687, Validation Accuracy:  0.698, Loss:  0.438
    Epoch   7 Batch  143/269 - Train Accuracy:  0.694, Validation Accuracy:  0.698, Loss:  0.443
    Epoch   7 Batch  144/269 - Train Accuracy:  0.686, Validation Accuracy:  0.700, Loss:  0.426
    Epoch   7 Batch  145/269 - Train Accuracy:  0.686, Validation Accuracy:  0.696, Loss:  0.445
    Epoch   7 Batch  146/269 - Train Accuracy:  0.687, Validation Accuracy:  0.702, Loss:  0.441
    Epoch   7 Batch  147/269 - Train Accuracy:  0.691, Validation Accuracy:  0.693, Loss:  0.424
    Epoch   7 Batch  148/269 - Train Accuracy:  0.679, Validation Accuracy:  0.694, Loss:  0.453
    Epoch   7 Batch  149/269 - Train Accuracy:  0.685, Validation Accuracy:  0.703, Loss:  0.463
    Epoch   7 Batch  150/269 - Train Accuracy:  0.701, Validation Accuracy:  0.706, Loss:  0.440
    Epoch   7 Batch  151/269 - Train Accuracy:  0.717, Validation Accuracy:  0.704, Loss:  0.430
    Epoch   7 Batch  152/269 - Train Accuracy:  0.697, Validation Accuracy:  0.700, Loss:  0.442
    Epoch   7 Batch  153/269 - Train Accuracy:  0.692, Validation Accuracy:  0.701, Loss:  0.439
    Epoch   7 Batch  154/269 - Train Accuracy:  0.679, Validation Accuracy:  0.699, Loss:  0.449
    Epoch   7 Batch  155/269 - Train Accuracy:  0.713, Validation Accuracy:  0.700, Loss:  0.422
    Epoch   7 Batch  156/269 - Train Accuracy:  0.677, Validation Accuracy:  0.696, Loss:  0.460
    Epoch   7 Batch  157/269 - Train Accuracy:  0.689, Validation Accuracy:  0.702, Loss:  0.440
    Epoch   7 Batch  158/269 - Train Accuracy:  0.687, Validation Accuracy:  0.703, Loss:  0.442
    Epoch   7 Batch  159/269 - Train Accuracy:  0.689, Validation Accuracy:  0.701, Loss:  0.443
    Epoch   7 Batch  160/269 - Train Accuracy:  0.694, Validation Accuracy:  0.694, Loss:  0.435
    Epoch   7 Batch  161/269 - Train Accuracy:  0.686, Validation Accuracy:  0.700, Loss:  0.448
    Epoch   7 Batch  162/269 - Train Accuracy:  0.691, Validation Accuracy:  0.702, Loss:  0.438
    Epoch   7 Batch  163/269 - Train Accuracy:  0.707, Validation Accuracy:  0.703, Loss:  0.440
    Epoch   7 Batch  164/269 - Train Accuracy:  0.689, Validation Accuracy:  0.702, Loss:  0.436
    Epoch   7 Batch  165/269 - Train Accuracy:  0.667, Validation Accuracy:  0.700, Loss:  0.451
    Epoch   7 Batch  166/269 - Train Accuracy:  0.706, Validation Accuracy:  0.701, Loss:  0.417
    Epoch   7 Batch  167/269 - Train Accuracy:  0.692, Validation Accuracy:  0.703, Loss:  0.438
    Epoch   7 Batch  168/269 - Train Accuracy:  0.674, Validation Accuracy:  0.701, Loss:  0.447
    Epoch   7 Batch  169/269 - Train Accuracy:  0.691, Validation Accuracy:  0.702, Loss:  0.450
    Epoch   7 Batch  170/269 - Train Accuracy:  0.690, Validation Accuracy:  0.704, Loss:  0.432
    Epoch   7 Batch  171/269 - Train Accuracy:  0.695, Validation Accuracy:  0.697, Loss:  0.453
    Epoch   7 Batch  172/269 - Train Accuracy:  0.699, Validation Accuracy:  0.701, Loss:  0.448
    Epoch   7 Batch  173/269 - Train Accuracy:  0.698, Validation Accuracy:  0.703, Loss:  0.427
    Epoch   7 Batch  174/269 - Train Accuracy:  0.684, Validation Accuracy:  0.703, Loss:  0.442
    Epoch   7 Batch  175/269 - Train Accuracy:  0.694, Validation Accuracy:  0.708, Loss:  0.455
    Epoch   7 Batch  176/269 - Train Accuracy:  0.673, Validation Accuracy:  0.704, Loss:  0.465
    Epoch   7 Batch  177/269 - Train Accuracy:  0.700, Validation Accuracy:  0.704, Loss:  0.424
    Epoch   7 Batch  178/269 - Train Accuracy:  0.683, Validation Accuracy:  0.703, Loss:  0.448
    Epoch   7 Batch  179/269 - Train Accuracy:  0.693, Validation Accuracy:  0.703, Loss:  0.438
    Epoch   7 Batch  180/269 - Train Accuracy:  0.700, Validation Accuracy:  0.707, Loss:  0.428
    Epoch   7 Batch  181/269 - Train Accuracy:  0.671, Validation Accuracy:  0.703, Loss:  0.435
    Epoch   7 Batch  182/269 - Train Accuracy:  0.696, Validation Accuracy:  0.706, Loss:  0.435
    Epoch   7 Batch  183/269 - Train Accuracy:  0.742, Validation Accuracy:  0.707, Loss:  0.383
    Epoch   7 Batch  184/269 - Train Accuracy:  0.671, Validation Accuracy:  0.703, Loss:  0.449
    Epoch   7 Batch  185/269 - Train Accuracy:  0.703, Validation Accuracy:  0.704, Loss:  0.433
    Epoch   7 Batch  186/269 - Train Accuracy:  0.683, Validation Accuracy:  0.704, Loss:  0.445
    Epoch   7 Batch  187/269 - Train Accuracy:  0.708, Validation Accuracy:  0.706, Loss:  0.428
    Epoch   7 Batch  188/269 - Train Accuracy:  0.714, Validation Accuracy:  0.708, Loss:  0.423
    Epoch   7 Batch  189/269 - Train Accuracy:  0.700, Validation Accuracy:  0.702, Loss:  0.425
    Epoch   7 Batch  190/269 - Train Accuracy:  0.700, Validation Accuracy:  0.707, Loss:  0.426
    Epoch   7 Batch  191/269 - Train Accuracy:  0.709, Validation Accuracy:  0.706, Loss:  0.426
    Epoch   7 Batch  192/269 - Train Accuracy:  0.699, Validation Accuracy:  0.701, Loss:  0.437
    Epoch   7 Batch  193/269 - Train Accuracy:  0.701, Validation Accuracy:  0.706, Loss:  0.436
    Epoch   7 Batch  194/269 - Train Accuracy:  0.702, Validation Accuracy:  0.698, Loss:  0.433
    Epoch   7 Batch  195/269 - Train Accuracy:  0.692, Validation Accuracy:  0.704, Loss:  0.437
    Epoch   7 Batch  196/269 - Train Accuracy:  0.683, Validation Accuracy:  0.706, Loss:  0.429
    Epoch   7 Batch  197/269 - Train Accuracy:  0.660, Validation Accuracy:  0.706, Loss:  0.459
    Epoch   7 Batch  198/269 - Train Accuracy:  0.684, Validation Accuracy:  0.701, Loss:  0.460
    Epoch   7 Batch  199/269 - Train Accuracy:  0.683, Validation Accuracy:  0.698, Loss:  0.436
    Epoch   7 Batch  200/269 - Train Accuracy:  0.694, Validation Accuracy:  0.701, Loss:  0.448
    Epoch   7 Batch  201/269 - Train Accuracy:  0.695, Validation Accuracy:  0.709, Loss:  0.435
    Epoch   7 Batch  202/269 - Train Accuracy:  0.680, Validation Accuracy:  0.700, Loss:  0.426
    Epoch   7 Batch  203/269 - Train Accuracy:  0.684, Validation Accuracy:  0.708, Loss:  0.470
    Epoch   7 Batch  204/269 - Train Accuracy:  0.676, Validation Accuracy:  0.702, Loss:  0.450
    Epoch   7 Batch  205/269 - Train Accuracy:  0.685, Validation Accuracy:  0.708, Loss:  0.431
    Epoch   7 Batch  206/269 - Train Accuracy:  0.686, Validation Accuracy:  0.708, Loss:  0.451
    Epoch   7 Batch  207/269 - Train Accuracy:  0.714, Validation Accuracy:  0.705, Loss:  0.420
    Epoch   7 Batch  208/269 - Train Accuracy:  0.680, Validation Accuracy:  0.707, Loss:  0.451
    Epoch   7 Batch  209/269 - Train Accuracy:  0.695, Validation Accuracy:  0.709, Loss:  0.438
    Epoch   7 Batch  210/269 - Train Accuracy:  0.697, Validation Accuracy:  0.706, Loss:  0.419
    Epoch   7 Batch  211/269 - Train Accuracy:  0.694, Validation Accuracy:  0.708, Loss:  0.432
    Epoch   7 Batch  212/269 - Train Accuracy:  0.700, Validation Accuracy:  0.704, Loss:  0.427
    Epoch   7 Batch  213/269 - Train Accuracy:  0.697, Validation Accuracy:  0.706, Loss:  0.430
    Epoch   7 Batch  214/269 - Train Accuracy:  0.706, Validation Accuracy:  0.703, Loss:  0.427
    Epoch   7 Batch  215/269 - Train Accuracy:  0.724, Validation Accuracy:  0.705, Loss:  0.405
    Epoch   7 Batch  216/269 - Train Accuracy:  0.681, Validation Accuracy:  0.707, Loss:  0.459
    Epoch   7 Batch  217/269 - Train Accuracy:  0.673, Validation Accuracy:  0.708, Loss:  0.442
    Epoch   7 Batch  218/269 - Train Accuracy:  0.688, Validation Accuracy:  0.710, Loss:  0.443
    Epoch   7 Batch  219/269 - Train Accuracy:  0.700, Validation Accuracy:  0.709, Loss:  0.452
    Epoch   7 Batch  220/269 - Train Accuracy:  0.699, Validation Accuracy:  0.709, Loss:  0.404
    Epoch   7 Batch  221/269 - Train Accuracy:  0.722, Validation Accuracy:  0.708, Loss:  0.432
    Epoch   7 Batch  222/269 - Train Accuracy:  0.712, Validation Accuracy:  0.707, Loss:  0.414
    Epoch   7 Batch  223/269 - Train Accuracy:  0.697, Validation Accuracy:  0.705, Loss:  0.416
    Epoch   7 Batch  224/269 - Train Accuracy:  0.709, Validation Accuracy:  0.710, Loss:  0.440
    Epoch   7 Batch  225/269 - Train Accuracy:  0.687, Validation Accuracy:  0.708, Loss:  0.427
    Epoch   7 Batch  226/269 - Train Accuracy:  0.695, Validation Accuracy:  0.709, Loss:  0.422
    Epoch   7 Batch  227/269 - Train Accuracy:  0.743, Validation Accuracy:  0.708, Loss:  0.390
    Epoch   7 Batch  228/269 - Train Accuracy:  0.691, Validation Accuracy:  0.708, Loss:  0.423
    Epoch   7 Batch  229/269 - Train Accuracy:  0.697, Validation Accuracy:  0.708, Loss:  0.420
    Epoch   7 Batch  230/269 - Train Accuracy:  0.690, Validation Accuracy:  0.712, Loss:  0.424
    Epoch   7 Batch  231/269 - Train Accuracy:  0.679, Validation Accuracy:  0.713, Loss:  0.446
    Epoch   7 Batch  232/269 - Train Accuracy:  0.666, Validation Accuracy:  0.714, Loss:  0.447
    Epoch   7 Batch  233/269 - Train Accuracy:  0.706, Validation Accuracy:  0.712, Loss:  0.432
    Epoch   7 Batch  234/269 - Train Accuracy:  0.694, Validation Accuracy:  0.711, Loss:  0.421
    Epoch   7 Batch  235/269 - Train Accuracy:  0.711, Validation Accuracy:  0.709, Loss:  0.410
    Epoch   7 Batch  236/269 - Train Accuracy:  0.678, Validation Accuracy:  0.709, Loss:  0.414
    Epoch   7 Batch  237/269 - Train Accuracy:  0.690, Validation Accuracy:  0.711, Loss:  0.423
    Epoch   7 Batch  238/269 - Train Accuracy:  0.706, Validation Accuracy:  0.711, Loss:  0.420
    Epoch   7 Batch  239/269 - Train Accuracy:  0.710, Validation Accuracy:  0.712, Loss:  0.419
    Epoch   7 Batch  240/269 - Train Accuracy:  0.712, Validation Accuracy:  0.710, Loss:  0.386
    Epoch   7 Batch  241/269 - Train Accuracy:  0.694, Validation Accuracy:  0.707, Loss:  0.432
    Epoch   7 Batch  242/269 - Train Accuracy:  0.684, Validation Accuracy:  0.706, Loss:  0.425
    Epoch   7 Batch  243/269 - Train Accuracy:  0.708, Validation Accuracy:  0.708, Loss:  0.413
    Epoch   7 Batch  244/269 - Train Accuracy:  0.697, Validation Accuracy:  0.709, Loss:  0.425
    Epoch   7 Batch  245/269 - Train Accuracy:  0.686, Validation Accuracy:  0.713, Loss:  0.451
    Epoch   7 Batch  246/269 - Train Accuracy:  0.678, Validation Accuracy:  0.711, Loss:  0.423
    Epoch   7 Batch  247/269 - Train Accuracy:  0.695, Validation Accuracy:  0.709, Loss:  0.436
    Epoch   7 Batch  248/269 - Train Accuracy:  0.692, Validation Accuracy:  0.706, Loss:  0.419
    Epoch   7 Batch  249/269 - Train Accuracy:  0.710, Validation Accuracy:  0.696, Loss:  0.390
    Epoch   7 Batch  250/269 - Train Accuracy:  0.687, Validation Accuracy:  0.705, Loss:  0.429
    Epoch   7 Batch  251/269 - Train Accuracy:  0.715, Validation Accuracy:  0.708, Loss:  0.409
    Epoch   7 Batch  252/269 - Train Accuracy:  0.697, Validation Accuracy:  0.709, Loss:  0.429
    Epoch   7 Batch  253/269 - Train Accuracy:  0.698, Validation Accuracy:  0.709, Loss:  0.435
    Epoch   7 Batch  254/269 - Train Accuracy:  0.699, Validation Accuracy:  0.712, Loss:  0.418
    Epoch   7 Batch  255/269 - Train Accuracy:  0.708, Validation Accuracy:  0.710, Loss:  0.403
    Epoch   7 Batch  256/269 - Train Accuracy:  0.671, Validation Accuracy:  0.709, Loss:  0.425
    Epoch   7 Batch  257/269 - Train Accuracy:  0.676, Validation Accuracy:  0.707, Loss:  0.429
    Epoch   7 Batch  258/269 - Train Accuracy:  0.684, Validation Accuracy:  0.708, Loss:  0.432
    Epoch   7 Batch  259/269 - Train Accuracy:  0.719, Validation Accuracy:  0.710, Loss:  0.420
    Epoch   7 Batch  260/269 - Train Accuracy:  0.675, Validation Accuracy:  0.707, Loss:  0.444
    Epoch   7 Batch  261/269 - Train Accuracy:  0.669, Validation Accuracy:  0.711, Loss:  0.442
    Epoch   7 Batch  262/269 - Train Accuracy:  0.712, Validation Accuracy:  0.711, Loss:  0.415
    Epoch   7 Batch  263/269 - Train Accuracy:  0.693, Validation Accuracy:  0.706, Loss:  0.429
    Epoch   7 Batch  264/269 - Train Accuracy:  0.689, Validation Accuracy:  0.704, Loss:  0.440
    Epoch   7 Batch  265/269 - Train Accuracy:  0.694, Validation Accuracy:  0.709, Loss:  0.428
    Epoch   7 Batch  266/269 - Train Accuracy:  0.708, Validation Accuracy:  0.713, Loss:  0.409
    Epoch   7 Batch  267/269 - Train Accuracy:  0.704, Validation Accuracy:  0.714, Loss:  0.425
    Epoch   8 Batch    0/269 - Train Accuracy:  0.695, Validation Accuracy:  0.711, Loss:  0.443
    Epoch   8 Batch    1/269 - Train Accuracy:  0.688, Validation Accuracy:  0.714, Loss:  0.427
    Epoch   8 Batch    2/269 - Train Accuracy:  0.681, Validation Accuracy:  0.714, Loss:  0.428
    Epoch   8 Batch    3/269 - Train Accuracy:  0.697, Validation Accuracy:  0.707, Loss:  0.427
    Epoch   8 Batch    4/269 - Train Accuracy:  0.683, Validation Accuracy:  0.708, Loss:  0.442
    Epoch   8 Batch    5/269 - Train Accuracy:  0.679, Validation Accuracy:  0.714, Loss:  0.434
    Epoch   8 Batch    6/269 - Train Accuracy:  0.705, Validation Accuracy:  0.715, Loss:  0.403
    Epoch   8 Batch    7/269 - Train Accuracy:  0.701, Validation Accuracy:  0.713, Loss:  0.409
    Epoch   8 Batch    8/269 - Train Accuracy:  0.679, Validation Accuracy:  0.717, Loss:  0.437
    Epoch   8 Batch    9/269 - Train Accuracy:  0.683, Validation Accuracy:  0.715, Loss:  0.430
    Epoch   8 Batch   10/269 - Train Accuracy:  0.691, Validation Accuracy:  0.714, Loss:  0.429
    Epoch   8 Batch   11/269 - Train Accuracy:  0.687, Validation Accuracy:  0.716, Loss:  0.425
    Epoch   8 Batch   12/269 - Train Accuracy:  0.672, Validation Accuracy:  0.713, Loss:  0.440
    Epoch   8 Batch   13/269 - Train Accuracy:  0.726, Validation Accuracy:  0.713, Loss:  0.385
    Epoch   8 Batch   14/269 - Train Accuracy:  0.698, Validation Accuracy:  0.717, Loss:  0.414
    Epoch   8 Batch   15/269 - Train Accuracy:  0.690, Validation Accuracy:  0.717, Loss:  0.404
    Epoch   8 Batch   16/269 - Train Accuracy:  0.696, Validation Accuracy:  0.714, Loss:  0.415
    Epoch   8 Batch   17/269 - Train Accuracy:  0.694, Validation Accuracy:  0.716, Loss:  0.407
    Epoch   8 Batch   18/269 - Train Accuracy:  0.681, Validation Accuracy:  0.716, Loss:  0.425
    Epoch   8 Batch   19/269 - Train Accuracy:  0.717, Validation Accuracy:  0.713, Loss:  0.384
    Epoch   8 Batch   20/269 - Train Accuracy:  0.693, Validation Accuracy:  0.713, Loss:  0.429
    Epoch   8 Batch   21/269 - Train Accuracy:  0.685, Validation Accuracy:  0.717, Loss:  0.448
    Epoch   8 Batch   22/269 - Train Accuracy:  0.711, Validation Accuracy:  0.718, Loss:  0.398
    Epoch   8 Batch   23/269 - Train Accuracy:  0.701, Validation Accuracy:  0.716, Loss:  0.406
    Epoch   8 Batch   24/269 - Train Accuracy:  0.692, Validation Accuracy:  0.716, Loss:  0.427
    Epoch   8 Batch   25/269 - Train Accuracy:  0.687, Validation Accuracy:  0.716, Loss:  0.438
    Epoch   8 Batch   26/269 - Train Accuracy:  0.713, Validation Accuracy:  0.716, Loss:  0.384
    Epoch   8 Batch   27/269 - Train Accuracy:  0.690, Validation Accuracy:  0.715, Loss:  0.410
    Epoch   8 Batch   28/269 - Train Accuracy:  0.666, Validation Accuracy:  0.716, Loss:  0.452
    Epoch   8 Batch   29/269 - Train Accuracy:  0.699, Validation Accuracy:  0.714, Loss:  0.432
    Epoch   8 Batch   30/269 - Train Accuracy:  0.698, Validation Accuracy:  0.702, Loss:  0.405
    Epoch   8 Batch   31/269 - Train Accuracy:  0.708, Validation Accuracy:  0.718, Loss:  0.407
    Epoch   8 Batch   32/269 - Train Accuracy:  0.695, Validation Accuracy:  0.718, Loss:  0.403
    Epoch   8 Batch   33/269 - Train Accuracy:  0.707, Validation Accuracy:  0.710, Loss:  0.397
    Epoch   8 Batch   34/269 - Train Accuracy:  0.711, Validation Accuracy:  0.715, Loss:  0.415
    Epoch   8 Batch   35/269 - Train Accuracy:  0.702, Validation Accuracy:  0.711, Loss:  0.422
    Epoch   8 Batch   36/269 - Train Accuracy:  0.693, Validation Accuracy:  0.709, Loss:  0.412
    Epoch   8 Batch   37/269 - Train Accuracy:  0.704, Validation Accuracy:  0.709, Loss:  0.412
    Epoch   8 Batch   38/269 - Train Accuracy:  0.698, Validation Accuracy:  0.711, Loss:  0.409
    Epoch   8 Batch   39/269 - Train Accuracy:  0.699, Validation Accuracy:  0.716, Loss:  0.416
    Epoch   8 Batch   40/269 - Train Accuracy:  0.691, Validation Accuracy:  0.705, Loss:  0.427
    Epoch   8 Batch   41/269 - Train Accuracy:  0.702, Validation Accuracy:  0.711, Loss:  0.418
    Epoch   8 Batch   42/269 - Train Accuracy:  0.716, Validation Accuracy:  0.714, Loss:  0.388
    Epoch   8 Batch   43/269 - Train Accuracy:  0.687, Validation Accuracy:  0.712, Loss:  0.422
    Epoch   8 Batch   44/269 - Train Accuracy:  0.707, Validation Accuracy:  0.712, Loss:  0.407
    Epoch   8 Batch   45/269 - Train Accuracy:  0.689, Validation Accuracy:  0.712, Loss:  0.427
    Epoch   8 Batch   46/269 - Train Accuracy:  0.692, Validation Accuracy:  0.711, Loss:  0.422
    Epoch   8 Batch   47/269 - Train Accuracy:  0.729, Validation Accuracy:  0.716, Loss:  0.386
    Epoch   8 Batch   48/269 - Train Accuracy:  0.712, Validation Accuracy:  0.714, Loss:  0.392
    Epoch   8 Batch   49/269 - Train Accuracy:  0.687, Validation Accuracy:  0.714, Loss:  0.412
    Epoch   8 Batch   50/269 - Train Accuracy:  0.694, Validation Accuracy:  0.715, Loss:  0.430
    Epoch   8 Batch   51/269 - Train Accuracy:  0.691, Validation Accuracy:  0.713, Loss:  0.407
    Epoch   8 Batch   52/269 - Train Accuracy:  0.694, Validation Accuracy:  0.710, Loss:  0.385
    Epoch   8 Batch   53/269 - Train Accuracy:  0.688, Validation Accuracy:  0.715, Loss:  0.431
    Epoch   8 Batch   54/269 - Train Accuracy:  0.702, Validation Accuracy:  0.717, Loss:  0.419
    Epoch   8 Batch   55/269 - Train Accuracy:  0.721, Validation Accuracy:  0.717, Loss:  0.401
    Epoch   8 Batch   56/269 - Train Accuracy:  0.718, Validation Accuracy:  0.717, Loss:  0.409
    Epoch   8 Batch   57/269 - Train Accuracy:  0.705, Validation Accuracy:  0.717, Loss:  0.413
    Epoch   8 Batch   58/269 - Train Accuracy:  0.706, Validation Accuracy:  0.718, Loss:  0.401
    Epoch   8 Batch   59/269 - Train Accuracy:  0.719, Validation Accuracy:  0.715, Loss:  0.380
    Epoch   8 Batch   60/269 - Train Accuracy:  0.698, Validation Accuracy:  0.714, Loss:  0.392
    Epoch   8 Batch   61/269 - Train Accuracy:  0.716, Validation Accuracy:  0.715, Loss:  0.377
    Epoch   8 Batch   62/269 - Train Accuracy:  0.702, Validation Accuracy:  0.716, Loss:  0.392
    Epoch   8 Batch   63/269 - Train Accuracy:  0.700, Validation Accuracy:  0.719, Loss:  0.410
    Epoch   8 Batch   64/269 - Train Accuracy:  0.696, Validation Accuracy:  0.718, Loss:  0.394
    Epoch   8 Batch   65/269 - Train Accuracy:  0.691, Validation Accuracy:  0.713, Loss:  0.399
    Epoch   8 Batch   66/269 - Train Accuracy:  0.703, Validation Accuracy:  0.717, Loss:  0.391
    Epoch   8 Batch   67/269 - Train Accuracy:  0.697, Validation Accuracy:  0.719, Loss:  0.413
    Epoch   8 Batch   68/269 - Train Accuracy:  0.688, Validation Accuracy:  0.718, Loss:  0.408
    Epoch   8 Batch   69/269 - Train Accuracy:  0.678, Validation Accuracy:  0.717, Loss:  0.442
    Epoch   8 Batch   70/269 - Train Accuracy:  0.727, Validation Accuracy:  0.712, Loss:  0.406
    Epoch   8 Batch   71/269 - Train Accuracy:  0.707, Validation Accuracy:  0.710, Loss:  0.422
    Epoch   8 Batch   72/269 - Train Accuracy:  0.705, Validation Accuracy:  0.709, Loss:  0.400
    Epoch   8 Batch   73/269 - Train Accuracy:  0.701, Validation Accuracy:  0.715, Loss:  0.416
    Epoch   8 Batch   74/269 - Train Accuracy:  0.696, Validation Accuracy:  0.720, Loss:  0.403
    Epoch   8 Batch   75/269 - Train Accuracy:  0.708, Validation Accuracy:  0.711, Loss:  0.399
    Epoch   8 Batch   76/269 - Train Accuracy:  0.699, Validation Accuracy:  0.716, Loss:  0.406
    Epoch   8 Batch   77/269 - Train Accuracy:  0.722, Validation Accuracy:  0.718, Loss:  0.398
    Epoch   8 Batch   78/269 - Train Accuracy:  0.704, Validation Accuracy:  0.718, Loss:  0.398
    Epoch   8 Batch   79/269 - Train Accuracy:  0.712, Validation Accuracy:  0.715, Loss:  0.401
    Epoch   8 Batch   80/269 - Train Accuracy:  0.719, Validation Accuracy:  0.714, Loss:  0.391
    Epoch   8 Batch   81/269 - Train Accuracy:  0.716, Validation Accuracy:  0.715, Loss:  0.407
    Epoch   8 Batch   82/269 - Train Accuracy:  0.715, Validation Accuracy:  0.719, Loss:  0.383
    Epoch   8 Batch   83/269 - Train Accuracy:  0.717, Validation Accuracy:  0.719, Loss:  0.415
    Epoch   8 Batch   84/269 - Train Accuracy:  0.720, Validation Accuracy:  0.716, Loss:  0.399
    Epoch   8 Batch   85/269 - Train Accuracy:  0.708, Validation Accuracy:  0.716, Loss:  0.402
    Epoch   8 Batch   86/269 - Train Accuracy:  0.688, Validation Accuracy:  0.714, Loss:  0.393
    Epoch   8 Batch   87/269 - Train Accuracy:  0.686, Validation Accuracy:  0.717, Loss:  0.425
    Epoch   8 Batch   88/269 - Train Accuracy:  0.703, Validation Accuracy:  0.715, Loss:  0.400
    Epoch   8 Batch   89/269 - Train Accuracy:  0.717, Validation Accuracy:  0.720, Loss:  0.402
    Epoch   8 Batch   90/269 - Train Accuracy:  0.667, Validation Accuracy:  0.721, Loss:  0.423
    Epoch   8 Batch   91/269 - Train Accuracy:  0.702, Validation Accuracy:  0.723, Loss:  0.388
    Epoch   8 Batch   92/269 - Train Accuracy:  0.705, Validation Accuracy:  0.721, Loss:  0.392
    Epoch   8 Batch   93/269 - Train Accuracy:  0.711, Validation Accuracy:  0.722, Loss:  0.382
    Epoch   8 Batch   94/269 - Train Accuracy:  0.709, Validation Accuracy:  0.721, Loss:  0.405
    Epoch   8 Batch   95/269 - Train Accuracy:  0.701, Validation Accuracy:  0.721, Loss:  0.398
    Epoch   8 Batch   96/269 - Train Accuracy:  0.705, Validation Accuracy:  0.718, Loss:  0.397
    Epoch   8 Batch   97/269 - Train Accuracy:  0.697, Validation Accuracy:  0.714, Loss:  0.393
    Epoch   8 Batch   98/269 - Train Accuracy:  0.707, Validation Accuracy:  0.717, Loss:  0.401
    Epoch   8 Batch   99/269 - Train Accuracy:  0.697, Validation Accuracy:  0.719, Loss:  0.409
    Epoch   8 Batch  100/269 - Train Accuracy:  0.729, Validation Accuracy:  0.717, Loss:  0.387
    Epoch   8 Batch  101/269 - Train Accuracy:  0.672, Validation Accuracy:  0.720, Loss:  0.427
    Epoch   8 Batch  102/269 - Train Accuracy:  0.709, Validation Accuracy:  0.721, Loss:  0.398
    Epoch   8 Batch  103/269 - Train Accuracy:  0.697, Validation Accuracy:  0.716, Loss:  0.393
    Epoch   8 Batch  104/269 - Train Accuracy:  0.698, Validation Accuracy:  0.714, Loss:  0.400
    Epoch   8 Batch  105/269 - Train Accuracy:  0.705, Validation Accuracy:  0.714, Loss:  0.401
    Epoch   8 Batch  106/269 - Train Accuracy:  0.699, Validation Accuracy:  0.719, Loss:  0.397
    Epoch   8 Batch  107/269 - Train Accuracy:  0.669, Validation Accuracy:  0.710, Loss:  0.416
    Epoch   8 Batch  108/269 - Train Accuracy:  0.702, Validation Accuracy:  0.724, Loss:  0.401
    Epoch   8 Batch  109/269 - Train Accuracy:  0.676, Validation Accuracy:  0.710, Loss:  0.396
    Epoch   8 Batch  110/269 - Train Accuracy:  0.686, Validation Accuracy:  0.715, Loss:  0.409
    Epoch   8 Batch  111/269 - Train Accuracy:  0.685, Validation Accuracy:  0.716, Loss:  0.440
    Epoch   8 Batch  112/269 - Train Accuracy:  0.714, Validation Accuracy:  0.712, Loss:  0.403
    Epoch   8 Batch  113/269 - Train Accuracy:  0.699, Validation Accuracy:  0.709, Loss:  0.400
    Epoch   8 Batch  114/269 - Train Accuracy:  0.699, Validation Accuracy:  0.714, Loss:  0.412
    Epoch   8 Batch  115/269 - Train Accuracy:  0.685, Validation Accuracy:  0.710, Loss:  0.415
    Epoch   8 Batch  116/269 - Train Accuracy:  0.700, Validation Accuracy:  0.718, Loss:  0.416
    Epoch   8 Batch  117/269 - Train Accuracy:  0.691, Validation Accuracy:  0.717, Loss:  0.397
    Epoch   8 Batch  118/269 - Train Accuracy:  0.735, Validation Accuracy:  0.722, Loss:  0.395
    Epoch   8 Batch  119/269 - Train Accuracy:  0.694, Validation Accuracy:  0.719, Loss:  0.416
    Epoch   8 Batch  120/269 - Train Accuracy:  0.696, Validation Accuracy:  0.713, Loss:  0.421
    Epoch   8 Batch  121/269 - Train Accuracy:  0.693, Validation Accuracy:  0.711, Loss:  0.402
    Epoch   8 Batch  122/269 - Train Accuracy:  0.708, Validation Accuracy:  0.711, Loss:  0.401
    Epoch   8 Batch  123/269 - Train Accuracy:  0.684, Validation Accuracy:  0.718, Loss:  0.420
    Epoch   8 Batch  124/269 - Train Accuracy:  0.702, Validation Accuracy:  0.712, Loss:  0.392
    Epoch   8 Batch  125/269 - Train Accuracy:  0.708, Validation Accuracy:  0.712, Loss:  0.393
    Epoch   8 Batch  126/269 - Train Accuracy:  0.704, Validation Accuracy:  0.705, Loss:  0.406
    Epoch   8 Batch  127/269 - Train Accuracy:  0.691, Validation Accuracy:  0.716, Loss:  0.423
    Epoch   8 Batch  128/269 - Train Accuracy:  0.713, Validation Accuracy:  0.712, Loss:  0.398
    Epoch   8 Batch  129/269 - Train Accuracy:  0.696, Validation Accuracy:  0.713, Loss:  0.399
    Epoch   8 Batch  130/269 - Train Accuracy:  0.685, Validation Accuracy:  0.714, Loss:  0.419
    Epoch   8 Batch  131/269 - Train Accuracy:  0.687, Validation Accuracy:  0.713, Loss:  0.412
    Epoch   8 Batch  132/269 - Train Accuracy:  0.693, Validation Accuracy:  0.713, Loss:  0.403
    Epoch   8 Batch  133/269 - Train Accuracy:  0.715, Validation Accuracy:  0.719, Loss:  0.388
    Epoch   8 Batch  134/269 - Train Accuracy:  0.695, Validation Accuracy:  0.719, Loss:  0.407
    Epoch   8 Batch  135/269 - Train Accuracy:  0.684, Validation Accuracy:  0.717, Loss:  0.428
    Epoch   8 Batch  136/269 - Train Accuracy:  0.675, Validation Accuracy:  0.713, Loss:  0.421
    Epoch   8 Batch  137/269 - Train Accuracy:  0.698, Validation Accuracy:  0.720, Loss:  0.417
    Epoch   8 Batch  138/269 - Train Accuracy:  0.710, Validation Accuracy:  0.719, Loss:  0.407
    Epoch   8 Batch  139/269 - Train Accuracy:  0.728, Validation Accuracy:  0.717, Loss:  0.387
    Epoch   8 Batch  140/269 - Train Accuracy:  0.714, Validation Accuracy:  0.720, Loss:  0.405
    Epoch   8 Batch  141/269 - Train Accuracy:  0.699, Validation Accuracy:  0.719, Loss:  0.408
    Epoch   8 Batch  142/269 - Train Accuracy:  0.709, Validation Accuracy:  0.719, Loss:  0.384
    Epoch   8 Batch  143/269 - Train Accuracy:  0.709, Validation Accuracy:  0.714, Loss:  0.393
    Epoch   8 Batch  144/269 - Train Accuracy:  0.709, Validation Accuracy:  0.719, Loss:  0.374
    Epoch   8 Batch  145/269 - Train Accuracy:  0.705, Validation Accuracy:  0.718, Loss:  0.389
    Epoch   8 Batch  146/269 - Train Accuracy:  0.700, Validation Accuracy:  0.720, Loss:  0.379
    Epoch   8 Batch  147/269 - Train Accuracy:  0.708, Validation Accuracy:  0.717, Loss:  0.371
    Epoch   8 Batch  148/269 - Train Accuracy:  0.701, Validation Accuracy:  0.720, Loss:  0.396
    Epoch   8 Batch  149/269 - Train Accuracy:  0.704, Validation Accuracy:  0.724, Loss:  0.399
    Epoch   8 Batch  150/269 - Train Accuracy:  0.714, Validation Accuracy:  0.722, Loss:  0.386
    Epoch   8 Batch  151/269 - Train Accuracy:  0.729, Validation Accuracy:  0.720, Loss:  0.377
    Epoch   8 Batch  152/269 - Train Accuracy:  0.717, Validation Accuracy:  0.722, Loss:  0.391
    Epoch   8 Batch  153/269 - Train Accuracy:  0.713, Validation Accuracy:  0.721, Loss:  0.389
    Epoch   8 Batch  154/269 - Train Accuracy:  0.691, Validation Accuracy:  0.719, Loss:  0.395
    Epoch   8 Batch  155/269 - Train Accuracy:  0.727, Validation Accuracy:  0.720, Loss:  0.370
    Epoch   8 Batch  156/269 - Train Accuracy:  0.698, Validation Accuracy:  0.725, Loss:  0.409
    Epoch   8 Batch  157/269 - Train Accuracy:  0.705, Validation Accuracy:  0.721, Loss:  0.387
    Epoch   8 Batch  158/269 - Train Accuracy:  0.706, Validation Accuracy:  0.718, Loss:  0.393
    Epoch   8 Batch  159/269 - Train Accuracy:  0.709, Validation Accuracy:  0.719, Loss:  0.388
    Epoch   8 Batch  160/269 - Train Accuracy:  0.707, Validation Accuracy:  0.721, Loss:  0.383
    Epoch   8 Batch  161/269 - Train Accuracy:  0.703, Validation Accuracy:  0.719, Loss:  0.392
    Epoch   8 Batch  162/269 - Train Accuracy:  0.709, Validation Accuracy:  0.720, Loss:  0.387
    Epoch   8 Batch  163/269 - Train Accuracy:  0.719, Validation Accuracy:  0.721, Loss:  0.388
    Epoch   8 Batch  164/269 - Train Accuracy:  0.702, Validation Accuracy:  0.721, Loss:  0.381
    Epoch   8 Batch  165/269 - Train Accuracy:  0.696, Validation Accuracy:  0.724, Loss:  0.396
    Epoch   8 Batch  166/269 - Train Accuracy:  0.730, Validation Accuracy:  0.729, Loss:  0.367
    Epoch   8 Batch  167/269 - Train Accuracy:  0.712, Validation Accuracy:  0.725, Loss:  0.381
    Epoch   8 Batch  168/269 - Train Accuracy:  0.699, Validation Accuracy:  0.723, Loss:  0.392
    Epoch   8 Batch  169/269 - Train Accuracy:  0.707, Validation Accuracy:  0.723, Loss:  0.387
    Epoch   8 Batch  170/269 - Train Accuracy:  0.718, Validation Accuracy:  0.724, Loss:  0.386
    Epoch   8 Batch  171/269 - Train Accuracy:  0.718, Validation Accuracy:  0.724, Loss:  0.403
    Epoch   8 Batch  172/269 - Train Accuracy:  0.714, Validation Accuracy:  0.723, Loss:  0.397
    Epoch   8 Batch  173/269 - Train Accuracy:  0.716, Validation Accuracy:  0.725, Loss:  0.373
    Epoch   8 Batch  174/269 - Train Accuracy:  0.704, Validation Accuracy:  0.724, Loss:  0.387
    Epoch   8 Batch  175/269 - Train Accuracy:  0.713, Validation Accuracy:  0.724, Loss:  0.399
    Epoch   8 Batch  176/269 - Train Accuracy:  0.692, Validation Accuracy:  0.724, Loss:  0.409
    Epoch   8 Batch  177/269 - Train Accuracy:  0.721, Validation Accuracy:  0.725, Loss:  0.375
    Epoch   8 Batch  178/269 - Train Accuracy:  0.703, Validation Accuracy:  0.725, Loss:  0.393
    Epoch   8 Batch  179/269 - Train Accuracy:  0.704, Validation Accuracy:  0.726, Loss:  0.383
    Epoch   8 Batch  180/269 - Train Accuracy:  0.717, Validation Accuracy:  0.725, Loss:  0.379
    Epoch   8 Batch  181/269 - Train Accuracy:  0.693, Validation Accuracy:  0.722, Loss:  0.388
    Epoch   8 Batch  182/269 - Train Accuracy:  0.722, Validation Accuracy:  0.721, Loss:  0.383
    Epoch   8 Batch  183/269 - Train Accuracy:  0.754, Validation Accuracy:  0.718, Loss:  0.336
    Epoch   8 Batch  184/269 - Train Accuracy:  0.696, Validation Accuracy:  0.722, Loss:  0.401
    Epoch   8 Batch  185/269 - Train Accuracy:  0.722, Validation Accuracy:  0.722, Loss:  0.376
    Epoch   8 Batch  186/269 - Train Accuracy:  0.697, Validation Accuracy:  0.718, Loss:  0.395
    Epoch   8 Batch  187/269 - Train Accuracy:  0.726, Validation Accuracy:  0.722, Loss:  0.380
    Epoch   8 Batch  188/269 - Train Accuracy:  0.733, Validation Accuracy:  0.725, Loss:  0.375
    Epoch   8 Batch  189/269 - Train Accuracy:  0.715, Validation Accuracy:  0.724, Loss:  0.374
    Epoch   8 Batch  190/269 - Train Accuracy:  0.713, Validation Accuracy:  0.726, Loss:  0.377
    Epoch   8 Batch  191/269 - Train Accuracy:  0.725, Validation Accuracy:  0.723, Loss:  0.376
    Epoch   8 Batch  192/269 - Train Accuracy:  0.718, Validation Accuracy:  0.723, Loss:  0.379
    Epoch   8 Batch  193/269 - Train Accuracy:  0.721, Validation Accuracy:  0.722, Loss:  0.376
    Epoch   8 Batch  194/269 - Train Accuracy:  0.731, Validation Accuracy:  0.724, Loss:  0.383
    Epoch   8 Batch  195/269 - Train Accuracy:  0.715, Validation Accuracy:  0.726, Loss:  0.384
    Epoch   8 Batch  196/269 - Train Accuracy:  0.709, Validation Accuracy:  0.722, Loss:  0.375
    Epoch   8 Batch  197/269 - Train Accuracy:  0.689, Validation Accuracy:  0.724, Loss:  0.401
    Epoch   8 Batch  198/269 - Train Accuracy:  0.695, Validation Accuracy:  0.724, Loss:  0.401
    Epoch   8 Batch  199/269 - Train Accuracy:  0.697, Validation Accuracy:  0.727, Loss:  0.381
    Epoch   8 Batch  200/269 - Train Accuracy:  0.725, Validation Accuracy:  0.730, Loss:  0.395
    Epoch   8 Batch  201/269 - Train Accuracy:  0.713, Validation Accuracy:  0.729, Loss:  0.379
    Epoch   8 Batch  202/269 - Train Accuracy:  0.701, Validation Accuracy:  0.728, Loss:  0.379
    Epoch   8 Batch  203/269 - Train Accuracy:  0.703, Validation Accuracy:  0.723, Loss:  0.407
    Epoch   8 Batch  204/269 - Train Accuracy:  0.700, Validation Accuracy:  0.724, Loss:  0.401
    Epoch   8 Batch  205/269 - Train Accuracy:  0.695, Validation Accuracy:  0.723, Loss:  0.376
    Epoch   8 Batch  206/269 - Train Accuracy:  0.700, Validation Accuracy:  0.723, Loss:  0.394
    Epoch   8 Batch  207/269 - Train Accuracy:  0.727, Validation Accuracy:  0.723, Loss:  0.369
    Epoch   8 Batch  208/269 - Train Accuracy:  0.703, Validation Accuracy:  0.728, Loss:  0.391
    Epoch   8 Batch  209/269 - Train Accuracy:  0.714, Validation Accuracy:  0.729, Loss:  0.390
    Epoch   8 Batch  210/269 - Train Accuracy:  0.716, Validation Accuracy:  0.725, Loss:  0.376
    Epoch   8 Batch  211/269 - Train Accuracy:  0.713, Validation Accuracy:  0.726, Loss:  0.387
    Epoch   8 Batch  212/269 - Train Accuracy:  0.721, Validation Accuracy:  0.727, Loss:  0.379
    Epoch   8 Batch  213/269 - Train Accuracy:  0.712, Validation Accuracy:  0.725, Loss:  0.381
    Epoch   8 Batch  214/269 - Train Accuracy:  0.727, Validation Accuracy:  0.725, Loss:  0.376
    Epoch   8 Batch  215/269 - Train Accuracy:  0.740, Validation Accuracy:  0.726, Loss:  0.361
    Epoch   8 Batch  216/269 - Train Accuracy:  0.704, Validation Accuracy:  0.728, Loss:  0.405
    Epoch   8 Batch  217/269 - Train Accuracy:  0.694, Validation Accuracy:  0.728, Loss:  0.394
    Epoch   8 Batch  218/269 - Train Accuracy:  0.700, Validation Accuracy:  0.730, Loss:  0.390
    Epoch   8 Batch  219/269 - Train Accuracy:  0.726, Validation Accuracy:  0.726, Loss:  0.396
    Epoch   8 Batch  220/269 - Train Accuracy:  0.719, Validation Accuracy:  0.726, Loss:  0.361
    Epoch   8 Batch  221/269 - Train Accuracy:  0.732, Validation Accuracy:  0.723, Loss:  0.379
    Epoch   8 Batch  222/269 - Train Accuracy:  0.729, Validation Accuracy:  0.728, Loss:  0.367
    Epoch   8 Batch  223/269 - Train Accuracy:  0.717, Validation Accuracy:  0.725, Loss:  0.369
    Epoch   8 Batch  224/269 - Train Accuracy:  0.730, Validation Accuracy:  0.726, Loss:  0.385
    Epoch   8 Batch  225/269 - Train Accuracy:  0.701, Validation Accuracy:  0.722, Loss:  0.378
    Epoch   8 Batch  226/269 - Train Accuracy:  0.708, Validation Accuracy:  0.725, Loss:  0.377
    Epoch   8 Batch  227/269 - Train Accuracy:  0.757, Validation Accuracy:  0.729, Loss:  0.340
    Epoch   8 Batch  228/269 - Train Accuracy:  0.713, Validation Accuracy:  0.726, Loss:  0.375
    Epoch   8 Batch  229/269 - Train Accuracy:  0.723, Validation Accuracy:  0.723, Loss:  0.374
    Epoch   8 Batch  230/269 - Train Accuracy:  0.710, Validation Accuracy:  0.728, Loss:  0.376
    Epoch   8 Batch  231/269 - Train Accuracy:  0.702, Validation Accuracy:  0.731, Loss:  0.399
    Epoch   8 Batch  232/269 - Train Accuracy:  0.694, Validation Accuracy:  0.727, Loss:  0.395
    Epoch   8 Batch  233/269 - Train Accuracy:  0.735, Validation Accuracy:  0.728, Loss:  0.382
    Epoch   8 Batch  234/269 - Train Accuracy:  0.713, Validation Accuracy:  0.725, Loss:  0.375
    Epoch   8 Batch  235/269 - Train Accuracy:  0.731, Validation Accuracy:  0.722, Loss:  0.364
    Epoch   8 Batch  236/269 - Train Accuracy:  0.702, Validation Accuracy:  0.719, Loss:  0.369
    Epoch   8 Batch  237/269 - Train Accuracy:  0.705, Validation Accuracy:  0.724, Loss:  0.375
    Epoch   8 Batch  238/269 - Train Accuracy:  0.729, Validation Accuracy:  0.727, Loss:  0.369
    Epoch   8 Batch  239/269 - Train Accuracy:  0.727, Validation Accuracy:  0.725, Loss:  0.376
    Epoch   8 Batch  240/269 - Train Accuracy:  0.738, Validation Accuracy:  0.728, Loss:  0.341
    Epoch   8 Batch  241/269 - Train Accuracy:  0.712, Validation Accuracy:  0.731, Loss:  0.380
    Epoch   8 Batch  242/269 - Train Accuracy:  0.713, Validation Accuracy:  0.730, Loss:  0.365
    Epoch   8 Batch  243/269 - Train Accuracy:  0.732, Validation Accuracy:  0.728, Loss:  0.364
    Epoch   8 Batch  244/269 - Train Accuracy:  0.715, Validation Accuracy:  0.726, Loss:  0.376
    Epoch   8 Batch  245/269 - Train Accuracy:  0.709, Validation Accuracy:  0.730, Loss:  0.390
    Epoch   8 Batch  246/269 - Train Accuracy:  0.701, Validation Accuracy:  0.732, Loss:  0.372
    Epoch   8 Batch  247/269 - Train Accuracy:  0.726, Validation Accuracy:  0.733, Loss:  0.379
    Epoch   8 Batch  248/269 - Train Accuracy:  0.716, Validation Accuracy:  0.729, Loss:  0.362
    Epoch   8 Batch  249/269 - Train Accuracy:  0.743, Validation Accuracy:  0.734, Loss:  0.347
    Epoch   8 Batch  250/269 - Train Accuracy:  0.710, Validation Accuracy:  0.735, Loss:  0.376
    Epoch   8 Batch  251/269 - Train Accuracy:  0.737, Validation Accuracy:  0.732, Loss:  0.367
    Epoch   8 Batch  252/269 - Train Accuracy:  0.713, Validation Accuracy:  0.729, Loss:  0.374
    Epoch   8 Batch  253/269 - Train Accuracy:  0.721, Validation Accuracy:  0.729, Loss:  0.385
    Epoch   8 Batch  254/269 - Train Accuracy:  0.716, Validation Accuracy:  0.727, Loss:  0.368
    Epoch   8 Batch  255/269 - Train Accuracy:  0.736, Validation Accuracy:  0.724, Loss:  0.354
    Epoch   8 Batch  256/269 - Train Accuracy:  0.694, Validation Accuracy:  0.726, Loss:  0.383
    Epoch   8 Batch  257/269 - Train Accuracy:  0.693, Validation Accuracy:  0.728, Loss:  0.385
    Epoch   8 Batch  258/269 - Train Accuracy:  0.704, Validation Accuracy:  0.729, Loss:  0.378
    Epoch   8 Batch  259/269 - Train Accuracy:  0.737, Validation Accuracy:  0.729, Loss:  0.375
    Epoch   8 Batch  260/269 - Train Accuracy:  0.706, Validation Accuracy:  0.729, Loss:  0.396
    Epoch   8 Batch  261/269 - Train Accuracy:  0.690, Validation Accuracy:  0.730, Loss:  0.389
    Epoch   8 Batch  262/269 - Train Accuracy:  0.728, Validation Accuracy:  0.729, Loss:  0.368
    Epoch   8 Batch  263/269 - Train Accuracy:  0.725, Validation Accuracy:  0.731, Loss:  0.383
    Epoch   8 Batch  264/269 - Train Accuracy:  0.711, Validation Accuracy:  0.732, Loss:  0.393
    Epoch   8 Batch  265/269 - Train Accuracy:  0.715, Validation Accuracy:  0.735, Loss:  0.378
    Epoch   8 Batch  266/269 - Train Accuracy:  0.728, Validation Accuracy:  0.738, Loss:  0.365
    Epoch   8 Batch  267/269 - Train Accuracy:  0.727, Validation Accuracy:  0.737, Loss:  0.380
    Epoch   9 Batch    0/269 - Train Accuracy:  0.718, Validation Accuracy:  0.737, Loss:  0.391
    Epoch   9 Batch    1/269 - Train Accuracy:  0.709, Validation Accuracy:  0.733, Loss:  0.375
    Epoch   9 Batch    2/269 - Train Accuracy:  0.704, Validation Accuracy:  0.734, Loss:  0.380
    Epoch   9 Batch    3/269 - Train Accuracy:  0.720, Validation Accuracy:  0.736, Loss:  0.380
    Epoch   9 Batch    4/269 - Train Accuracy:  0.704, Validation Accuracy:  0.733, Loss:  0.388
    Epoch   9 Batch    5/269 - Train Accuracy:  0.690, Validation Accuracy:  0.734, Loss:  0.386
    Epoch   9 Batch    6/269 - Train Accuracy:  0.725, Validation Accuracy:  0.734, Loss:  0.361
    Epoch   9 Batch    7/269 - Train Accuracy:  0.720, Validation Accuracy:  0.736, Loss:  0.365
    Epoch   9 Batch    8/269 - Train Accuracy:  0.705, Validation Accuracy:  0.737, Loss:  0.390
    Epoch   9 Batch    9/269 - Train Accuracy:  0.715, Validation Accuracy:  0.739, Loss:  0.381
    Epoch   9 Batch   10/269 - Train Accuracy:  0.718, Validation Accuracy:  0.739, Loss:  0.377
    Epoch   9 Batch   11/269 - Train Accuracy:  0.719, Validation Accuracy:  0.738, Loss:  0.381
    Epoch   9 Batch   12/269 - Train Accuracy:  0.697, Validation Accuracy:  0.741, Loss:  0.388
    Epoch   9 Batch   13/269 - Train Accuracy:  0.746, Validation Accuracy:  0.741, Loss:  0.343
    Epoch   9 Batch   14/269 - Train Accuracy:  0.717, Validation Accuracy:  0.737, Loss:  0.364
    Epoch   9 Batch   15/269 - Train Accuracy:  0.705, Validation Accuracy:  0.740, Loss:  0.358
    Epoch   9 Batch   16/269 - Train Accuracy:  0.729, Validation Accuracy:  0.741, Loss:  0.366
    Epoch   9 Batch   17/269 - Train Accuracy:  0.716, Validation Accuracy:  0.740, Loss:  0.360
    Epoch   9 Batch   18/269 - Train Accuracy:  0.707, Validation Accuracy:  0.736, Loss:  0.380
    Epoch   9 Batch   19/269 - Train Accuracy:  0.736, Validation Accuracy:  0.737, Loss:  0.343
    Epoch   9 Batch   20/269 - Train Accuracy:  0.720, Validation Accuracy:  0.735, Loss:  0.383
    Epoch   9 Batch   21/269 - Train Accuracy:  0.712, Validation Accuracy:  0.736, Loss:  0.403
    Epoch   9 Batch   22/269 - Train Accuracy:  0.742, Validation Accuracy:  0.741, Loss:  0.358
    Epoch   9 Batch   23/269 - Train Accuracy:  0.717, Validation Accuracy:  0.739, Loss:  0.363
    Epoch   9 Batch   24/269 - Train Accuracy:  0.711, Validation Accuracy:  0.739, Loss:  0.387
    Epoch   9 Batch   25/269 - Train Accuracy:  0.708, Validation Accuracy:  0.738, Loss:  0.396
    Epoch   9 Batch   26/269 - Train Accuracy:  0.736, Validation Accuracy:  0.743, Loss:  0.342
    Epoch   9 Batch   27/269 - Train Accuracy:  0.708, Validation Accuracy:  0.740, Loss:  0.368
    Epoch   9 Batch   28/269 - Train Accuracy:  0.687, Validation Accuracy:  0.740, Loss:  0.399
    Epoch   9 Batch   29/269 - Train Accuracy:  0.718, Validation Accuracy:  0.744, Loss:  0.385
    Epoch   9 Batch   30/269 - Train Accuracy:  0.728, Validation Accuracy:  0.737, Loss:  0.367
    Epoch   9 Batch   31/269 - Train Accuracy:  0.733, Validation Accuracy:  0.738, Loss:  0.354
    Epoch   9 Batch   32/269 - Train Accuracy:  0.719, Validation Accuracy:  0.737, Loss:  0.362
    Epoch   9 Batch   33/269 - Train Accuracy:  0.726, Validation Accuracy:  0.732, Loss:  0.356
    Epoch   9 Batch   34/269 - Train Accuracy:  0.722, Validation Accuracy:  0.728, Loss:  0.361
    Epoch   9 Batch   35/269 - Train Accuracy:  0.722, Validation Accuracy:  0.728, Loss:  0.384
    Epoch   9 Batch   36/269 - Train Accuracy:  0.708, Validation Accuracy:  0.726, Loss:  0.369
    Epoch   9 Batch   37/269 - Train Accuracy:  0.720, Validation Accuracy:  0.727, Loss:  0.363
    Epoch   9 Batch   38/269 - Train Accuracy:  0.726, Validation Accuracy:  0.727, Loss:  0.367
    Epoch   9 Batch   39/269 - Train Accuracy:  0.724, Validation Accuracy:  0.730, Loss:  0.369
    Epoch   9 Batch   40/269 - Train Accuracy:  0.717, Validation Accuracy:  0.731, Loss:  0.385
    Epoch   9 Batch   41/269 - Train Accuracy:  0.721, Validation Accuracy:  0.734, Loss:  0.369
    Epoch   9 Batch   42/269 - Train Accuracy:  0.736, Validation Accuracy:  0.739, Loss:  0.348
    Epoch   9 Batch   43/269 - Train Accuracy:  0.712, Validation Accuracy:  0.739, Loss:  0.370
    Epoch   9 Batch   44/269 - Train Accuracy:  0.726, Validation Accuracy:  0.735, Loss:  0.372
    Epoch   9 Batch   45/269 - Train Accuracy:  0.704, Validation Accuracy:  0.731, Loss:  0.383
    Epoch   9 Batch   46/269 - Train Accuracy:  0.725, Validation Accuracy:  0.733, Loss:  0.388
    Epoch   9 Batch   47/269 - Train Accuracy:  0.751, Validation Accuracy:  0.735, Loss:  0.349
    Epoch   9 Batch   48/269 - Train Accuracy:  0.732, Validation Accuracy:  0.729, Loss:  0.359
    Epoch   9 Batch   49/269 - Train Accuracy:  0.714, Validation Accuracy:  0.733, Loss:  0.366
    Epoch   9 Batch   50/269 - Train Accuracy:  0.714, Validation Accuracy:  0.731, Loss:  0.385
    Epoch   9 Batch   51/269 - Train Accuracy:  0.718, Validation Accuracy:  0.735, Loss:  0.366
    Epoch   9 Batch   52/269 - Train Accuracy:  0.712, Validation Accuracy:  0.731, Loss:  0.350
    Epoch   9 Batch   53/269 - Train Accuracy:  0.707, Validation Accuracy:  0.735, Loss:  0.387
    Epoch   9 Batch   54/269 - Train Accuracy:  0.733, Validation Accuracy:  0.734, Loss:  0.368
    Epoch   9 Batch   55/269 - Train Accuracy:  0.740, Validation Accuracy:  0.734, Loss:  0.364
    Epoch   9 Batch   56/269 - Train Accuracy:  0.734, Validation Accuracy:  0.733, Loss:  0.360
    Epoch   9 Batch   57/269 - Train Accuracy:  0.723, Validation Accuracy:  0.731, Loss:  0.369
    Epoch   9 Batch   58/269 - Train Accuracy:  0.727, Validation Accuracy:  0.733, Loss:  0.356
    Epoch   9 Batch   59/269 - Train Accuracy:  0.746, Validation Accuracy:  0.732, Loss:  0.338
    Epoch   9 Batch   60/269 - Train Accuracy:  0.720, Validation Accuracy:  0.725, Loss:  0.348
    Epoch   9 Batch   61/269 - Train Accuracy:  0.738, Validation Accuracy:  0.732, Loss:  0.338
    Epoch   9 Batch   62/269 - Train Accuracy:  0.730, Validation Accuracy:  0.738, Loss:  0.351
    Epoch   9 Batch   63/269 - Train Accuracy:  0.719, Validation Accuracy:  0.742, Loss:  0.365
    Epoch   9 Batch   64/269 - Train Accuracy:  0.723, Validation Accuracy:  0.741, Loss:  0.358
    Epoch   9 Batch   65/269 - Train Accuracy:  0.718, Validation Accuracy:  0.740, Loss:  0.361
    Epoch   9 Batch   66/269 - Train Accuracy:  0.730, Validation Accuracy:  0.742, Loss:  0.348
    Epoch   9 Batch   67/269 - Train Accuracy:  0.716, Validation Accuracy:  0.739, Loss:  0.372
    Epoch   9 Batch   68/269 - Train Accuracy:  0.702, Validation Accuracy:  0.744, Loss:  0.368
    Epoch   9 Batch   69/269 - Train Accuracy:  0.701, Validation Accuracy:  0.742, Loss:  0.400
    Epoch   9 Batch   70/269 - Train Accuracy:  0.753, Validation Accuracy:  0.739, Loss:  0.359
    Epoch   9 Batch   71/269 - Train Accuracy:  0.732, Validation Accuracy:  0.737, Loss:  0.380
    Epoch   9 Batch   72/269 - Train Accuracy:  0.728, Validation Accuracy:  0.731, Loss:  0.359
    Epoch   9 Batch   73/269 - Train Accuracy:  0.732, Validation Accuracy:  0.734, Loss:  0.373
    Epoch   9 Batch   74/269 - Train Accuracy:  0.721, Validation Accuracy:  0.732, Loss:  0.360
    Epoch   9 Batch   75/269 - Train Accuracy:  0.727, Validation Accuracy:  0.731, Loss:  0.359
    Epoch   9 Batch   76/269 - Train Accuracy:  0.717, Validation Accuracy:  0.734, Loss:  0.362
    Epoch   9 Batch   77/269 - Train Accuracy:  0.737, Validation Accuracy:  0.735, Loss:  0.358
    Epoch   9 Batch   78/269 - Train Accuracy:  0.734, Validation Accuracy:  0.739, Loss:  0.355
    Epoch   9 Batch   79/269 - Train Accuracy:  0.732, Validation Accuracy:  0.737, Loss:  0.363
    Epoch   9 Batch   80/269 - Train Accuracy:  0.742, Validation Accuracy:  0.734, Loss:  0.355
    Epoch   9 Batch   81/269 - Train Accuracy:  0.733, Validation Accuracy:  0.739, Loss:  0.371
    Epoch   9 Batch   82/269 - Train Accuracy:  0.741, Validation Accuracy:  0.737, Loss:  0.341
    Epoch   9 Batch   83/269 - Train Accuracy:  0.736, Validation Accuracy:  0.739, Loss:  0.377
    Epoch   9 Batch   84/269 - Train Accuracy:  0.740, Validation Accuracy:  0.740, Loss:  0.355
    Epoch   9 Batch   85/269 - Train Accuracy:  0.730, Validation Accuracy:  0.737, Loss:  0.361
    Epoch   9 Batch   86/269 - Train Accuracy:  0.716, Validation Accuracy:  0.737, Loss:  0.349
    Epoch   9 Batch   87/269 - Train Accuracy:  0.713, Validation Accuracy:  0.738, Loss:  0.379
    Epoch   9 Batch   88/269 - Train Accuracy:  0.730, Validation Accuracy:  0.737, Loss:  0.360
    Epoch   9 Batch   89/269 - Train Accuracy:  0.745, Validation Accuracy:  0.740, Loss:  0.355
    Epoch   9 Batch   90/269 - Train Accuracy:  0.690, Validation Accuracy:  0.739, Loss:  0.380
    Epoch   9 Batch   91/269 - Train Accuracy:  0.727, Validation Accuracy:  0.740, Loss:  0.345
    Epoch   9 Batch   92/269 - Train Accuracy:  0.723, Validation Accuracy:  0.740, Loss:  0.347
    Epoch   9 Batch   93/269 - Train Accuracy:  0.736, Validation Accuracy:  0.741, Loss:  0.349
    Epoch   9 Batch   94/269 - Train Accuracy:  0.723, Validation Accuracy:  0.738, Loss:  0.370
    Epoch   9 Batch   95/269 - Train Accuracy:  0.716, Validation Accuracy:  0.740, Loss:  0.366
    Epoch   9 Batch   96/269 - Train Accuracy:  0.724, Validation Accuracy:  0.740, Loss:  0.388
    Epoch   9 Batch   97/269 - Train Accuracy:  0.721, Validation Accuracy:  0.733, Loss:  0.361
    Epoch   9 Batch   98/269 - Train Accuracy:  0.725, Validation Accuracy:  0.737, Loss:  0.419
    Epoch   9 Batch   99/269 - Train Accuracy:  0.713, Validation Accuracy:  0.727, Loss:  0.377
    Epoch   9 Batch  100/269 - Train Accuracy:  0.737, Validation Accuracy:  0.730, Loss:  0.429
    Epoch   9 Batch  101/269 - Train Accuracy:  0.695, Validation Accuracy:  0.735, Loss:  0.439
    Epoch   9 Batch  102/269 - Train Accuracy:  0.722, Validation Accuracy:  0.732, Loss:  0.376
    Epoch   9 Batch  103/269 - Train Accuracy:  0.708, Validation Accuracy:  0.714, Loss:  0.374
    Epoch   9 Batch  104/269 - Train Accuracy:  0.710, Validation Accuracy:  0.718, Loss:  0.403
    Epoch   9 Batch  105/269 - Train Accuracy:  0.720, Validation Accuracy:  0.728, Loss:  0.391
    Epoch   9 Batch  106/269 - Train Accuracy:  0.710, Validation Accuracy:  0.729, Loss:  0.381
    Epoch   9 Batch  107/269 - Train Accuracy:  0.690, Validation Accuracy:  0.732, Loss:  0.396
    Epoch   9 Batch  108/269 - Train Accuracy:  0.723, Validation Accuracy:  0.736, Loss:  0.374
    Epoch   9 Batch  109/269 - Train Accuracy:  0.699, Validation Accuracy:  0.739, Loss:  0.385
    Epoch   9 Batch  110/269 - Train Accuracy:  0.711, Validation Accuracy:  0.736, Loss:  0.369
    Epoch   9 Batch  111/269 - Train Accuracy:  0.702, Validation Accuracy:  0.738, Loss:  0.392
    Epoch   9 Batch  112/269 - Train Accuracy:  0.727, Validation Accuracy:  0.740, Loss:  0.377
    Epoch   9 Batch  113/269 - Train Accuracy:  0.723, Validation Accuracy:  0.733, Loss:  0.351
    Epoch   9 Batch  114/269 - Train Accuracy:  0.726, Validation Accuracy:  0.737, Loss:  0.364
    Epoch   9 Batch  115/269 - Train Accuracy:  0.713, Validation Accuracy:  0.739, Loss:  0.391
    Epoch   9 Batch  116/269 - Train Accuracy:  0.729, Validation Accuracy:  0.733, Loss:  0.367
    Epoch   9 Batch  117/269 - Train Accuracy:  0.714, Validation Accuracy:  0.738, Loss:  0.362
    Epoch   9 Batch  118/269 - Train Accuracy:  0.756, Validation Accuracy:  0.736, Loss:  0.356
    Epoch   9 Batch  119/269 - Train Accuracy:  0.721, Validation Accuracy:  0.741, Loss:  0.376
    Epoch   9 Batch  120/269 - Train Accuracy:  0.718, Validation Accuracy:  0.736, Loss:  0.375
    Epoch   9 Batch  121/269 - Train Accuracy:  0.724, Validation Accuracy:  0.735, Loss:  0.357
    Epoch   9 Batch  122/269 - Train Accuracy:  0.724, Validation Accuracy:  0.738, Loss:  0.363
    Epoch   9 Batch  123/269 - Train Accuracy:  0.717, Validation Accuracy:  0.740, Loss:  0.376
    Epoch   9 Batch  124/269 - Train Accuracy:  0.725, Validation Accuracy:  0.743, Loss:  0.355
    Epoch   9 Batch  125/269 - Train Accuracy:  0.723, Validation Accuracy:  0.741, Loss:  0.350
    Epoch   9 Batch  126/269 - Train Accuracy:  0.728, Validation Accuracy:  0.742, Loss:  0.361
    Epoch   9 Batch  127/269 - Train Accuracy:  0.713, Validation Accuracy:  0.741, Loss:  0.369
    Epoch   9 Batch  128/269 - Train Accuracy:  0.744, Validation Accuracy:  0.737, Loss:  0.360
    Epoch   9 Batch  129/269 - Train Accuracy:  0.729, Validation Accuracy:  0.741, Loss:  0.357
    Epoch   9 Batch  130/269 - Train Accuracy:  0.721, Validation Accuracy:  0.741, Loss:  0.372
    Epoch   9 Batch  131/269 - Train Accuracy:  0.716, Validation Accuracy:  0.744, Loss:  0.368
    Epoch   9 Batch  132/269 - Train Accuracy:  0.716, Validation Accuracy:  0.745, Loss:  0.363
    Epoch   9 Batch  133/269 - Train Accuracy:  0.743, Validation Accuracy:  0.744, Loss:  0.345
    Epoch   9 Batch  134/269 - Train Accuracy:  0.719, Validation Accuracy:  0.744, Loss:  0.366
    Epoch   9 Batch  135/269 - Train Accuracy:  0.703, Validation Accuracy:  0.742, Loss:  0.379
    Epoch   9 Batch  136/269 - Train Accuracy:  0.710, Validation Accuracy:  0.743, Loss:  0.382
    Epoch   9 Batch  137/269 - Train Accuracy:  0.721, Validation Accuracy:  0.742, Loss:  0.374
    Epoch   9 Batch  138/269 - Train Accuracy:  0.738, Validation Accuracy:  0.737, Loss:  0.368
    Epoch   9 Batch  139/269 - Train Accuracy:  0.749, Validation Accuracy:  0.741, Loss:  0.343
    Epoch   9 Batch  140/269 - Train Accuracy:  0.733, Validation Accuracy:  0.743, Loss:  0.364
    Epoch   9 Batch  141/269 - Train Accuracy:  0.727, Validation Accuracy:  0.743, Loss:  0.366
    Epoch   9 Batch  142/269 - Train Accuracy:  0.733, Validation Accuracy:  0.743, Loss:  0.342
    Epoch   9 Batch  143/269 - Train Accuracy:  0.740, Validation Accuracy:  0.740, Loss:  0.352
    Epoch   9 Batch  144/269 - Train Accuracy:  0.734, Validation Accuracy:  0.742, Loss:  0.332
    Epoch   9 Batch  145/269 - Train Accuracy:  0.731, Validation Accuracy:  0.745, Loss:  0.349
    Epoch   9 Batch  146/269 - Train Accuracy:  0.727, Validation Accuracy:  0.748, Loss:  0.347
    Epoch   9 Batch  147/269 - Train Accuracy:  0.743, Validation Accuracy:  0.745, Loss:  0.338
    Epoch   9 Batch  148/269 - Train Accuracy:  0.736, Validation Accuracy:  0.742, Loss:  0.351
    Epoch   9 Batch  149/269 - Train Accuracy:  0.725, Validation Accuracy:  0.745, Loss:  0.358
    Epoch   9 Batch  150/269 - Train Accuracy:  0.734, Validation Accuracy:  0.745, Loss:  0.349
    Epoch   9 Batch  151/269 - Train Accuracy:  0.764, Validation Accuracy:  0.744, Loss:  0.343
    Epoch   9 Batch  152/269 - Train Accuracy:  0.732, Validation Accuracy:  0.746, Loss:  0.346
    Epoch   9 Batch  153/269 - Train Accuracy:  0.730, Validation Accuracy:  0.746, Loss:  0.349
    Epoch   9 Batch  154/269 - Train Accuracy:  0.728, Validation Accuracy:  0.748, Loss:  0.361
    Epoch   9 Batch  155/269 - Train Accuracy:  0.752, Validation Accuracy:  0.747, Loss:  0.335
    Epoch   9 Batch  156/269 - Train Accuracy:  0.729, Validation Accuracy:  0.748, Loss:  0.367
    Epoch   9 Batch  157/269 - Train Accuracy:  0.738, Validation Accuracy:  0.749, Loss:  0.344
    Epoch   9 Batch  158/269 - Train Accuracy:  0.738, Validation Accuracy:  0.749, Loss:  0.356
    Epoch   9 Batch  159/269 - Train Accuracy:  0.745, Validation Accuracy:  0.747, Loss:  0.350
    Epoch   9 Batch  160/269 - Train Accuracy:  0.741, Validation Accuracy:  0.747, Loss:  0.346
    Epoch   9 Batch  161/269 - Train Accuracy:  0.732, Validation Accuracy:  0.747, Loss:  0.350
    Epoch   9 Batch  162/269 - Train Accuracy:  0.744, Validation Accuracy:  0.743, Loss:  0.345
    Epoch   9 Batch  163/269 - Train Accuracy:  0.738, Validation Accuracy:  0.745, Loss:  0.353
    Epoch   9 Batch  164/269 - Train Accuracy:  0.736, Validation Accuracy:  0.748, Loss:  0.343
    Epoch   9 Batch  165/269 - Train Accuracy:  0.719, Validation Accuracy:  0.747, Loss:  0.355
    Epoch   9 Batch  166/269 - Train Accuracy:  0.750, Validation Accuracy:  0.748, Loss:  0.330
    Epoch   9 Batch  167/269 - Train Accuracy:  0.739, Validation Accuracy:  0.747, Loss:  0.349
    Epoch   9 Batch  168/269 - Train Accuracy:  0.726, Validation Accuracy:  0.743, Loss:  0.351
    Epoch   9 Batch  169/269 - Train Accuracy:  0.731, Validation Accuracy:  0.749, Loss:  0.352
    Epoch   9 Batch  170/269 - Train Accuracy:  0.746, Validation Accuracy:  0.745, Loss:  0.339
    Epoch   9 Batch  171/269 - Train Accuracy:  0.739, Validation Accuracy:  0.745, Loss:  0.364
    Epoch   9 Batch  172/269 - Train Accuracy:  0.725, Validation Accuracy:  0.743, Loss:  0.357
    Epoch   9 Batch  173/269 - Train Accuracy:  0.739, Validation Accuracy:  0.743, Loss:  0.338
    Epoch   9 Batch  174/269 - Train Accuracy:  0.731, Validation Accuracy:  0.744, Loss:  0.349
    Epoch   9 Batch  175/269 - Train Accuracy:  0.741, Validation Accuracy:  0.744, Loss:  0.361
    Epoch   9 Batch  176/269 - Train Accuracy:  0.718, Validation Accuracy:  0.746, Loss:  0.370
    Epoch   9 Batch  177/269 - Train Accuracy:  0.748, Validation Accuracy:  0.749, Loss:  0.338
    Epoch   9 Batch  178/269 - Train Accuracy:  0.738, Validation Accuracy:  0.748, Loss:  0.359
    Epoch   9 Batch  179/269 - Train Accuracy:  0.724, Validation Accuracy:  0.749, Loss:  0.346
    Epoch   9 Batch  180/269 - Train Accuracy:  0.743, Validation Accuracy:  0.750, Loss:  0.339
    Epoch   9 Batch  181/269 - Train Accuracy:  0.717, Validation Accuracy:  0.749, Loss:  0.351
    Epoch   9 Batch  182/269 - Train Accuracy:  0.744, Validation Accuracy:  0.750, Loss:  0.342
    Epoch   9 Batch  183/269 - Train Accuracy:  0.782, Validation Accuracy:  0.750, Loss:  0.304
    Epoch   9 Batch  184/269 - Train Accuracy:  0.733, Validation Accuracy:  0.748, Loss:  0.359
    Epoch   9 Batch  185/269 - Train Accuracy:  0.743, Validation Accuracy:  0.750, Loss:  0.340
    Epoch   9 Batch  186/269 - Train Accuracy:  0.730, Validation Accuracy:  0.747, Loss:  0.352
    Epoch   9 Batch  187/269 - Train Accuracy:  0.746, Validation Accuracy:  0.746, Loss:  0.337
    Epoch   9 Batch  188/269 - Train Accuracy:  0.757, Validation Accuracy:  0.748, Loss:  0.339
    Epoch   9 Batch  189/269 - Train Accuracy:  0.739, Validation Accuracy:  0.748, Loss:  0.339
    Epoch   9 Batch  190/269 - Train Accuracy:  0.743, Validation Accuracy:  0.744, Loss:  0.340
    Epoch   9 Batch  191/269 - Train Accuracy:  0.745, Validation Accuracy:  0.747, Loss:  0.340
    Epoch   9 Batch  192/269 - Train Accuracy:  0.744, Validation Accuracy:  0.747, Loss:  0.350
    Epoch   9 Batch  193/269 - Train Accuracy:  0.746, Validation Accuracy:  0.746, Loss:  0.337
    Epoch   9 Batch  194/269 - Train Accuracy:  0.747, Validation Accuracy:  0.739, Loss:  0.346
    Epoch   9 Batch  195/269 - Train Accuracy:  0.741, Validation Accuracy:  0.744, Loss:  0.352
    Epoch   9 Batch  196/269 - Train Accuracy:  0.725, Validation Accuracy:  0.747, Loss:  0.339
    Epoch   9 Batch  197/269 - Train Accuracy:  0.703, Validation Accuracy:  0.748, Loss:  0.375
    Epoch   9 Batch  198/269 - Train Accuracy:  0.723, Validation Accuracy:  0.750, Loss:  0.356
    Epoch   9 Batch  199/269 - Train Accuracy:  0.724, Validation Accuracy:  0.747, Loss:  0.347
    Epoch   9 Batch  200/269 - Train Accuracy:  0.744, Validation Accuracy:  0.744, Loss:  0.362
    Epoch   9 Batch  201/269 - Train Accuracy:  0.730, Validation Accuracy:  0.751, Loss:  0.347
    Epoch   9 Batch  202/269 - Train Accuracy:  0.722, Validation Accuracy:  0.748, Loss:  0.346
    Epoch   9 Batch  203/269 - Train Accuracy:  0.736, Validation Accuracy:  0.744, Loss:  0.372
    Epoch   9 Batch  204/269 - Train Accuracy:  0.720, Validation Accuracy:  0.747, Loss:  0.361
    Epoch   9 Batch  205/269 - Train Accuracy:  0.716, Validation Accuracy:  0.747, Loss:  0.341
    Epoch   9 Batch  206/269 - Train Accuracy:  0.739, Validation Accuracy:  0.749, Loss:  0.359
    Epoch   9 Batch  207/269 - Train Accuracy:  0.747, Validation Accuracy:  0.747, Loss:  0.335
    Epoch   9 Batch  208/269 - Train Accuracy:  0.735, Validation Accuracy:  0.751, Loss:  0.356
    Epoch   9 Batch  209/269 - Train Accuracy:  0.748, Validation Accuracy:  0.748, Loss:  0.345
    Epoch   9 Batch  210/269 - Train Accuracy:  0.750, Validation Accuracy:  0.745, Loss:  0.341
    Epoch   9 Batch  211/269 - Train Accuracy:  0.739, Validation Accuracy:  0.747, Loss:  0.349
    Epoch   9 Batch  212/269 - Train Accuracy:  0.751, Validation Accuracy:  0.756, Loss:  0.345
    Epoch   9 Batch  213/269 - Train Accuracy:  0.735, Validation Accuracy:  0.752, Loss:  0.352
    Epoch   9 Batch  214/269 - Train Accuracy:  0.748, Validation Accuracy:  0.749, Loss:  0.343
    Epoch   9 Batch  215/269 - Train Accuracy:  0.765, Validation Accuracy:  0.748, Loss:  0.325
    Epoch   9 Batch  216/269 - Train Accuracy:  0.716, Validation Accuracy:  0.747, Loss:  0.369
    Epoch   9 Batch  217/269 - Train Accuracy:  0.715, Validation Accuracy:  0.746, Loss:  0.366
    Epoch   9 Batch  218/269 - Train Accuracy:  0.727, Validation Accuracy:  0.750, Loss:  0.353
    Epoch   9 Batch  219/269 - Train Accuracy:  0.745, Validation Accuracy:  0.748, Loss:  0.365
    Epoch   9 Batch  220/269 - Train Accuracy:  0.741, Validation Accuracy:  0.752, Loss:  0.331
    Epoch   9 Batch  221/269 - Train Accuracy:  0.757, Validation Accuracy:  0.746, Loss:  0.342
    Epoch   9 Batch  222/269 - Train Accuracy:  0.755, Validation Accuracy:  0.752, Loss:  0.334
    Epoch   9 Batch  223/269 - Train Accuracy:  0.744, Validation Accuracy:  0.749, Loss:  0.329
    Epoch   9 Batch  224/269 - Train Accuracy:  0.755, Validation Accuracy:  0.748, Loss:  0.352
    Epoch   9 Batch  225/269 - Train Accuracy:  0.738, Validation Accuracy:  0.750, Loss:  0.340
    Epoch   9 Batch  226/269 - Train Accuracy:  0.738, Validation Accuracy:  0.749, Loss:  0.339
    Epoch   9 Batch  227/269 - Train Accuracy:  0.782, Validation Accuracy:  0.748, Loss:  0.311
    Epoch   9 Batch  228/269 - Train Accuracy:  0.730, Validation Accuracy:  0.748, Loss:  0.337
    Epoch   9 Batch  229/269 - Train Accuracy:  0.746, Validation Accuracy:  0.750, Loss:  0.333
    Epoch   9 Batch  230/269 - Train Accuracy:  0.734, Validation Accuracy:  0.749, Loss:  0.343
    Epoch   9 Batch  231/269 - Train Accuracy:  0.720, Validation Accuracy:  0.751, Loss:  0.357
    Epoch   9 Batch  232/269 - Train Accuracy:  0.722, Validation Accuracy:  0.748, Loss:  0.365
    Epoch   9 Batch  233/269 - Train Accuracy:  0.756, Validation Accuracy:  0.750, Loss:  0.345
    Epoch   9 Batch  234/269 - Train Accuracy:  0.741, Validation Accuracy:  0.746, Loss:  0.337
    Epoch   9 Batch  235/269 - Train Accuracy:  0.760, Validation Accuracy:  0.749, Loss:  0.330
    Epoch   9 Batch  236/269 - Train Accuracy:  0.725, Validation Accuracy:  0.746, Loss:  0.333
    Epoch   9 Batch  237/269 - Train Accuracy:  0.729, Validation Accuracy:  0.747, Loss:  0.340
    Epoch   9 Batch  238/269 - Train Accuracy:  0.746, Validation Accuracy:  0.749, Loss:  0.331
    Epoch   9 Batch  239/269 - Train Accuracy:  0.752, Validation Accuracy:  0.750, Loss:  0.337
    Epoch   9 Batch  240/269 - Train Accuracy:  0.751, Validation Accuracy:  0.748, Loss:  0.308
    Epoch   9 Batch  241/269 - Train Accuracy:  0.738, Validation Accuracy:  0.751, Loss:  0.351
    Epoch   9 Batch  242/269 - Train Accuracy:  0.717, Validation Accuracy:  0.749, Loss:  0.333
    Epoch   9 Batch  243/269 - Train Accuracy:  0.759, Validation Accuracy:  0.751, Loss:  0.326
    Epoch   9 Batch  244/269 - Train Accuracy:  0.742, Validation Accuracy:  0.752, Loss:  0.347
    Epoch   9 Batch  245/269 - Train Accuracy:  0.741, Validation Accuracy:  0.756, Loss:  0.356
    Epoch   9 Batch  246/269 - Train Accuracy:  0.723, Validation Accuracy:  0.755, Loss:  0.340
    Epoch   9 Batch  247/269 - Train Accuracy:  0.748, Validation Accuracy:  0.754, Loss:  0.344
    Epoch   9 Batch  248/269 - Train Accuracy:  0.742, Validation Accuracy:  0.755, Loss:  0.332
    Epoch   9 Batch  249/269 - Train Accuracy:  0.762, Validation Accuracy:  0.755, Loss:  0.317
    Epoch   9 Batch  250/269 - Train Accuracy:  0.741, Validation Accuracy:  0.757, Loss:  0.343
    Epoch   9 Batch  251/269 - Train Accuracy:  0.761, Validation Accuracy:  0.757, Loss:  0.331
    Epoch   9 Batch  252/269 - Train Accuracy:  0.739, Validation Accuracy:  0.755, Loss:  0.340
    Epoch   9 Batch  253/269 - Train Accuracy:  0.743, Validation Accuracy:  0.756, Loss:  0.352
    Epoch   9 Batch  254/269 - Train Accuracy:  0.740, Validation Accuracy:  0.754, Loss:  0.336
    Epoch   9 Batch  255/269 - Train Accuracy:  0.752, Validation Accuracy:  0.754, Loss:  0.325
    Epoch   9 Batch  256/269 - Train Accuracy:  0.715, Validation Accuracy:  0.752, Loss:  0.341
    Epoch   9 Batch  257/269 - Train Accuracy:  0.716, Validation Accuracy:  0.751, Loss:  0.349
    Epoch   9 Batch  258/269 - Train Accuracy:  0.734, Validation Accuracy:  0.748, Loss:  0.344
    Epoch   9 Batch  259/269 - Train Accuracy:  0.751, Validation Accuracy:  0.743, Loss:  0.336
    Epoch   9 Batch  260/269 - Train Accuracy:  0.733, Validation Accuracy:  0.749, Loss:  0.359
    Epoch   9 Batch  261/269 - Train Accuracy:  0.720, Validation Accuracy:  0.754, Loss:  0.352
    Epoch   9 Batch  262/269 - Train Accuracy:  0.750, Validation Accuracy:  0.752, Loss:  0.331
    Epoch   9 Batch  263/269 - Train Accuracy:  0.746, Validation Accuracy:  0.750, Loss:  0.347
    Epoch   9 Batch  264/269 - Train Accuracy:  0.730, Validation Accuracy:  0.752, Loss:  0.352
    Epoch   9 Batch  265/269 - Train Accuracy:  0.740, Validation Accuracy:  0.752, Loss:  0.344
    Epoch   9 Batch  266/269 - Train Accuracy:  0.738, Validation Accuracy:  0.752, Loss:  0.327
    Epoch   9 Batch  267/269 - Train Accuracy:  0.743, Validation Accuracy:  0.756, Loss:  0.343
    Epoch  10 Batch    0/269 - Train Accuracy:  0.733, Validation Accuracy:  0.755, Loss:  0.355
    Epoch  10 Batch    1/269 - Train Accuracy:  0.733, Validation Accuracy:  0.753, Loss:  0.338
    Epoch  10 Batch    2/269 - Train Accuracy:  0.732, Validation Accuracy:  0.749, Loss:  0.348
    Epoch  10 Batch    3/269 - Train Accuracy:  0.746, Validation Accuracy:  0.754, Loss:  0.342
    Epoch  10 Batch    4/269 - Train Accuracy:  0.730, Validation Accuracy:  0.754, Loss:  0.354
    Epoch  10 Batch    5/269 - Train Accuracy:  0.720, Validation Accuracy:  0.757, Loss:  0.351
    Epoch  10 Batch    6/269 - Train Accuracy:  0.748, Validation Accuracy:  0.755, Loss:  0.326
    Epoch  10 Batch    7/269 - Train Accuracy:  0.746, Validation Accuracy:  0.755, Loss:  0.329
    Epoch  10 Batch    8/269 - Train Accuracy:  0.733, Validation Accuracy:  0.755, Loss:  0.354
    Epoch  10 Batch    9/269 - Train Accuracy:  0.729, Validation Accuracy:  0.759, Loss:  0.350
    Epoch  10 Batch   10/269 - Train Accuracy:  0.741, Validation Accuracy:  0.755, Loss:  0.338
    Epoch  10 Batch   11/269 - Train Accuracy:  0.737, Validation Accuracy:  0.752, Loss:  0.343
    Epoch  10 Batch   12/269 - Train Accuracy:  0.721, Validation Accuracy:  0.750, Loss:  0.355
    Epoch  10 Batch   13/269 - Train Accuracy:  0.764, Validation Accuracy:  0.752, Loss:  0.305
    Epoch  10 Batch   14/269 - Train Accuracy:  0.729, Validation Accuracy:  0.752, Loss:  0.335
    Epoch  10 Batch   15/269 - Train Accuracy:  0.729, Validation Accuracy:  0.754, Loss:  0.320
    Epoch  10 Batch   16/269 - Train Accuracy:  0.758, Validation Accuracy:  0.754, Loss:  0.334
    Epoch  10 Batch   17/269 - Train Accuracy:  0.737, Validation Accuracy:  0.755, Loss:  0.324
    Epoch  10 Batch   18/269 - Train Accuracy:  0.739, Validation Accuracy:  0.755, Loss:  0.345
    Epoch  10 Batch   19/269 - Train Accuracy:  0.763, Validation Accuracy:  0.760, Loss:  0.310
    Epoch  10 Batch   20/269 - Train Accuracy:  0.747, Validation Accuracy:  0.757, Loss:  0.347
    Epoch  10 Batch   21/269 - Train Accuracy:  0.739, Validation Accuracy:  0.756, Loss:  0.357
    Epoch  10 Batch   22/269 - Train Accuracy:  0.752, Validation Accuracy:  0.757, Loss:  0.322
    Epoch  10 Batch   23/269 - Train Accuracy:  0.743, Validation Accuracy:  0.759, Loss:  0.328
    Epoch  10 Batch   24/269 - Train Accuracy:  0.734, Validation Accuracy:  0.758, Loss:  0.347
    Epoch  10 Batch   25/269 - Train Accuracy:  0.729, Validation Accuracy:  0.757, Loss:  0.358
    Epoch  10 Batch   26/269 - Train Accuracy:  0.756, Validation Accuracy:  0.754, Loss:  0.310
    Epoch  10 Batch   27/269 - Train Accuracy:  0.733, Validation Accuracy:  0.755, Loss:  0.328
    Epoch  10 Batch   28/269 - Train Accuracy:  0.712, Validation Accuracy:  0.755, Loss:  0.361
    Epoch  10 Batch   29/269 - Train Accuracy:  0.736, Validation Accuracy:  0.756, Loss:  0.350
    Epoch  10 Batch   30/269 - Train Accuracy:  0.743, Validation Accuracy:  0.753, Loss:  0.331
    Epoch  10 Batch   31/269 - Train Accuracy:  0.758, Validation Accuracy:  0.755, Loss:  0.318
    Epoch  10 Batch   32/269 - Train Accuracy:  0.724, Validation Accuracy:  0.752, Loss:  0.326
    Epoch  10 Batch   33/269 - Train Accuracy:  0.755, Validation Accuracy:  0.751, Loss:  0.320
    Epoch  10 Batch   34/269 - Train Accuracy:  0.748, Validation Accuracy:  0.752, Loss:  0.329
    Epoch  10 Batch   35/269 - Train Accuracy:  0.748, Validation Accuracy:  0.746, Loss:  0.348
    Epoch  10 Batch   36/269 - Train Accuracy:  0.734, Validation Accuracy:  0.748, Loss:  0.334
    Epoch  10 Batch   37/269 - Train Accuracy:  0.751, Validation Accuracy:  0.751, Loss:  0.328
    Epoch  10 Batch   38/269 - Train Accuracy:  0.744, Validation Accuracy:  0.754, Loss:  0.326
    Epoch  10 Batch   39/269 - Train Accuracy:  0.748, Validation Accuracy:  0.754, Loss:  0.336
    Epoch  10 Batch   40/269 - Train Accuracy:  0.743, Validation Accuracy:  0.758, Loss:  0.342
    Epoch  10 Batch   41/269 - Train Accuracy:  0.745, Validation Accuracy:  0.757, Loss:  0.341
    Epoch  10 Batch   42/269 - Train Accuracy:  0.758, Validation Accuracy:  0.755, Loss:  0.308
    Epoch  10 Batch   43/269 - Train Accuracy:  0.742, Validation Accuracy:  0.760, Loss:  0.340
    Epoch  10 Batch   44/269 - Train Accuracy:  0.760, Validation Accuracy:  0.759, Loss:  0.327
    Epoch  10 Batch   45/269 - Train Accuracy:  0.732, Validation Accuracy:  0.755, Loss:  0.342
    Epoch  10 Batch   46/269 - Train Accuracy:  0.749, Validation Accuracy:  0.758, Loss:  0.342
    Epoch  10 Batch   47/269 - Train Accuracy:  0.767, Validation Accuracy:  0.759, Loss:  0.312
    Epoch  10 Batch   48/269 - Train Accuracy:  0.754, Validation Accuracy:  0.758, Loss:  0.320
    Epoch  10 Batch   49/269 - Train Accuracy:  0.739, Validation Accuracy:  0.755, Loss:  0.329
    Epoch  10 Batch   50/269 - Train Accuracy:  0.740, Validation Accuracy:  0.762, Loss:  0.346
    Epoch  10 Batch   51/269 - Train Accuracy:  0.741, Validation Accuracy:  0.762, Loss:  0.327
    Epoch  10 Batch   52/269 - Train Accuracy:  0.735, Validation Accuracy:  0.756, Loss:  0.312
    Epoch  10 Batch   53/269 - Train Accuracy:  0.735, Validation Accuracy:  0.757, Loss:  0.346
    Epoch  10 Batch   54/269 - Train Accuracy:  0.754, Validation Accuracy:  0.757, Loss:  0.335
    Epoch  10 Batch   55/269 - Train Accuracy:  0.757, Validation Accuracy:  0.756, Loss:  0.328
    Epoch  10 Batch   56/269 - Train Accuracy:  0.756, Validation Accuracy:  0.754, Loss:  0.328
    Epoch  10 Batch   57/269 - Train Accuracy:  0.740, Validation Accuracy:  0.756, Loss:  0.335
    Epoch  10 Batch   58/269 - Train Accuracy:  0.747, Validation Accuracy:  0.755, Loss:  0.325
    Epoch  10 Batch   59/269 - Train Accuracy:  0.767, Validation Accuracy:  0.752, Loss:  0.303
    Epoch  10 Batch   60/269 - Train Accuracy:  0.740, Validation Accuracy:  0.752, Loss:  0.317
    Epoch  10 Batch   61/269 - Train Accuracy:  0.762, Validation Accuracy:  0.753, Loss:  0.303
    Epoch  10 Batch   62/269 - Train Accuracy:  0.751, Validation Accuracy:  0.755, Loss:  0.316
    Epoch  10 Batch   63/269 - Train Accuracy:  0.742, Validation Accuracy:  0.754, Loss:  0.329
    Epoch  10 Batch   64/269 - Train Accuracy:  0.738, Validation Accuracy:  0.754, Loss:  0.320
    Epoch  10 Batch   65/269 - Train Accuracy:  0.737, Validation Accuracy:  0.756, Loss:  0.326
    Epoch  10 Batch   66/269 - Train Accuracy:  0.744, Validation Accuracy:  0.758, Loss:  0.316
    Epoch  10 Batch   67/269 - Train Accuracy:  0.738, Validation Accuracy:  0.759, Loss:  0.338
    Epoch  10 Batch   68/269 - Train Accuracy:  0.733, Validation Accuracy:  0.758, Loss:  0.333
    Epoch  10 Batch   69/269 - Train Accuracy:  0.732, Validation Accuracy:  0.759, Loss:  0.366
    Epoch  10 Batch   70/269 - Train Accuracy:  0.775, Validation Accuracy:  0.752, Loss:  0.327
    Epoch  10 Batch   71/269 - Train Accuracy:  0.746, Validation Accuracy:  0.753, Loss:  0.343
    Epoch  10 Batch   72/269 - Train Accuracy:  0.744, Validation Accuracy:  0.755, Loss:  0.325
    Epoch  10 Batch   73/269 - Train Accuracy:  0.747, Validation Accuracy:  0.757, Loss:  0.335
    Epoch  10 Batch   74/269 - Train Accuracy:  0.740, Validation Accuracy:  0.753, Loss:  0.327
    Epoch  10 Batch   75/269 - Train Accuracy:  0.743, Validation Accuracy:  0.756, Loss:  0.327
    Epoch  10 Batch   76/269 - Train Accuracy:  0.735, Validation Accuracy:  0.755, Loss:  0.332
    Epoch  10 Batch   77/269 - Train Accuracy:  0.759, Validation Accuracy:  0.757, Loss:  0.328
    Epoch  10 Batch   78/269 - Train Accuracy:  0.758, Validation Accuracy:  0.758, Loss:  0.324
    Epoch  10 Batch   79/269 - Train Accuracy:  0.748, Validation Accuracy:  0.755, Loss:  0.327
    Epoch  10 Batch   80/269 - Train Accuracy:  0.760, Validation Accuracy:  0.755, Loss:  0.320
    Epoch  10 Batch   81/269 - Train Accuracy:  0.749, Validation Accuracy:  0.756, Loss:  0.334
    Epoch  10 Batch   82/269 - Train Accuracy:  0.759, Validation Accuracy:  0.757, Loss:  0.309
    Epoch  10 Batch   83/269 - Train Accuracy:  0.750, Validation Accuracy:  0.757, Loss:  0.336
    Epoch  10 Batch   84/269 - Train Accuracy:  0.758, Validation Accuracy:  0.759, Loss:  0.322
    Epoch  10 Batch   85/269 - Train Accuracy:  0.754, Validation Accuracy:  0.756, Loss:  0.327
    Epoch  10 Batch   86/269 - Train Accuracy:  0.736, Validation Accuracy:  0.754, Loss:  0.317
    Epoch  10 Batch   87/269 - Train Accuracy:  0.730, Validation Accuracy:  0.757, Loss:  0.348
    Epoch  10 Batch   88/269 - Train Accuracy:  0.747, Validation Accuracy:  0.756, Loss:  0.326
    Epoch  10 Batch   89/269 - Train Accuracy:  0.752, Validation Accuracy:  0.749, Loss:  0.324
    Epoch  10 Batch   90/269 - Train Accuracy:  0.705, Validation Accuracy:  0.755, Loss:  0.351
    Epoch  10 Batch   91/269 - Train Accuracy:  0.739, Validation Accuracy:  0.757, Loss:  0.316
    Epoch  10 Batch   92/269 - Train Accuracy:  0.741, Validation Accuracy:  0.750, Loss:  0.321
    Epoch  10 Batch   93/269 - Train Accuracy:  0.754, Validation Accuracy:  0.746, Loss:  0.307
    Epoch  10 Batch   94/269 - Train Accuracy:  0.747, Validation Accuracy:  0.754, Loss:  0.336
    Epoch  10 Batch   95/269 - Train Accuracy:  0.740, Validation Accuracy:  0.754, Loss:  0.326
    Epoch  10 Batch   96/269 - Train Accuracy:  0.747, Validation Accuracy:  0.754, Loss:  0.327
    Epoch  10 Batch   97/269 - Train Accuracy:  0.745, Validation Accuracy:  0.752, Loss:  0.326
    Epoch  10 Batch   98/269 - Train Accuracy:  0.751, Validation Accuracy:  0.752, Loss:  0.328
    Epoch  10 Batch   99/269 - Train Accuracy:  0.743, Validation Accuracy:  0.749, Loss:  0.337
    Epoch  10 Batch  100/269 - Train Accuracy:  0.772, Validation Accuracy:  0.750, Loss:  0.321
    Epoch  10 Batch  101/269 - Train Accuracy:  0.713, Validation Accuracy:  0.748, Loss:  0.354
    Epoch  10 Batch  102/269 - Train Accuracy:  0.753, Validation Accuracy:  0.753, Loss:  0.328
    Epoch  10 Batch  103/269 - Train Accuracy:  0.740, Validation Accuracy:  0.749, Loss:  0.320
    Epoch  10 Batch  104/269 - Train Accuracy:  0.748, Validation Accuracy:  0.757, Loss:  0.332
    Epoch  10 Batch  105/269 - Train Accuracy:  0.741, Validation Accuracy:  0.754, Loss:  0.331
    Epoch  10 Batch  106/269 - Train Accuracy:  0.744, Validation Accuracy:  0.754, Loss:  0.325
    Epoch  10 Batch  107/269 - Train Accuracy:  0.724, Validation Accuracy:  0.762, Loss:  0.346
    Epoch  10 Batch  108/269 - Train Accuracy:  0.740, Validation Accuracy:  0.756, Loss:  0.319
    Epoch  10 Batch  109/269 - Train Accuracy:  0.716, Validation Accuracy:  0.758, Loss:  0.334
    Epoch  10 Batch  110/269 - Train Accuracy:  0.739, Validation Accuracy:  0.760, Loss:  0.323
    Epoch  10 Batch  111/269 - Train Accuracy:  0.735, Validation Accuracy:  0.755, Loss:  0.343
    Epoch  10 Batch  112/269 - Train Accuracy:  0.750, Validation Accuracy:  0.754, Loss:  0.328
    Epoch  10 Batch  113/269 - Train Accuracy:  0.738, Validation Accuracy:  0.751, Loss:  0.310
    Epoch  10 Batch  114/269 - Train Accuracy:  0.738, Validation Accuracy:  0.755, Loss:  0.322
    Epoch  10 Batch  115/269 - Train Accuracy:  0.720, Validation Accuracy:  0.754, Loss:  0.340
    Epoch  10 Batch  116/269 - Train Accuracy:  0.746, Validation Accuracy:  0.754, Loss:  0.321
    Epoch  10 Batch  117/269 - Train Accuracy:  0.743, Validation Accuracy:  0.754, Loss:  0.319
    Epoch  10 Batch  118/269 - Train Accuracy:  0.775, Validation Accuracy:  0.756, Loss:  0.309
    Epoch  10 Batch  119/269 - Train Accuracy:  0.751, Validation Accuracy:  0.759, Loss:  0.335
    Epoch  10 Batch  120/269 - Train Accuracy:  0.744, Validation Accuracy:  0.759, Loss:  0.330
    Epoch  10 Batch  121/269 - Train Accuracy:  0.747, Validation Accuracy:  0.757, Loss:  0.315
    Epoch  10 Batch  122/269 - Train Accuracy:  0.753, Validation Accuracy:  0.759, Loss:  0.317
    Epoch  10 Batch  123/269 - Train Accuracy:  0.737, Validation Accuracy:  0.750, Loss:  0.334
    Epoch  10 Batch  124/269 - Train Accuracy:  0.750, Validation Accuracy:  0.752, Loss:  0.316
    Epoch  10 Batch  125/269 - Train Accuracy:  0.738, Validation Accuracy:  0.753, Loss:  0.316
    Epoch  10 Batch  126/269 - Train Accuracy:  0.749, Validation Accuracy:  0.757, Loss:  0.320
    Epoch  10 Batch  127/269 - Train Accuracy:  0.726, Validation Accuracy:  0.754, Loss:  0.332
    Epoch  10 Batch  128/269 - Train Accuracy:  0.761, Validation Accuracy:  0.756, Loss:  0.322
    Epoch  10 Batch  129/269 - Train Accuracy:  0.751, Validation Accuracy:  0.757, Loss:  0.315
    Epoch  10 Batch  130/269 - Train Accuracy:  0.741, Validation Accuracy:  0.760, Loss:  0.335
    Epoch  10 Batch  131/269 - Train Accuracy:  0.739, Validation Accuracy:  0.762, Loss:  0.327
    Epoch  10 Batch  132/269 - Train Accuracy:  0.737, Validation Accuracy:  0.763, Loss:  0.327
    Epoch  10 Batch  133/269 - Train Accuracy:  0.761, Validation Accuracy:  0.762, Loss:  0.310
    Epoch  10 Batch  134/269 - Train Accuracy:  0.733, Validation Accuracy:  0.762, Loss:  0.331
    Epoch  10 Batch  135/269 - Train Accuracy:  0.738, Validation Accuracy:  0.759, Loss:  0.347
    Epoch  10 Batch  136/269 - Train Accuracy:  0.729, Validation Accuracy:  0.759, Loss:  0.338
    Epoch  10 Batch  137/269 - Train Accuracy:  0.743, Validation Accuracy:  0.760, Loss:  0.338
    Epoch  10 Batch  138/269 - Train Accuracy:  0.750, Validation Accuracy:  0.756, Loss:  0.328
    Epoch  10 Batch  139/269 - Train Accuracy:  0.771, Validation Accuracy:  0.759, Loss:  0.313
    Epoch  10 Batch  140/269 - Train Accuracy:  0.748, Validation Accuracy:  0.756, Loss:  0.331
    Epoch  10 Batch  141/269 - Train Accuracy:  0.743, Validation Accuracy:  0.755, Loss:  0.333
    Epoch  10 Batch  142/269 - Train Accuracy:  0.759, Validation Accuracy:  0.758, Loss:  0.308
    Epoch  10 Batch  143/269 - Train Accuracy:  0.760, Validation Accuracy:  0.759, Loss:  0.315
    Epoch  10 Batch  144/269 - Train Accuracy:  0.753, Validation Accuracy:  0.758, Loss:  0.306
    Epoch  10 Batch  145/269 - Train Accuracy:  0.753, Validation Accuracy:  0.756, Loss:  0.316
    Epoch  10 Batch  146/269 - Train Accuracy:  0.755, Validation Accuracy:  0.762, Loss:  0.310
    Epoch  10 Batch  147/269 - Train Accuracy:  0.762, Validation Accuracy:  0.760, Loss:  0.304
    Epoch  10 Batch  148/269 - Train Accuracy:  0.752, Validation Accuracy:  0.756, Loss:  0.317
    Epoch  10 Batch  149/269 - Train Accuracy:  0.745, Validation Accuracy:  0.758, Loss:  0.328
    Epoch  10 Batch  150/269 - Train Accuracy:  0.749, Validation Accuracy:  0.760, Loss:  0.316
    Epoch  10 Batch  151/269 - Train Accuracy:  0.775, Validation Accuracy:  0.763, Loss:  0.309
    Epoch  10 Batch  152/269 - Train Accuracy:  0.750, Validation Accuracy:  0.764, Loss:  0.316
    Epoch  10 Batch  153/269 - Train Accuracy:  0.758, Validation Accuracy:  0.762, Loss:  0.317
    Epoch  10 Batch  154/269 - Train Accuracy:  0.742, Validation Accuracy:  0.759, Loss:  0.322
    Epoch  10 Batch  155/269 - Train Accuracy:  0.765, Validation Accuracy:  0.756, Loss:  0.306
    Epoch  10 Batch  156/269 - Train Accuracy:  0.746, Validation Accuracy:  0.762, Loss:  0.331
    Epoch  10 Batch  157/269 - Train Accuracy:  0.755, Validation Accuracy:  0.763, Loss:  0.313
    Epoch  10 Batch  158/269 - Train Accuracy:  0.750, Validation Accuracy:  0.760, Loss:  0.319
    Epoch  10 Batch  159/269 - Train Accuracy:  0.757, Validation Accuracy:  0.757, Loss:  0.317
    Epoch  10 Batch  160/269 - Train Accuracy:  0.763, Validation Accuracy:  0.764, Loss:  0.313
    Epoch  10 Batch  161/269 - Train Accuracy:  0.753, Validation Accuracy:  0.759, Loss:  0.317
    Epoch  10 Batch  162/269 - Train Accuracy:  0.761, Validation Accuracy:  0.762, Loss:  0.317
    Epoch  10 Batch  163/269 - Train Accuracy:  0.759, Validation Accuracy:  0.762, Loss:  0.322
    Epoch  10 Batch  164/269 - Train Accuracy:  0.758, Validation Accuracy:  0.763, Loss:  0.313
    Epoch  10 Batch  165/269 - Train Accuracy:  0.732, Validation Accuracy:  0.765, Loss:  0.326
    Epoch  10 Batch  166/269 - Train Accuracy:  0.761, Validation Accuracy:  0.767, Loss:  0.300
    Epoch  10 Batch  167/269 - Train Accuracy:  0.757, Validation Accuracy:  0.766, Loss:  0.316
    Epoch  10 Batch  168/269 - Train Accuracy:  0.745, Validation Accuracy:  0.765, Loss:  0.320
    Epoch  10 Batch  169/269 - Train Accuracy:  0.744, Validation Accuracy:  0.764, Loss:  0.324
    Epoch  10 Batch  170/269 - Train Accuracy:  0.756, Validation Accuracy:  0.762, Loss:  0.309
    Epoch  10 Batch  171/269 - Train Accuracy:  0.758, Validation Accuracy:  0.763, Loss:  0.331
    Epoch  10 Batch  172/269 - Train Accuracy:  0.739, Validation Accuracy:  0.761, Loss:  0.330
    Epoch  10 Batch  173/269 - Train Accuracy:  0.753, Validation Accuracy:  0.760, Loss:  0.307
    Epoch  10 Batch  174/269 - Train Accuracy:  0.753, Validation Accuracy:  0.757, Loss:  0.320
    Epoch  10 Batch  175/269 - Train Accuracy:  0.757, Validation Accuracy:  0.760, Loss:  0.327
    Epoch  10 Batch  176/269 - Train Accuracy:  0.739, Validation Accuracy:  0.760, Loss:  0.331
    Epoch  10 Batch  177/269 - Train Accuracy:  0.767, Validation Accuracy:  0.764, Loss:  0.304
    Epoch  10 Batch  178/269 - Train Accuracy:  0.750, Validation Accuracy:  0.760, Loss:  0.320
    Epoch  10 Batch  179/269 - Train Accuracy:  0.748, Validation Accuracy:  0.758, Loss:  0.317
    Epoch  10 Batch  180/269 - Train Accuracy:  0.752, Validation Accuracy:  0.757, Loss:  0.310
    Epoch  10 Batch  181/269 - Train Accuracy:  0.737, Validation Accuracy:  0.760, Loss:  0.318
    Epoch  10 Batch  182/269 - Train Accuracy:  0.761, Validation Accuracy:  0.762, Loss:  0.316
    Epoch  10 Batch  183/269 - Train Accuracy:  0.793, Validation Accuracy:  0.761, Loss:  0.277
    Epoch  10 Batch  184/269 - Train Accuracy:  0.746, Validation Accuracy:  0.762, Loss:  0.330
    Epoch  10 Batch  185/269 - Train Accuracy:  0.763, Validation Accuracy:  0.764, Loss:  0.313
    Epoch  10 Batch  186/269 - Train Accuracy:  0.740, Validation Accuracy:  0.766, Loss:  0.314
    Epoch  10 Batch  187/269 - Train Accuracy:  0.753, Validation Accuracy:  0.763, Loss:  0.310
    Epoch  10 Batch  188/269 - Train Accuracy:  0.773, Validation Accuracy:  0.757, Loss:  0.305
    Epoch  10 Batch  189/269 - Train Accuracy:  0.757, Validation Accuracy:  0.762, Loss:  0.306
    Epoch  10 Batch  190/269 - Train Accuracy:  0.763, Validation Accuracy:  0.762, Loss:  0.307
    Epoch  10 Batch  191/269 - Train Accuracy:  0.762, Validation Accuracy:  0.757, Loss:  0.311
    Epoch  10 Batch  192/269 - Train Accuracy:  0.757, Validation Accuracy:  0.758, Loss:  0.321
    Epoch  10 Batch  193/269 - Train Accuracy:  0.758, Validation Accuracy:  0.760, Loss:  0.308
    Epoch  10 Batch  194/269 - Train Accuracy:  0.769, Validation Accuracy:  0.762, Loss:  0.320
    Epoch  10 Batch  195/269 - Train Accuracy:  0.751, Validation Accuracy:  0.750, Loss:  0.316
    Epoch  10 Batch  196/269 - Train Accuracy:  0.738, Validation Accuracy:  0.761, Loss:  0.316
    Epoch  10 Batch  197/269 - Train Accuracy:  0.720, Validation Accuracy:  0.758, Loss:  0.334
    Epoch  10 Batch  198/269 - Train Accuracy:  0.733, Validation Accuracy:  0.763, Loss:  0.332
    Epoch  10 Batch  199/269 - Train Accuracy:  0.744, Validation Accuracy:  0.759, Loss:  0.318
    Epoch  10 Batch  200/269 - Train Accuracy:  0.764, Validation Accuracy:  0.760, Loss:  0.321
    Epoch  10 Batch  201/269 - Train Accuracy:  0.743, Validation Accuracy:  0.756, Loss:  0.316
    Epoch  10 Batch  202/269 - Train Accuracy:  0.753, Validation Accuracy:  0.757, Loss:  0.318
    Epoch  10 Batch  203/269 - Train Accuracy:  0.752, Validation Accuracy:  0.757, Loss:  0.338
    Epoch  10 Batch  204/269 - Train Accuracy:  0.739, Validation Accuracy:  0.758, Loss:  0.334
    Epoch  10 Batch  205/269 - Train Accuracy:  0.735, Validation Accuracy:  0.759, Loss:  0.312
    Epoch  10 Batch  206/269 - Train Accuracy:  0.756, Validation Accuracy:  0.759, Loss:  0.322
    Epoch  10 Batch  207/269 - Train Accuracy:  0.762, Validation Accuracy:  0.762, Loss:  0.303
    Epoch  10 Batch  208/269 - Train Accuracy:  0.747, Validation Accuracy:  0.761, Loss:  0.324
    Epoch  10 Batch  209/269 - Train Accuracy:  0.762, Validation Accuracy:  0.763, Loss:  0.313
    Epoch  10 Batch  210/269 - Train Accuracy:  0.759, Validation Accuracy:  0.762, Loss:  0.308
    Epoch  10 Batch  211/269 - Train Accuracy:  0.756, Validation Accuracy:  0.762, Loss:  0.315
    Epoch  10 Batch  212/269 - Train Accuracy:  0.765, Validation Accuracy:  0.766, Loss:  0.312
    Epoch  10 Batch  213/269 - Train Accuracy:  0.752, Validation Accuracy:  0.765, Loss:  0.313
    Epoch  10 Batch  214/269 - Train Accuracy:  0.766, Validation Accuracy:  0.768, Loss:  0.308
    Epoch  10 Batch  215/269 - Train Accuracy:  0.783, Validation Accuracy:  0.763, Loss:  0.291
    Epoch  10 Batch  216/269 - Train Accuracy:  0.744, Validation Accuracy:  0.766, Loss:  0.338
    Epoch  10 Batch  217/269 - Train Accuracy:  0.736, Validation Accuracy:  0.765, Loss:  0.324
    Epoch  10 Batch  218/269 - Train Accuracy:  0.752, Validation Accuracy:  0.763, Loss:  0.316
    Epoch  10 Batch  219/269 - Train Accuracy:  0.765, Validation Accuracy:  0.766, Loss:  0.323
    Epoch  10 Batch  220/269 - Train Accuracy:  0.749, Validation Accuracy:  0.766, Loss:  0.293
    Epoch  10 Batch  221/269 - Train Accuracy:  0.778, Validation Accuracy:  0.766, Loss:  0.311
    Epoch  10 Batch  222/269 - Train Accuracy:  0.762, Validation Accuracy:  0.763, Loss:  0.304
    Epoch  10 Batch  223/269 - Train Accuracy:  0.752, Validation Accuracy:  0.758, Loss:  0.298
    Epoch  10 Batch  224/269 - Train Accuracy:  0.759, Validation Accuracy:  0.761, Loss:  0.320
    Epoch  10 Batch  225/269 - Train Accuracy:  0.753, Validation Accuracy:  0.762, Loss:  0.307
    Epoch  10 Batch  226/269 - Train Accuracy:  0.747, Validation Accuracy:  0.759, Loss:  0.311
    Epoch  10 Batch  227/269 - Train Accuracy:  0.793, Validation Accuracy:  0.760, Loss:  0.284
    Epoch  10 Batch  228/269 - Train Accuracy:  0.747, Validation Accuracy:  0.760, Loss:  0.305
    Epoch  10 Batch  229/269 - Train Accuracy:  0.758, Validation Accuracy:  0.761, Loss:  0.304
    Epoch  10 Batch  230/269 - Train Accuracy:  0.744, Validation Accuracy:  0.761, Loss:  0.311
    Epoch  10 Batch  231/269 - Train Accuracy:  0.732, Validation Accuracy:  0.760, Loss:  0.331
    Epoch  10 Batch  232/269 - Train Accuracy:  0.731, Validation Accuracy:  0.752, Loss:  0.326
    Epoch  10 Batch  233/269 - Train Accuracy:  0.765, Validation Accuracy:  0.757, Loss:  0.318
    Epoch  10 Batch  234/269 - Train Accuracy:  0.754, Validation Accuracy:  0.758, Loss:  0.308
    Epoch  10 Batch  235/269 - Train Accuracy:  0.776, Validation Accuracy:  0.761, Loss:  0.296
    Epoch  10 Batch  236/269 - Train Accuracy:  0.746, Validation Accuracy:  0.760, Loss:  0.306
    Epoch  10 Batch  237/269 - Train Accuracy:  0.741, Validation Accuracy:  0.761, Loss:  0.309
    Epoch  10 Batch  238/269 - Train Accuracy:  0.765, Validation Accuracy:  0.757, Loss:  0.300
    Epoch  10 Batch  239/269 - Train Accuracy:  0.765, Validation Accuracy:  0.756, Loss:  0.309
    Epoch  10 Batch  240/269 - Train Accuracy:  0.770, Validation Accuracy:  0.759, Loss:  0.281
    Epoch  10 Batch  241/269 - Train Accuracy:  0.748, Validation Accuracy:  0.763, Loss:  0.316
    Epoch  10 Batch  242/269 - Train Accuracy:  0.740, Validation Accuracy:  0.764, Loss:  0.304
    Epoch  10 Batch  243/269 - Train Accuracy:  0.771, Validation Accuracy:  0.761, Loss:  0.295
    Epoch  10 Batch  244/269 - Train Accuracy:  0.758, Validation Accuracy:  0.763, Loss:  0.313
    Epoch  10 Batch  245/269 - Train Accuracy:  0.752, Validation Accuracy:  0.764, Loss:  0.318
    Epoch  10 Batch  246/269 - Train Accuracy:  0.734, Validation Accuracy:  0.761, Loss:  0.308
    Epoch  10 Batch  247/269 - Train Accuracy:  0.755, Validation Accuracy:  0.763, Loss:  0.314
    Epoch  10 Batch  248/269 - Train Accuracy:  0.754, Validation Accuracy:  0.763, Loss:  0.305
    Epoch  10 Batch  249/269 - Train Accuracy:  0.778, Validation Accuracy:  0.761, Loss:  0.286
    Epoch  10 Batch  250/269 - Train Accuracy:  0.756, Validation Accuracy:  0.762, Loss:  0.310
    Epoch  10 Batch  251/269 - Train Accuracy:  0.773, Validation Accuracy:  0.769, Loss:  0.297
    Epoch  10 Batch  252/269 - Train Accuracy:  0.751, Validation Accuracy:  0.769, Loss:  0.309
    Epoch  10 Batch  253/269 - Train Accuracy:  0.750, Validation Accuracy:  0.768, Loss:  0.320
    Epoch  10 Batch  254/269 - Train Accuracy:  0.758, Validation Accuracy:  0.767, Loss:  0.309
    Epoch  10 Batch  255/269 - Train Accuracy:  0.772, Validation Accuracy:  0.758, Loss:  0.291
    Epoch  10 Batch  256/269 - Train Accuracy:  0.730, Validation Accuracy:  0.762, Loss:  0.309
    Epoch  10 Batch  257/269 - Train Accuracy:  0.733, Validation Accuracy:  0.766, Loss:  0.321
    Epoch  10 Batch  258/269 - Train Accuracy:  0.742, Validation Accuracy:  0.765, Loss:  0.318
    Epoch  10 Batch  259/269 - Train Accuracy:  0.768, Validation Accuracy:  0.765, Loss:  0.304
    Epoch  10 Batch  260/269 - Train Accuracy:  0.745, Validation Accuracy:  0.764, Loss:  0.324
    Epoch  10 Batch  261/269 - Train Accuracy:  0.736, Validation Accuracy:  0.765, Loss:  0.323
    Epoch  10 Batch  262/269 - Train Accuracy:  0.767, Validation Accuracy:  0.762, Loss:  0.303
    Epoch  10 Batch  263/269 - Train Accuracy:  0.759, Validation Accuracy:  0.762, Loss:  0.316
    Epoch  10 Batch  264/269 - Train Accuracy:  0.747, Validation Accuracy:  0.759, Loss:  0.318
    Epoch  10 Batch  265/269 - Train Accuracy:  0.749, Validation Accuracy:  0.760, Loss:  0.312
    Epoch  10 Batch  266/269 - Train Accuracy:  0.751, Validation Accuracy:  0.759, Loss:  0.304
    Epoch  10 Batch  267/269 - Train Accuracy:  0.756, Validation Accuracy:  0.760, Loss:  0.317
    Epoch  11 Batch    0/269 - Train Accuracy:  0.750, Validation Accuracy:  0.761, Loss:  0.326
    Epoch  11 Batch    1/269 - Train Accuracy:  0.748, Validation Accuracy:  0.760, Loss:  0.310
    Epoch  11 Batch    2/269 - Train Accuracy:  0.742, Validation Accuracy:  0.756, Loss:  0.315
    Epoch  11 Batch    3/269 - Train Accuracy:  0.753, Validation Accuracy:  0.756, Loss:  0.312
    Epoch  11 Batch    4/269 - Train Accuracy:  0.735, Validation Accuracy:  0.759, Loss:  0.324
    Epoch  11 Batch    5/269 - Train Accuracy:  0.729, Validation Accuracy:  0.761, Loss:  0.319
    Epoch  11 Batch    6/269 - Train Accuracy:  0.764, Validation Accuracy:  0.759, Loss:  0.296
    Epoch  11 Batch    7/269 - Train Accuracy:  0.763, Validation Accuracy:  0.764, Loss:  0.301
    Epoch  11 Batch    8/269 - Train Accuracy:  0.737, Validation Accuracy:  0.765, Loss:  0.324
    Epoch  11 Batch    9/269 - Train Accuracy:  0.745, Validation Accuracy:  0.764, Loss:  0.318
    Epoch  11 Batch   10/269 - Train Accuracy:  0.755, Validation Accuracy:  0.764, Loss:  0.314
    Epoch  11 Batch   11/269 - Train Accuracy:  0.758, Validation Accuracy:  0.764, Loss:  0.310
    Epoch  11 Batch   12/269 - Train Accuracy:  0.738, Validation Accuracy:  0.765, Loss:  0.319
    Epoch  11 Batch   13/269 - Train Accuracy:  0.769, Validation Accuracy:  0.764, Loss:  0.280
    Epoch  11 Batch   14/269 - Train Accuracy:  0.741, Validation Accuracy:  0.765, Loss:  0.308
    Epoch  11 Batch   15/269 - Train Accuracy:  0.743, Validation Accuracy:  0.765, Loss:  0.297
    Epoch  11 Batch   16/269 - Train Accuracy:  0.773, Validation Accuracy:  0.762, Loss:  0.301
    Epoch  11 Batch   17/269 - Train Accuracy:  0.750, Validation Accuracy:  0.762, Loss:  0.296
    Epoch  11 Batch   18/269 - Train Accuracy:  0.751, Validation Accuracy:  0.763, Loss:  0.318
    Epoch  11 Batch   19/269 - Train Accuracy:  0.776, Validation Accuracy:  0.764, Loss:  0.283
    Epoch  11 Batch   20/269 - Train Accuracy:  0.758, Validation Accuracy:  0.763, Loss:  0.312
    Epoch  11 Batch   21/269 - Train Accuracy:  0.744, Validation Accuracy:  0.763, Loss:  0.329
    Epoch  11 Batch   22/269 - Train Accuracy:  0.770, Validation Accuracy:  0.770, Loss:  0.297
    Epoch  11 Batch   23/269 - Train Accuracy:  0.762, Validation Accuracy:  0.769, Loss:  0.292
    Epoch  11 Batch   24/269 - Train Accuracy:  0.741, Validation Accuracy:  0.766, Loss:  0.317
    Epoch  11 Batch   25/269 - Train Accuracy:  0.744, Validation Accuracy:  0.766, Loss:  0.331
    Epoch  11 Batch   26/269 - Train Accuracy:  0.769, Validation Accuracy:  0.769, Loss:  0.280
    Epoch  11 Batch   27/269 - Train Accuracy:  0.747, Validation Accuracy:  0.768, Loss:  0.300
    Epoch  11 Batch   28/269 - Train Accuracy:  0.719, Validation Accuracy:  0.768, Loss:  0.330
    Epoch  11 Batch   29/269 - Train Accuracy:  0.749, Validation Accuracy:  0.767, Loss:  0.326
    Epoch  11 Batch   30/269 - Train Accuracy:  0.744, Validation Accuracy:  0.763, Loss:  0.299
    Epoch  11 Batch   31/269 - Train Accuracy:  0.768, Validation Accuracy:  0.761, Loss:  0.289
    Epoch  11 Batch   32/269 - Train Accuracy:  0.747, Validation Accuracy:  0.761, Loss:  0.294
    Epoch  11 Batch   33/269 - Train Accuracy:  0.763, Validation Accuracy:  0.764, Loss:  0.291
    Epoch  11 Batch   34/269 - Train Accuracy:  0.760, Validation Accuracy:  0.766, Loss:  0.301
    Epoch  11 Batch   35/269 - Train Accuracy:  0.760, Validation Accuracy:  0.763, Loss:  0.312
    Epoch  11 Batch   36/269 - Train Accuracy:  0.755, Validation Accuracy:  0.760, Loss:  0.301
    Epoch  11 Batch   37/269 - Train Accuracy:  0.761, Validation Accuracy:  0.761, Loss:  0.300
    Epoch  11 Batch   38/269 - Train Accuracy:  0.756, Validation Accuracy:  0.762, Loss:  0.299
    Epoch  11 Batch   39/269 - Train Accuracy:  0.759, Validation Accuracy:  0.762, Loss:  0.307
    Epoch  11 Batch   40/269 - Train Accuracy:  0.752, Validation Accuracy:  0.764, Loss:  0.316
    Epoch  11 Batch   41/269 - Train Accuracy:  0.752, Validation Accuracy:  0.761, Loss:  0.310
    Epoch  11 Batch   42/269 - Train Accuracy:  0.765, Validation Accuracy:  0.763, Loss:  0.289
    Epoch  11 Batch   43/269 - Train Accuracy:  0.747, Validation Accuracy:  0.762, Loss:  0.309
    Epoch  11 Batch   44/269 - Train Accuracy:  0.765, Validation Accuracy:  0.759, Loss:  0.302
    Epoch  11 Batch   45/269 - Train Accuracy:  0.747, Validation Accuracy:  0.759, Loss:  0.311
    Epoch  11 Batch   46/269 - Train Accuracy:  0.757, Validation Accuracy:  0.760, Loss:  0.310
    Epoch  11 Batch   47/269 - Train Accuracy:  0.784, Validation Accuracy:  0.766, Loss:  0.284
    Epoch  11 Batch   48/269 - Train Accuracy:  0.763, Validation Accuracy:  0.764, Loss:  0.293
    Epoch  11 Batch   49/269 - Train Accuracy:  0.749, Validation Accuracy:  0.765, Loss:  0.298
    Epoch  11 Batch   50/269 - Train Accuracy:  0.747, Validation Accuracy:  0.769, Loss:  0.319
    Epoch  11 Batch   51/269 - Train Accuracy:  0.754, Validation Accuracy:  0.766, Loss:  0.304
    Epoch  11 Batch   52/269 - Train Accuracy:  0.752, Validation Accuracy:  0.767, Loss:  0.289
    Epoch  11 Batch   53/269 - Train Accuracy:  0.750, Validation Accuracy:  0.764, Loss:  0.316
    Epoch  11 Batch   54/269 - Train Accuracy:  0.760, Validation Accuracy:  0.763, Loss:  0.303
    Epoch  11 Batch   55/269 - Train Accuracy:  0.762, Validation Accuracy:  0.763, Loss:  0.302
    Epoch  11 Batch   56/269 - Train Accuracy:  0.766, Validation Accuracy:  0.763, Loss:  0.306
    Epoch  11 Batch   57/269 - Train Accuracy:  0.752, Validation Accuracy:  0.764, Loss:  0.310
    Epoch  11 Batch   58/269 - Train Accuracy:  0.756, Validation Accuracy:  0.763, Loss:  0.297
    Epoch  11 Batch   59/269 - Train Accuracy:  0.778, Validation Accuracy:  0.763, Loss:  0.285
    Epoch  11 Batch   60/269 - Train Accuracy:  0.751, Validation Accuracy:  0.763, Loss:  0.289
    Epoch  11 Batch   61/269 - Train Accuracy:  0.773, Validation Accuracy:  0.764, Loss:  0.279
    Epoch  11 Batch   62/269 - Train Accuracy:  0.760, Validation Accuracy:  0.764, Loss:  0.289
    Epoch  11 Batch   63/269 - Train Accuracy:  0.756, Validation Accuracy:  0.764, Loss:  0.301
    Epoch  11 Batch   64/269 - Train Accuracy:  0.755, Validation Accuracy:  0.761, Loss:  0.296
    Epoch  11 Batch   65/269 - Train Accuracy:  0.753, Validation Accuracy:  0.766, Loss:  0.302
    Epoch  11 Batch   66/269 - Train Accuracy:  0.752, Validation Accuracy:  0.768, Loss:  0.293
    Epoch  11 Batch   67/269 - Train Accuracy:  0.747, Validation Accuracy:  0.768, Loss:  0.312
    Epoch  11 Batch   68/269 - Train Accuracy:  0.740, Validation Accuracy:  0.768, Loss:  0.306
    Epoch  11 Batch   69/269 - Train Accuracy:  0.736, Validation Accuracy:  0.764, Loss:  0.337
    Epoch  11 Batch   70/269 - Train Accuracy:  0.787, Validation Accuracy:  0.763, Loss:  0.301
    Epoch  11 Batch   71/269 - Train Accuracy:  0.751, Validation Accuracy:  0.756, Loss:  0.313
    Epoch  11 Batch   72/269 - Train Accuracy:  0.750, Validation Accuracy:  0.760, Loss:  0.301
    Epoch  11 Batch   73/269 - Train Accuracy:  0.755, Validation Accuracy:  0.760, Loss:  0.305
    Epoch  11 Batch   74/269 - Train Accuracy:  0.755, Validation Accuracy:  0.763, Loss:  0.300
    Epoch  11 Batch   75/269 - Train Accuracy:  0.751, Validation Accuracy:  0.764, Loss:  0.299
    Epoch  11 Batch   76/269 - Train Accuracy:  0.747, Validation Accuracy:  0.763, Loss:  0.302
    Epoch  11 Batch   77/269 - Train Accuracy:  0.766, Validation Accuracy:  0.759, Loss:  0.297
    Epoch  11 Batch   78/269 - Train Accuracy:  0.767, Validation Accuracy:  0.763, Loss:  0.294
    Epoch  11 Batch   79/269 - Train Accuracy:  0.756, Validation Accuracy:  0.760, Loss:  0.300
    Epoch  11 Batch   80/269 - Train Accuracy:  0.767, Validation Accuracy:  0.761, Loss:  0.289
    Epoch  11 Batch   81/269 - Train Accuracy:  0.758, Validation Accuracy:  0.761, Loss:  0.303
    Epoch  11 Batch   82/269 - Train Accuracy:  0.774, Validation Accuracy:  0.761, Loss:  0.279
    Epoch  11 Batch   83/269 - Train Accuracy:  0.753, Validation Accuracy:  0.760, Loss:  0.317
    Epoch  11 Batch   84/269 - Train Accuracy:  0.767, Validation Accuracy:  0.765, Loss:  0.296
    Epoch  11 Batch   85/269 - Train Accuracy:  0.766, Validation Accuracy:  0.762, Loss:  0.298
    Epoch  11 Batch   86/269 - Train Accuracy:  0.745, Validation Accuracy:  0.764, Loss:  0.294
    Epoch  11 Batch   87/269 - Train Accuracy:  0.743, Validation Accuracy:  0.762, Loss:  0.318
    Epoch  11 Batch   88/269 - Train Accuracy:  0.756, Validation Accuracy:  0.763, Loss:  0.297
    Epoch  11 Batch   89/269 - Train Accuracy:  0.768, Validation Accuracy:  0.763, Loss:  0.298
    Epoch  11 Batch   90/269 - Train Accuracy:  0.721, Validation Accuracy:  0.764, Loss:  0.316
    Epoch  11 Batch   91/269 - Train Accuracy:  0.757, Validation Accuracy:  0.765, Loss:  0.286
    Epoch  11 Batch   92/269 - Train Accuracy:  0.750, Validation Accuracy:  0.765, Loss:  0.289
    Epoch  11 Batch   93/269 - Train Accuracy:  0.763, Validation Accuracy:  0.763, Loss:  0.281
    Epoch  11 Batch   94/269 - Train Accuracy:  0.751, Validation Accuracy:  0.761, Loss:  0.310
    Epoch  11 Batch   95/269 - Train Accuracy:  0.750, Validation Accuracy:  0.764, Loss:  0.296
    Epoch  11 Batch   96/269 - Train Accuracy:  0.758, Validation Accuracy:  0.764, Loss:  0.300
    Epoch  11 Batch   97/269 - Train Accuracy:  0.764, Validation Accuracy:  0.766, Loss:  0.291
    Epoch  11 Batch   98/269 - Train Accuracy:  0.760, Validation Accuracy:  0.764, Loss:  0.294
    Epoch  11 Batch   99/269 - Train Accuracy:  0.761, Validation Accuracy:  0.765, Loss:  0.297
    Epoch  11 Batch  100/269 - Train Accuracy:  0.784, Validation Accuracy:  0.762, Loss:  0.288
    Epoch  11 Batch  101/269 - Train Accuracy:  0.725, Validation Accuracy:  0.759, Loss:  0.318
    Epoch  11 Batch  102/269 - Train Accuracy:  0.757, Validation Accuracy:  0.763, Loss:  0.296
    Epoch  11 Batch  103/269 - Train Accuracy:  0.746, Validation Accuracy:  0.761, Loss:  0.295
    Epoch  11 Batch  104/269 - Train Accuracy:  0.759, Validation Accuracy:  0.762, Loss:  0.298
    Epoch  11 Batch  105/269 - Train Accuracy:  0.753, Validation Accuracy:  0.762, Loss:  0.300
    Epoch  11 Batch  106/269 - Train Accuracy:  0.754, Validation Accuracy:  0.763, Loss:  0.292
    Epoch  11 Batch  107/269 - Train Accuracy:  0.733, Validation Accuracy:  0.765, Loss:  0.312
    Epoch  11 Batch  108/269 - Train Accuracy:  0.756, Validation Accuracy:  0.768, Loss:  0.290
    Epoch  11 Batch  109/269 - Train Accuracy:  0.733, Validation Accuracy:  0.766, Loss:  0.298
    Epoch  11 Batch  110/269 - Train Accuracy:  0.751, Validation Accuracy:  0.764, Loss:  0.290
    Epoch  11 Batch  111/269 - Train Accuracy:  0.747, Validation Accuracy:  0.763, Loss:  0.309
    Epoch  11 Batch  112/269 - Train Accuracy:  0.762, Validation Accuracy:  0.763, Loss:  0.297
    Epoch  11 Batch  113/269 - Train Accuracy:  0.752, Validation Accuracy:  0.761, Loss:  0.287
    Epoch  11 Batch  114/269 - Train Accuracy:  0.747, Validation Accuracy:  0.763, Loss:  0.295
    Epoch  11 Batch  115/269 - Train Accuracy:  0.726, Validation Accuracy:  0.762, Loss:  0.317
    Epoch  11 Batch  116/269 - Train Accuracy:  0.760, Validation Accuracy:  0.760, Loss:  0.297
    Epoch  11 Batch  117/269 - Train Accuracy:  0.755, Validation Accuracy:  0.761, Loss:  0.292
    Epoch  11 Batch  118/269 - Train Accuracy:  0.786, Validation Accuracy:  0.755, Loss:  0.284
    Epoch  11 Batch  119/269 - Train Accuracy:  0.754, Validation Accuracy:  0.759, Loss:  0.307
    Epoch  11 Batch  120/269 - Train Accuracy:  0.758, Validation Accuracy:  0.761, Loss:  0.310
    Epoch  11 Batch  121/269 - Train Accuracy:  0.754, Validation Accuracy:  0.761, Loss:  0.290
    Epoch  11 Batch  122/269 - Train Accuracy:  0.761, Validation Accuracy:  0.760, Loss:  0.294
    Epoch  11 Batch  123/269 - Train Accuracy:  0.742, Validation Accuracy:  0.760, Loss:  0.304
    Epoch  11 Batch  124/269 - Train Accuracy:  0.753, Validation Accuracy:  0.761, Loss:  0.284
    Epoch  11 Batch  125/269 - Train Accuracy:  0.750, Validation Accuracy:  0.759, Loss:  0.293
    Epoch  11 Batch  126/269 - Train Accuracy:  0.759, Validation Accuracy:  0.760, Loss:  0.296
    Epoch  11 Batch  127/269 - Train Accuracy:  0.738, Validation Accuracy:  0.758, Loss:  0.304
    Epoch  11 Batch  128/269 - Train Accuracy:  0.772, Validation Accuracy:  0.766, Loss:  0.298
    Epoch  11 Batch  129/269 - Train Accuracy:  0.757, Validation Accuracy:  0.767, Loss:  0.288
    Epoch  11 Batch  130/269 - Train Accuracy:  0.751, Validation Accuracy:  0.769, Loss:  0.308
    Epoch  11 Batch  131/269 - Train Accuracy:  0.748, Validation Accuracy:  0.766, Loss:  0.297
    Epoch  11 Batch  132/269 - Train Accuracy:  0.750, Validation Accuracy:  0.768, Loss:  0.300
    Epoch  11 Batch  133/269 - Train Accuracy:  0.768, Validation Accuracy:  0.766, Loss:  0.283
    Epoch  11 Batch  134/269 - Train Accuracy:  0.740, Validation Accuracy:  0.763, Loss:  0.299
    Epoch  11 Batch  135/269 - Train Accuracy:  0.740, Validation Accuracy:  0.758, Loss:  0.317
    Epoch  11 Batch  136/269 - Train Accuracy:  0.735, Validation Accuracy:  0.760, Loss:  0.314
    Epoch  11 Batch  137/269 - Train Accuracy:  0.757, Validation Accuracy:  0.761, Loss:  0.310
    Epoch  11 Batch  138/269 - Train Accuracy:  0.752, Validation Accuracy:  0.763, Loss:  0.303
    Epoch  11 Batch  139/269 - Train Accuracy:  0.771, Validation Accuracy:  0.764, Loss:  0.288
    Epoch  11 Batch  140/269 - Train Accuracy:  0.757, Validation Accuracy:  0.765, Loss:  0.306
    Epoch  11 Batch  141/269 - Train Accuracy:  0.754, Validation Accuracy:  0.763, Loss:  0.304
    Epoch  11 Batch  142/269 - Train Accuracy:  0.768, Validation Accuracy:  0.764, Loss:  0.283
    Epoch  11 Batch  143/269 - Train Accuracy:  0.769, Validation Accuracy:  0.768, Loss:  0.291
    Epoch  11 Batch  144/269 - Train Accuracy:  0.762, Validation Accuracy:  0.766, Loss:  0.279
    Epoch  11 Batch  145/269 - Train Accuracy:  0.759, Validation Accuracy:  0.764, Loss:  0.291
    Epoch  11 Batch  146/269 - Train Accuracy:  0.768, Validation Accuracy:  0.767, Loss:  0.285
    Epoch  11 Batch  147/269 - Train Accuracy:  0.766, Validation Accuracy:  0.764, Loss:  0.284
    Epoch  11 Batch  148/269 - Train Accuracy:  0.762, Validation Accuracy:  0.766, Loss:  0.296
    Epoch  11 Batch  149/269 - Train Accuracy:  0.755, Validation Accuracy:  0.766, Loss:  0.301
    Epoch  11 Batch  150/269 - Train Accuracy:  0.758, Validation Accuracy:  0.768, Loss:  0.292
    Epoch  11 Batch  151/269 - Train Accuracy:  0.784, Validation Accuracy:  0.767, Loss:  0.284
    Epoch  11 Batch  152/269 - Train Accuracy:  0.762, Validation Accuracy:  0.768, Loss:  0.291
    Epoch  11 Batch  153/269 - Train Accuracy:  0.765, Validation Accuracy:  0.768, Loss:  0.285
    Epoch  11 Batch  154/269 - Train Accuracy:  0.748, Validation Accuracy:  0.767, Loss:  0.295
    Epoch  11 Batch  155/269 - Train Accuracy:  0.774, Validation Accuracy:  0.760, Loss:  0.282
    Epoch  11 Batch  156/269 - Train Accuracy:  0.758, Validation Accuracy:  0.762, Loss:  0.307
    Epoch  11 Batch  157/269 - Train Accuracy:  0.758, Validation Accuracy:  0.764, Loss:  0.288
    Epoch  11 Batch  158/269 - Train Accuracy:  0.760, Validation Accuracy:  0.766, Loss:  0.292
    Epoch  11 Batch  159/269 - Train Accuracy:  0.759, Validation Accuracy:  0.766, Loss:  0.286
    Epoch  11 Batch  160/269 - Train Accuracy:  0.769, Validation Accuracy:  0.766, Loss:  0.284
    Epoch  11 Batch  161/269 - Train Accuracy:  0.759, Validation Accuracy:  0.769, Loss:  0.289
    Epoch  11 Batch  162/269 - Train Accuracy:  0.767, Validation Accuracy:  0.767, Loss:  0.294
    Epoch  11 Batch  163/269 - Train Accuracy:  0.772, Validation Accuracy:  0.771, Loss:  0.298
    Epoch  11 Batch  164/269 - Train Accuracy:  0.766, Validation Accuracy:  0.770, Loss:  0.286
    Epoch  11 Batch  165/269 - Train Accuracy:  0.738, Validation Accuracy:  0.769, Loss:  0.302
    Epoch  11 Batch  166/269 - Train Accuracy:  0.769, Validation Accuracy:  0.768, Loss:  0.276
    Epoch  11 Batch  167/269 - Train Accuracy:  0.766, Validation Accuracy:  0.767, Loss:  0.285
    Epoch  11 Batch  168/269 - Train Accuracy:  0.757, Validation Accuracy:  0.769, Loss:  0.297
    Epoch  11 Batch  169/269 - Train Accuracy:  0.754, Validation Accuracy:  0.772, Loss:  0.296
    Epoch  11 Batch  170/269 - Train Accuracy:  0.762, Validation Accuracy:  0.767, Loss:  0.283
    Epoch  11 Batch  171/269 - Train Accuracy:  0.760, Validation Accuracy:  0.773, Loss:  0.304
    Epoch  11 Batch  172/269 - Train Accuracy:  0.743, Validation Accuracy:  0.771, Loss:  0.304
    Epoch  11 Batch  173/269 - Train Accuracy:  0.761, Validation Accuracy:  0.767, Loss:  0.287
    Epoch  11 Batch  174/269 - Train Accuracy:  0.758, Validation Accuracy:  0.761, Loss:  0.292
    Epoch  11 Batch  175/269 - Train Accuracy:  0.767, Validation Accuracy:  0.767, Loss:  0.310
    Epoch  11 Batch  176/269 - Train Accuracy:  0.750, Validation Accuracy:  0.767, Loss:  0.308
    Epoch  11 Batch  177/269 - Train Accuracy:  0.781, Validation Accuracy:  0.769, Loss:  0.279
    Epoch  11 Batch  178/269 - Train Accuracy:  0.760, Validation Accuracy:  0.767, Loss:  0.293
    Epoch  11 Batch  179/269 - Train Accuracy:  0.759, Validation Accuracy:  0.771, Loss:  0.290
    Epoch  11 Batch  180/269 - Train Accuracy:  0.765, Validation Accuracy:  0.768, Loss:  0.286
    Epoch  11 Batch  181/269 - Train Accuracy:  0.745, Validation Accuracy:  0.768, Loss:  0.292
    Epoch  11 Batch  182/269 - Train Accuracy:  0.773, Validation Accuracy:  0.768, Loss:  0.288
    Epoch  11 Batch  183/269 - Train Accuracy:  0.799, Validation Accuracy:  0.772, Loss:  0.255
    Epoch  11 Batch  184/269 - Train Accuracy:  0.757, Validation Accuracy:  0.770, Loss:  0.297
    Epoch  11 Batch  185/269 - Train Accuracy:  0.771, Validation Accuracy:  0.772, Loss:  0.285
    Epoch  11 Batch  186/269 - Train Accuracy:  0.751, Validation Accuracy:  0.768, Loss:  0.290
    Epoch  11 Batch  187/269 - Train Accuracy:  0.763, Validation Accuracy:  0.765, Loss:  0.279
    Epoch  11 Batch  188/269 - Train Accuracy:  0.785, Validation Accuracy:  0.768, Loss:  0.277
    Epoch  11 Batch  189/269 - Train Accuracy:  0.768, Validation Accuracy:  0.768, Loss:  0.276
    Epoch  11 Batch  190/269 - Train Accuracy:  0.770, Validation Accuracy:  0.768, Loss:  0.284
    Epoch  11 Batch  191/269 - Train Accuracy:  0.773, Validation Accuracy:  0.771, Loss:  0.282
    Epoch  11 Batch  192/269 - Train Accuracy:  0.773, Validation Accuracy:  0.772, Loss:  0.287
    Epoch  11 Batch  193/269 - Train Accuracy:  0.769, Validation Accuracy:  0.771, Loss:  0.279
    Epoch  11 Batch  194/269 - Train Accuracy:  0.780, Validation Accuracy:  0.769, Loss:  0.289
    Epoch  11 Batch  195/269 - Train Accuracy:  0.765, Validation Accuracy:  0.771, Loss:  0.287
    Epoch  11 Batch  196/269 - Train Accuracy:  0.752, Validation Accuracy:  0.769, Loss:  0.287
    Epoch  11 Batch  197/269 - Train Accuracy:  0.732, Validation Accuracy:  0.767, Loss:  0.306
    Epoch  11 Batch  198/269 - Train Accuracy:  0.743, Validation Accuracy:  0.766, Loss:  0.296
    Epoch  11 Batch  199/269 - Train Accuracy:  0.752, Validation Accuracy:  0.765, Loss:  0.290
    Epoch  11 Batch  200/269 - Train Accuracy:  0.777, Validation Accuracy:  0.764, Loss:  0.294
    Epoch  11 Batch  201/269 - Train Accuracy:  0.756, Validation Accuracy:  0.767, Loss:  0.288
    Epoch  11 Batch  202/269 - Train Accuracy:  0.758, Validation Accuracy:  0.769, Loss:  0.289
    Epoch  11 Batch  203/269 - Train Accuracy:  0.759, Validation Accuracy:  0.771, Loss:  0.308
    Epoch  11 Batch  204/269 - Train Accuracy:  0.745, Validation Accuracy:  0.770, Loss:  0.305
    Epoch  11 Batch  205/269 - Train Accuracy:  0.746, Validation Accuracy:  0.767, Loss:  0.283
    Epoch  11 Batch  206/269 - Train Accuracy:  0.765, Validation Accuracy:  0.768, Loss:  0.298
    Epoch  11 Batch  207/269 - Train Accuracy:  0.776, Validation Accuracy:  0.770, Loss:  0.280
    Epoch  11 Batch  208/269 - Train Accuracy:  0.756, Validation Accuracy:  0.767, Loss:  0.296
    Epoch  11 Batch  209/269 - Train Accuracy:  0.773, Validation Accuracy:  0.770, Loss:  0.284
    Epoch  11 Batch  210/269 - Train Accuracy:  0.766, Validation Accuracy:  0.767, Loss:  0.283
    Epoch  11 Batch  211/269 - Train Accuracy:  0.764, Validation Accuracy:  0.769, Loss:  0.289
    Epoch  11 Batch  212/269 - Train Accuracy:  0.770, Validation Accuracy:  0.772, Loss:  0.290
    Epoch  11 Batch  213/269 - Train Accuracy:  0.760, Validation Accuracy:  0.771, Loss:  0.293
    Epoch  11 Batch  214/269 - Train Accuracy:  0.775, Validation Accuracy:  0.770, Loss:  0.288
    Epoch  11 Batch  215/269 - Train Accuracy:  0.787, Validation Accuracy:  0.773, Loss:  0.269
    Epoch  11 Batch  216/269 - Train Accuracy:  0.748, Validation Accuracy:  0.776, Loss:  0.313
    Epoch  11 Batch  217/269 - Train Accuracy:  0.744, Validation Accuracy:  0.773, Loss:  0.299
    Epoch  11 Batch  218/269 - Train Accuracy:  0.758, Validation Accuracy:  0.772, Loss:  0.288
    Epoch  11 Batch  219/269 - Train Accuracy:  0.775, Validation Accuracy:  0.773, Loss:  0.296
    Epoch  11 Batch  220/269 - Train Accuracy:  0.752, Validation Accuracy:  0.770, Loss:  0.271
    Epoch  11 Batch  221/269 - Train Accuracy:  0.781, Validation Accuracy:  0.767, Loss:  0.284
    Epoch  11 Batch  222/269 - Train Accuracy:  0.775, Validation Accuracy:  0.768, Loss:  0.275
    Epoch  11 Batch  223/269 - Train Accuracy:  0.770, Validation Accuracy:  0.768, Loss:  0.278
    Epoch  11 Batch  224/269 - Train Accuracy:  0.773, Validation Accuracy:  0.769, Loss:  0.288
    Epoch  11 Batch  225/269 - Train Accuracy:  0.766, Validation Accuracy:  0.769, Loss:  0.283
    Epoch  11 Batch  226/269 - Train Accuracy:  0.755, Validation Accuracy:  0.765, Loss:  0.283
    Epoch  11 Batch  227/269 - Train Accuracy:  0.801, Validation Accuracy:  0.770, Loss:  0.259
    Epoch  11 Batch  228/269 - Train Accuracy:  0.752, Validation Accuracy:  0.768, Loss:  0.276
    Epoch  11 Batch  229/269 - Train Accuracy:  0.763, Validation Accuracy:  0.771, Loss:  0.272
    Epoch  11 Batch  230/269 - Train Accuracy:  0.750, Validation Accuracy:  0.768, Loss:  0.285
    Epoch  11 Batch  231/269 - Train Accuracy:  0.736, Validation Accuracy:  0.774, Loss:  0.303
    Epoch  11 Batch  232/269 - Train Accuracy:  0.746, Validation Accuracy:  0.771, Loss:  0.303
    Epoch  11 Batch  233/269 - Train Accuracy:  0.772, Validation Accuracy:  0.769, Loss:  0.292
    Epoch  11 Batch  234/269 - Train Accuracy:  0.763, Validation Accuracy:  0.768, Loss:  0.283
    Epoch  11 Batch  235/269 - Train Accuracy:  0.785, Validation Accuracy:  0.771, Loss:  0.266
    Epoch  11 Batch  236/269 - Train Accuracy:  0.749, Validation Accuracy:  0.772, Loss:  0.281
    Epoch  11 Batch  237/269 - Train Accuracy:  0.751, Validation Accuracy:  0.769, Loss:  0.286
    Epoch  11 Batch  238/269 - Train Accuracy:  0.772, Validation Accuracy:  0.769, Loss:  0.279
    Epoch  11 Batch  239/269 - Train Accuracy:  0.772, Validation Accuracy:  0.769, Loss:  0.281
    Epoch  11 Batch  240/269 - Train Accuracy:  0.774, Validation Accuracy:  0.767, Loss:  0.259
    Epoch  11 Batch  241/269 - Train Accuracy:  0.755, Validation Accuracy:  0.774, Loss:  0.288
    Epoch  11 Batch  242/269 - Train Accuracy:  0.741, Validation Accuracy:  0.771, Loss:  0.279
    Epoch  11 Batch  243/269 - Train Accuracy:  0.784, Validation Accuracy:  0.769, Loss:  0.269
    Epoch  11 Batch  244/269 - Train Accuracy:  0.757, Validation Accuracy:  0.764, Loss:  0.290
    Epoch  11 Batch  245/269 - Train Accuracy:  0.754, Validation Accuracy:  0.769, Loss:  0.298
    Epoch  11 Batch  246/269 - Train Accuracy:  0.747, Validation Accuracy:  0.773, Loss:  0.285
    Epoch  11 Batch  247/269 - Train Accuracy:  0.770, Validation Accuracy:  0.776, Loss:  0.289
    Epoch  11 Batch  248/269 - Train Accuracy:  0.767, Validation Accuracy:  0.777, Loss:  0.282
    Epoch  11 Batch  249/269 - Train Accuracy:  0.787, Validation Accuracy:  0.777, Loss:  0.268
    Epoch  11 Batch  250/269 - Train Accuracy:  0.761, Validation Accuracy:  0.780, Loss:  0.286
    Epoch  11 Batch  251/269 - Train Accuracy:  0.778, Validation Accuracy:  0.774, Loss:  0.277
    Epoch  11 Batch  252/269 - Train Accuracy:  0.759, Validation Accuracy:  0.775, Loss:  0.290
    Epoch  11 Batch  253/269 - Train Accuracy:  0.759, Validation Accuracy:  0.776, Loss:  0.295
    Epoch  11 Batch  254/269 - Train Accuracy:  0.764, Validation Accuracy:  0.777, Loss:  0.287
    Epoch  11 Batch  255/269 - Train Accuracy:  0.788, Validation Accuracy:  0.777, Loss:  0.270
    Epoch  11 Batch  256/269 - Train Accuracy:  0.742, Validation Accuracy:  0.775, Loss:  0.289
    Epoch  11 Batch  257/269 - Train Accuracy:  0.738, Validation Accuracy:  0.771, Loss:  0.300
    Epoch  11 Batch  258/269 - Train Accuracy:  0.748, Validation Accuracy:  0.769, Loss:  0.290
    Epoch  11 Batch  259/269 - Train Accuracy:  0.784, Validation Accuracy:  0.768, Loss:  0.281
    Epoch  11 Batch  260/269 - Train Accuracy:  0.756, Validation Accuracy:  0.772, Loss:  0.306
    Epoch  11 Batch  261/269 - Train Accuracy:  0.745, Validation Accuracy:  0.771, Loss:  0.299
    Epoch  11 Batch  262/269 - Train Accuracy:  0.774, Validation Accuracy:  0.772, Loss:  0.279
    Epoch  11 Batch  263/269 - Train Accuracy:  0.763, Validation Accuracy:  0.773, Loss:  0.290
    Epoch  11 Batch  264/269 - Train Accuracy:  0.756, Validation Accuracy:  0.770, Loss:  0.295
    Epoch  11 Batch  265/269 - Train Accuracy:  0.757, Validation Accuracy:  0.769, Loss:  0.288
    Epoch  11 Batch  266/269 - Train Accuracy:  0.766, Validation Accuracy:  0.769, Loss:  0.276
    Epoch  11 Batch  267/269 - Train Accuracy:  0.773, Validation Accuracy:  0.763, Loss:  0.288
    Epoch  12 Batch    0/269 - Train Accuracy:  0.757, Validation Accuracy:  0.765, Loss:  0.299
    Epoch  12 Batch    1/269 - Train Accuracy:  0.755, Validation Accuracy:  0.764, Loss:  0.284
    Epoch  12 Batch    2/269 - Train Accuracy:  0.755, Validation Accuracy:  0.763, Loss:  0.290
    Epoch  12 Batch    3/269 - Train Accuracy:  0.764, Validation Accuracy:  0.764, Loss:  0.285
    Epoch  12 Batch    4/269 - Train Accuracy:  0.741, Validation Accuracy:  0.762, Loss:  0.295
    Epoch  12 Batch    5/269 - Train Accuracy:  0.735, Validation Accuracy:  0.766, Loss:  0.297
    Epoch  12 Batch    6/269 - Train Accuracy:  0.760, Validation Accuracy:  0.764, Loss:  0.278
    Epoch  12 Batch    7/269 - Train Accuracy:  0.774, Validation Accuracy:  0.768, Loss:  0.277
    Epoch  12 Batch    8/269 - Train Accuracy:  0.746, Validation Accuracy:  0.772, Loss:  0.295
    Epoch  12 Batch    9/269 - Train Accuracy:  0.743, Validation Accuracy:  0.770, Loss:  0.291
    Epoch  12 Batch   10/269 - Train Accuracy:  0.762, Validation Accuracy:  0.773, Loss:  0.286
    Epoch  12 Batch   11/269 - Train Accuracy:  0.758, Validation Accuracy:  0.774, Loss:  0.291
    Epoch  12 Batch   12/269 - Train Accuracy:  0.743, Validation Accuracy:  0.775, Loss:  0.300
    Epoch  12 Batch   13/269 - Train Accuracy:  0.781, Validation Accuracy:  0.778, Loss:  0.261
    Epoch  12 Batch   14/269 - Train Accuracy:  0.746, Validation Accuracy:  0.774, Loss:  0.286
    Epoch  12 Batch   15/269 - Train Accuracy:  0.745, Validation Accuracy:  0.771, Loss:  0.277
    Epoch  12 Batch   16/269 - Train Accuracy:  0.775, Validation Accuracy:  0.767, Loss:  0.298
    Epoch  12 Batch   17/269 - Train Accuracy:  0.661, Validation Accuracy:  0.676, Loss:  0.281
    Epoch  12 Batch   18/269 - Train Accuracy:  0.592, Validation Accuracy:  0.641, Loss:  0.786
    Epoch  12 Batch   19/269 - Train Accuracy:  0.591, Validation Accuracy:  0.575, Loss:  0.491
    Epoch  12 Batch   20/269 - Train Accuracy:  0.570, Validation Accuracy:  0.618, Loss:  1.307
    Epoch  12 Batch   21/269 - Train Accuracy:  0.592, Validation Accuracy:  0.630, Loss:  0.848
    Epoch  12 Batch   22/269 - Train Accuracy:  0.563, Validation Accuracy:  0.582, Loss:  0.558
    Epoch  12 Batch   23/269 - Train Accuracy:  0.604, Validation Accuracy:  0.613, Loss:  0.883
    Epoch  12 Batch   24/269 - Train Accuracy:  0.532, Validation Accuracy:  0.571, Loss:  0.878
    Epoch  12 Batch   25/269 - Train Accuracy:  0.557, Validation Accuracy:  0.593, Loss:  0.993
    Epoch  12 Batch   26/269 - Train Accuracy:  0.617, Validation Accuracy:  0.598, Loss:  0.809
    Epoch  12 Batch   27/269 - Train Accuracy:  0.618, Validation Accuracy:  0.635, Loss:  0.888
    Epoch  12 Batch   28/269 - Train Accuracy:  0.538, Validation Accuracy:  0.596, Loss:  0.741
    Epoch  12 Batch   29/269 - Train Accuracy:  0.571, Validation Accuracy:  0.599, Loss:  0.718
    Epoch  12 Batch   30/269 - Train Accuracy:  0.627, Validation Accuracy:  0.648, Loss:  0.755
    Epoch  12 Batch   31/269 - Train Accuracy:  0.656, Validation Accuracy:  0.656, Loss:  0.626
    Epoch  12 Batch   32/269 - Train Accuracy:  0.631, Validation Accuracy:  0.640, Loss:  0.577
    Epoch  12 Batch   33/269 - Train Accuracy:  0.659, Validation Accuracy:  0.664, Loss:  0.572
    Epoch  12 Batch   34/269 - Train Accuracy:  0.651, Validation Accuracy:  0.670, Loss:  0.553
    Epoch  12 Batch   35/269 - Train Accuracy:  0.669, Validation Accuracy:  0.670, Loss:  0.524
    Epoch  12 Batch   36/269 - Train Accuracy:  0.649, Validation Accuracy:  0.668, Loss:  0.498
    Epoch  12 Batch   37/269 - Train Accuracy:  0.668, Validation Accuracy:  0.671, Loss:  0.519
    Epoch  12 Batch   38/269 - Train Accuracy:  0.683, Validation Accuracy:  0.688, Loss:  0.495
    Epoch  12 Batch   39/269 - Train Accuracy:  0.696, Validation Accuracy:  0.699, Loss:  0.463
    Epoch  12 Batch   40/269 - Train Accuracy:  0.683, Validation Accuracy:  0.692, Loss:  0.463
    Epoch  12 Batch   41/269 - Train Accuracy:  0.677, Validation Accuracy:  0.698, Loss:  0.452
    Epoch  12 Batch   42/269 - Train Accuracy:  0.705, Validation Accuracy:  0.708, Loss:  0.410
    Epoch  12 Batch   43/269 - Train Accuracy:  0.688, Validation Accuracy:  0.720, Loss:  0.432
    Epoch  12 Batch   44/269 - Train Accuracy:  0.718, Validation Accuracy:  0.717, Loss:  0.412
    Epoch  12 Batch   45/269 - Train Accuracy:  0.708, Validation Accuracy:  0.717, Loss:  0.418
    Epoch  12 Batch   46/269 - Train Accuracy:  0.718, Validation Accuracy:  0.723, Loss:  0.413
    Epoch  12 Batch   47/269 - Train Accuracy:  0.733, Validation Accuracy:  0.727, Loss:  0.360
    Epoch  12 Batch   48/269 - Train Accuracy:  0.726, Validation Accuracy:  0.727, Loss:  0.366
    Epoch  12 Batch   49/269 - Train Accuracy:  0.704, Validation Accuracy:  0.722, Loss:  0.380
    Epoch  12 Batch   50/269 - Train Accuracy:  0.711, Validation Accuracy:  0.732, Loss:  0.393
    Epoch  12 Batch   51/269 - Train Accuracy:  0.718, Validation Accuracy:  0.738, Loss:  0.368
    Epoch  12 Batch   52/269 - Train Accuracy:  0.722, Validation Accuracy:  0.736, Loss:  0.341
    Epoch  12 Batch   53/269 - Train Accuracy:  0.720, Validation Accuracy:  0.736, Loss:  0.381
    Epoch  12 Batch   54/269 - Train Accuracy:  0.741, Validation Accuracy:  0.741, Loss:  0.356
    Epoch  12 Batch   55/269 - Train Accuracy:  0.745, Validation Accuracy:  0.743, Loss:  0.344
    Epoch  12 Batch   56/269 - Train Accuracy:  0.741, Validation Accuracy:  0.740, Loss:  0.342
    Epoch  12 Batch   57/269 - Train Accuracy:  0.728, Validation Accuracy:  0.742, Loss:  0.355
    Epoch  12 Batch   58/269 - Train Accuracy:  0.745, Validation Accuracy:  0.740, Loss:  0.341
    Epoch  12 Batch   59/269 - Train Accuracy:  0.756, Validation Accuracy:  0.740, Loss:  0.320
    Epoch  12 Batch   60/269 - Train Accuracy:  0.739, Validation Accuracy:  0.745, Loss:  0.321
    Epoch  12 Batch   61/269 - Train Accuracy:  0.754, Validation Accuracy:  0.747, Loss:  0.306
    Epoch  12 Batch   62/269 - Train Accuracy:  0.746, Validation Accuracy:  0.748, Loss:  0.318
    Epoch  12 Batch   63/269 - Train Accuracy:  0.739, Validation Accuracy:  0.750, Loss:  0.340
    Epoch  12 Batch   64/269 - Train Accuracy:  0.737, Validation Accuracy:  0.752, Loss:  0.321
    Epoch  12 Batch   65/269 - Train Accuracy:  0.730, Validation Accuracy:  0.753, Loss:  0.318
    Epoch  12 Batch   66/269 - Train Accuracy:  0.741, Validation Accuracy:  0.758, Loss:  0.316
    Epoch  12 Batch   67/269 - Train Accuracy:  0.741, Validation Accuracy:  0.759, Loss:  0.328
    Epoch  12 Batch   68/269 - Train Accuracy:  0.732, Validation Accuracy:  0.757, Loss:  0.326
    Epoch  12 Batch   69/269 - Train Accuracy:  0.726, Validation Accuracy:  0.761, Loss:  0.362
    Epoch  12 Batch   70/269 - Train Accuracy:  0.775, Validation Accuracy:  0.762, Loss:  0.314
    Epoch  12 Batch   71/269 - Train Accuracy:  0.753, Validation Accuracy:  0.760, Loss:  0.329
    Epoch  12 Batch   72/269 - Train Accuracy:  0.745, Validation Accuracy:  0.760, Loss:  0.313
    Epoch  12 Batch   73/269 - Train Accuracy:  0.746, Validation Accuracy:  0.765, Loss:  0.318
    Epoch  12 Batch   74/269 - Train Accuracy:  0.746, Validation Accuracy:  0.764, Loss:  0.311
    Epoch  12 Batch   75/269 - Train Accuracy:  0.739, Validation Accuracy:  0.765, Loss:  0.310
    Epoch  12 Batch   76/269 - Train Accuracy:  0.742, Validation Accuracy:  0.763, Loss:  0.314
    Epoch  12 Batch   77/269 - Train Accuracy:  0.768, Validation Accuracy:  0.765, Loss:  0.302
    Epoch  12 Batch   78/269 - Train Accuracy:  0.766, Validation Accuracy:  0.767, Loss:  0.295
    Epoch  12 Batch   79/269 - Train Accuracy:  0.757, Validation Accuracy:  0.766, Loss:  0.306
    Epoch  12 Batch   80/269 - Train Accuracy:  0.768, Validation Accuracy:  0.765, Loss:  0.297
    Epoch  12 Batch   81/269 - Train Accuracy:  0.753, Validation Accuracy:  0.763, Loss:  0.314
    Epoch  12 Batch   82/269 - Train Accuracy:  0.771, Validation Accuracy:  0.762, Loss:  0.285
    Epoch  12 Batch   83/269 - Train Accuracy:  0.750, Validation Accuracy:  0.762, Loss:  0.317
    Epoch  12 Batch   84/269 - Train Accuracy:  0.758, Validation Accuracy:  0.764, Loss:  0.297
    Epoch  12 Batch   85/269 - Train Accuracy:  0.758, Validation Accuracy:  0.764, Loss:  0.301
    Epoch  12 Batch   86/269 - Train Accuracy:  0.744, Validation Accuracy:  0.766, Loss:  0.294
    Epoch  12 Batch   87/269 - Train Accuracy:  0.741, Validation Accuracy:  0.767, Loss:  0.316
    Epoch  12 Batch   88/269 - Train Accuracy:  0.754, Validation Accuracy:  0.769, Loss:  0.297
    Epoch  12 Batch   89/269 - Train Accuracy:  0.770, Validation Accuracy:  0.768, Loss:  0.297
    Epoch  12 Batch   90/269 - Train Accuracy:  0.723, Validation Accuracy:  0.768, Loss:  0.314
    Epoch  12 Batch   91/269 - Train Accuracy:  0.754, Validation Accuracy:  0.765, Loss:  0.289
    Epoch  12 Batch   92/269 - Train Accuracy:  0.748, Validation Accuracy:  0.764, Loss:  0.289
    Epoch  12 Batch   93/269 - Train Accuracy:  0.766, Validation Accuracy:  0.763, Loss:  0.283
    Epoch  12 Batch   94/269 - Train Accuracy:  0.752, Validation Accuracy:  0.762, Loss:  0.307
    Epoch  12 Batch   95/269 - Train Accuracy:  0.749, Validation Accuracy:  0.764, Loss:  0.292
    Epoch  12 Batch   96/269 - Train Accuracy:  0.753, Validation Accuracy:  0.764, Loss:  0.299
    Epoch  12 Batch   97/269 - Train Accuracy:  0.761, Validation Accuracy:  0.763, Loss:  0.292
    Epoch  12 Batch   98/269 - Train Accuracy:  0.760, Validation Accuracy:  0.766, Loss:  0.296
    Epoch  12 Batch   99/269 - Train Accuracy:  0.769, Validation Accuracy:  0.767, Loss:  0.294
    Epoch  12 Batch  100/269 - Train Accuracy:  0.789, Validation Accuracy:  0.768, Loss:  0.283
    Epoch  12 Batch  101/269 - Train Accuracy:  0.726, Validation Accuracy:  0.765, Loss:  0.315
    Epoch  12 Batch  102/269 - Train Accuracy:  0.757, Validation Accuracy:  0.766, Loss:  0.287
    Epoch  12 Batch  103/269 - Train Accuracy:  0.757, Validation Accuracy:  0.768, Loss:  0.288
    Epoch  12 Batch  104/269 - Train Accuracy:  0.756, Validation Accuracy:  0.771, Loss:  0.291
    Epoch  12 Batch  105/269 - Train Accuracy:  0.747, Validation Accuracy:  0.772, Loss:  0.297
    Epoch  12 Batch  106/269 - Train Accuracy:  0.749, Validation Accuracy:  0.773, Loss:  0.290
    Epoch  12 Batch  107/269 - Train Accuracy:  0.737, Validation Accuracy:  0.772, Loss:  0.307
    Epoch  12 Batch  108/269 - Train Accuracy:  0.754, Validation Accuracy:  0.772, Loss:  0.291
    Epoch  12 Batch  109/269 - Train Accuracy:  0.732, Validation Accuracy:  0.774, Loss:  0.296
    Epoch  12 Batch  110/269 - Train Accuracy:  0.750, Validation Accuracy:  0.772, Loss:  0.283
    Epoch  12 Batch  111/269 - Train Accuracy:  0.748, Validation Accuracy:  0.771, Loss:  0.303
    Epoch  12 Batch  112/269 - Train Accuracy:  0.760, Validation Accuracy:  0.769, Loss:  0.293
    Epoch  12 Batch  113/269 - Train Accuracy:  0.759, Validation Accuracy:  0.767, Loss:  0.277
    Epoch  12 Batch  114/269 - Train Accuracy:  0.745, Validation Accuracy:  0.764, Loss:  0.288
    Epoch  12 Batch  115/269 - Train Accuracy:  0.734, Validation Accuracy:  0.766, Loss:  0.310
    Epoch  12 Batch  116/269 - Train Accuracy:  0.762, Validation Accuracy:  0.767, Loss:  0.289
    Epoch  12 Batch  117/269 - Train Accuracy:  0.761, Validation Accuracy:  0.769, Loss:  0.287
    Epoch  12 Batch  118/269 - Train Accuracy:  0.793, Validation Accuracy:  0.770, Loss:  0.277
    Epoch  12 Batch  119/269 - Train Accuracy:  0.757, Validation Accuracy:  0.771, Loss:  0.293
    Epoch  12 Batch  120/269 - Train Accuracy:  0.759, Validation Accuracy:  0.770, Loss:  0.297
    Epoch  12 Batch  121/269 - Train Accuracy:  0.760, Validation Accuracy:  0.770, Loss:  0.285
    Epoch  12 Batch  122/269 - Train Accuracy:  0.770, Validation Accuracy:  0.768, Loss:  0.281
    Epoch  12 Batch  123/269 - Train Accuracy:  0.749, Validation Accuracy:  0.766, Loss:  0.292
    Epoch  12 Batch  124/269 - Train Accuracy:  0.755, Validation Accuracy:  0.763, Loss:  0.277
    Epoch  12 Batch  125/269 - Train Accuracy:  0.753, Validation Accuracy:  0.765, Loss:  0.283
    Epoch  12 Batch  126/269 - Train Accuracy:  0.752, Validation Accuracy:  0.764, Loss:  0.291
    Epoch  12 Batch  127/269 - Train Accuracy:  0.743, Validation Accuracy:  0.767, Loss:  0.297
    Epoch  12 Batch  128/269 - Train Accuracy:  0.776, Validation Accuracy:  0.765, Loss:  0.287
    Epoch  12 Batch  129/269 - Train Accuracy:  0.766, Validation Accuracy:  0.766, Loss:  0.280
    Epoch  12 Batch  130/269 - Train Accuracy:  0.749, Validation Accuracy:  0.767, Loss:  0.294
    Epoch  12 Batch  131/269 - Train Accuracy:  0.749, Validation Accuracy:  0.766, Loss:  0.291
    Epoch  12 Batch  132/269 - Train Accuracy:  0.751, Validation Accuracy:  0.765, Loss:  0.294
    Epoch  12 Batch  133/269 - Train Accuracy:  0.767, Validation Accuracy:  0.765, Loss:  0.276
    Epoch  12 Batch  134/269 - Train Accuracy:  0.745, Validation Accuracy:  0.763, Loss:  0.294
    Epoch  12 Batch  135/269 - Train Accuracy:  0.748, Validation Accuracy:  0.762, Loss:  0.303
    Epoch  12 Batch  136/269 - Train Accuracy:  0.737, Validation Accuracy:  0.763, Loss:  0.302
    Epoch  12 Batch  137/269 - Train Accuracy:  0.751, Validation Accuracy:  0.763, Loss:  0.302
    Epoch  12 Batch  138/269 - Train Accuracy:  0.758, Validation Accuracy:  0.762, Loss:  0.296
    Epoch  12 Batch  139/269 - Train Accuracy:  0.773, Validation Accuracy:  0.766, Loss:  0.275
    Epoch  12 Batch  140/269 - Train Accuracy:  0.762, Validation Accuracy:  0.772, Loss:  0.298
    Epoch  12 Batch  141/269 - Train Accuracy:  0.762, Validation Accuracy:  0.768, Loss:  0.293
    Epoch  12 Batch  142/269 - Train Accuracy:  0.769, Validation Accuracy:  0.768, Loss:  0.270
    Epoch  12 Batch  143/269 - Train Accuracy:  0.774, Validation Accuracy:  0.769, Loss:  0.280
    Epoch  12 Batch  144/269 - Train Accuracy:  0.772, Validation Accuracy:  0.769, Loss:  0.262
    Epoch  12 Batch  145/269 - Train Accuracy:  0.761, Validation Accuracy:  0.768, Loss:  0.277
    Epoch  12 Batch  146/269 - Train Accuracy:  0.765, Validation Accuracy:  0.770, Loss:  0.271
    Epoch  12 Batch  147/269 - Train Accuracy:  0.771, Validation Accuracy:  0.767, Loss:  0.275
    Epoch  12 Batch  148/269 - Train Accuracy:  0.768, Validation Accuracy:  0.768, Loss:  0.280
    Epoch  12 Batch  149/269 - Train Accuracy:  0.757, Validation Accuracy:  0.770, Loss:  0.289
    Epoch  12 Batch  150/269 - Train Accuracy:  0.762, Validation Accuracy:  0.768, Loss:  0.280
    Epoch  12 Batch  151/269 - Train Accuracy:  0.786, Validation Accuracy:  0.769, Loss:  0.276
    Epoch  12 Batch  152/269 - Train Accuracy:  0.769, Validation Accuracy:  0.768, Loss:  0.283
    Epoch  12 Batch  153/269 - Train Accuracy:  0.768, Validation Accuracy:  0.769, Loss:  0.280
    Epoch  12 Batch  154/269 - Train Accuracy:  0.754, Validation Accuracy:  0.768, Loss:  0.287
    Epoch  12 Batch  155/269 - Train Accuracy:  0.778, Validation Accuracy:  0.767, Loss:  0.268
    Epoch  12 Batch  156/269 - Train Accuracy:  0.755, Validation Accuracy:  0.766, Loss:  0.292
    Epoch  12 Batch  157/269 - Train Accuracy:  0.765, Validation Accuracy:  0.767, Loss:  0.272
    Epoch  12 Batch  158/269 - Train Accuracy:  0.760, Validation Accuracy:  0.767, Loss:  0.279
    Epoch  12 Batch  159/269 - Train Accuracy:  0.761, Validation Accuracy:  0.767, Loss:  0.277
    Epoch  12 Batch  160/269 - Train Accuracy:  0.773, Validation Accuracy:  0.767, Loss:  0.270
    Epoch  12 Batch  161/269 - Train Accuracy:  0.771, Validation Accuracy:  0.768, Loss:  0.279
    Epoch  12 Batch  162/269 - Train Accuracy:  0.773, Validation Accuracy:  0.770, Loss:  0.282
    Epoch  12 Batch  163/269 - Train Accuracy:  0.769, Validation Accuracy:  0.767, Loss:  0.284
    Epoch  12 Batch  164/269 - Train Accuracy:  0.769, Validation Accuracy:  0.771, Loss:  0.273
    Epoch  12 Batch  165/269 - Train Accuracy:  0.743, Validation Accuracy:  0.771, Loss:  0.284
    Epoch  12 Batch  166/269 - Train Accuracy:  0.765, Validation Accuracy:  0.773, Loss:  0.261
    Epoch  12 Batch  167/269 - Train Accuracy:  0.766, Validation Accuracy:  0.773, Loss:  0.275
    Epoch  12 Batch  168/269 - Train Accuracy:  0.762, Validation Accuracy:  0.773, Loss:  0.281
    Epoch  12 Batch  169/269 - Train Accuracy:  0.749, Validation Accuracy:  0.773, Loss:  0.289
    Epoch  12 Batch  170/269 - Train Accuracy:  0.767, Validation Accuracy:  0.772, Loss:  0.269
    Epoch  12 Batch  171/269 - Train Accuracy:  0.765, Validation Accuracy:  0.773, Loss:  0.283
    Epoch  12 Batch  172/269 - Train Accuracy:  0.750, Validation Accuracy:  0.773, Loss:  0.289
    Epoch  12 Batch  173/269 - Train Accuracy:  0.759, Validation Accuracy:  0.771, Loss:  0.269
    Epoch  12 Batch  174/269 - Train Accuracy:  0.763, Validation Accuracy:  0.770, Loss:  0.274
    Epoch  12 Batch  175/269 - Train Accuracy:  0.770, Validation Accuracy:  0.769, Loss:  0.296
    Epoch  12 Batch  176/269 - Train Accuracy:  0.749, Validation Accuracy:  0.770, Loss:  0.294
    Epoch  12 Batch  177/269 - Train Accuracy:  0.776, Validation Accuracy:  0.773, Loss:  0.267
    Epoch  12 Batch  178/269 - Train Accuracy:  0.767, Validation Accuracy:  0.772, Loss:  0.281
    Epoch  12 Batch  179/269 - Train Accuracy:  0.766, Validation Accuracy:  0.772, Loss:  0.278
    Epoch  12 Batch  180/269 - Train Accuracy:  0.766, Validation Accuracy:  0.773, Loss:  0.270
    Epoch  12 Batch  181/269 - Train Accuracy:  0.752, Validation Accuracy:  0.772, Loss:  0.279
    Epoch  12 Batch  182/269 - Train Accuracy:  0.779, Validation Accuracy:  0.772, Loss:  0.272
    Epoch  12 Batch  183/269 - Train Accuracy:  0.802, Validation Accuracy:  0.770, Loss:  0.239
    Epoch  12 Batch  184/269 - Train Accuracy:  0.758, Validation Accuracy:  0.770, Loss:  0.285
    Epoch  12 Batch  185/269 - Train Accuracy:  0.776, Validation Accuracy:  0.772, Loss:  0.268
    Epoch  12 Batch  186/269 - Train Accuracy:  0.758, Validation Accuracy:  0.774, Loss:  0.277
    Epoch  12 Batch  187/269 - Train Accuracy:  0.760, Validation Accuracy:  0.775, Loss:  0.272
    Epoch  12 Batch  188/269 - Train Accuracy:  0.784, Validation Accuracy:  0.772, Loss:  0.262
    Epoch  12 Batch  189/269 - Train Accuracy:  0.773, Validation Accuracy:  0.773, Loss:  0.266
    Epoch  12 Batch  190/269 - Train Accuracy:  0.775, Validation Accuracy:  0.776, Loss:  0.268
    Epoch  12 Batch  191/269 - Train Accuracy:  0.777, Validation Accuracy:  0.774, Loss:  0.270
    Epoch  12 Batch  192/269 - Train Accuracy:  0.778, Validation Accuracy:  0.771, Loss:  0.271
    Epoch  12 Batch  193/269 - Train Accuracy:  0.772, Validation Accuracy:  0.771, Loss:  0.268
    Epoch  12 Batch  194/269 - Train Accuracy:  0.784, Validation Accuracy:  0.773, Loss:  0.275
    Epoch  12 Batch  195/269 - Train Accuracy:  0.770, Validation Accuracy:  0.774, Loss:  0.273
    Epoch  12 Batch  196/269 - Train Accuracy:  0.753, Validation Accuracy:  0.774, Loss:  0.272
    Epoch  12 Batch  197/269 - Train Accuracy:  0.736, Validation Accuracy:  0.772, Loss:  0.289
    Epoch  12 Batch  198/269 - Train Accuracy:  0.753, Validation Accuracy:  0.773, Loss:  0.285
    Epoch  12 Batch  199/269 - Train Accuracy:  0.753, Validation Accuracy:  0.775, Loss:  0.277
    Epoch  12 Batch  200/269 - Train Accuracy:  0.778, Validation Accuracy:  0.774, Loss:  0.278
    Epoch  12 Batch  201/269 - Train Accuracy:  0.757, Validation Accuracy:  0.774, Loss:  0.276
    Epoch  12 Batch  202/269 - Train Accuracy:  0.765, Validation Accuracy:  0.775, Loss:  0.274
    Epoch  12 Batch  203/269 - Train Accuracy:  0.762, Validation Accuracy:  0.776, Loss:  0.286
    Epoch  12 Batch  204/269 - Train Accuracy:  0.747, Validation Accuracy:  0.776, Loss:  0.290
    Epoch  12 Batch  205/269 - Train Accuracy:  0.753, Validation Accuracy:  0.775, Loss:  0.270
    Epoch  12 Batch  206/269 - Train Accuracy:  0.766, Validation Accuracy:  0.776, Loss:  0.281
    Epoch  12 Batch  207/269 - Train Accuracy:  0.776, Validation Accuracy:  0.774, Loss:  0.264
    Epoch  12 Batch  208/269 - Train Accuracy:  0.766, Validation Accuracy:  0.774, Loss:  0.282
    Epoch  12 Batch  209/269 - Train Accuracy:  0.772, Validation Accuracy:  0.775, Loss:  0.273
    Epoch  12 Batch  210/269 - Train Accuracy:  0.770, Validation Accuracy:  0.775, Loss:  0.267
    Epoch  12 Batch  211/269 - Train Accuracy:  0.772, Validation Accuracy:  0.774, Loss:  0.274
    Epoch  12 Batch  212/269 - Train Accuracy:  0.771, Validation Accuracy:  0.776, Loss:  0.279
    Epoch  12 Batch  213/269 - Train Accuracy:  0.758, Validation Accuracy:  0.775, Loss:  0.274
    Epoch  12 Batch  214/269 - Train Accuracy:  0.777, Validation Accuracy:  0.775, Loss:  0.269
    Epoch  12 Batch  215/269 - Train Accuracy:  0.790, Validation Accuracy:  0.774, Loss:  0.258
    Epoch  12 Batch  216/269 - Train Accuracy:  0.744, Validation Accuracy:  0.774, Loss:  0.299
    Epoch  12 Batch  217/269 - Train Accuracy:  0.746, Validation Accuracy:  0.773, Loss:  0.280
    Epoch  12 Batch  218/269 - Train Accuracy:  0.764, Validation Accuracy:  0.773, Loss:  0.274
    Epoch  12 Batch  219/269 - Train Accuracy:  0.774, Validation Accuracy:  0.773, Loss:  0.282
    Epoch  12 Batch  220/269 - Train Accuracy:  0.759, Validation Accuracy:  0.773, Loss:  0.259
    Epoch  12 Batch  221/269 - Train Accuracy:  0.786, Validation Accuracy:  0.772, Loss:  0.271
    Epoch  12 Batch  222/269 - Train Accuracy:  0.775, Validation Accuracy:  0.772, Loss:  0.261
    Epoch  12 Batch  223/269 - Train Accuracy:  0.775, Validation Accuracy:  0.771, Loss:  0.264
    Epoch  12 Batch  224/269 - Train Accuracy:  0.776, Validation Accuracy:  0.769, Loss:  0.276
    Epoch  12 Batch  225/269 - Train Accuracy:  0.770, Validation Accuracy:  0.767, Loss:  0.266
    Epoch  12 Batch  226/269 - Train Accuracy:  0.765, Validation Accuracy:  0.768, Loss:  0.269
    Epoch  12 Batch  227/269 - Train Accuracy:  0.800, Validation Accuracy:  0.767, Loss:  0.245
    Epoch  12 Batch  228/269 - Train Accuracy:  0.756, Validation Accuracy:  0.768, Loss:  0.266
    Epoch  12 Batch  229/269 - Train Accuracy:  0.762, Validation Accuracy:  0.772, Loss:  0.263
    Epoch  12 Batch  230/269 - Train Accuracy:  0.758, Validation Accuracy:  0.774, Loss:  0.271
    Epoch  12 Batch  231/269 - Train Accuracy:  0.741, Validation Accuracy:  0.775, Loss:  0.281
    Epoch  12 Batch  232/269 - Train Accuracy:  0.742, Validation Accuracy:  0.775, Loss:  0.288
    Epoch  12 Batch  233/269 - Train Accuracy:  0.774, Validation Accuracy:  0.774, Loss:  0.275
    Epoch  12 Batch  234/269 - Train Accuracy:  0.764, Validation Accuracy:  0.773, Loss:  0.267
    Epoch  12 Batch  235/269 - Train Accuracy:  0.787, Validation Accuracy:  0.772, Loss:  0.255
    Epoch  12 Batch  236/269 - Train Accuracy:  0.755, Validation Accuracy:  0.775, Loss:  0.263
    Epoch  12 Batch  237/269 - Train Accuracy:  0.761, Validation Accuracy:  0.775, Loss:  0.268
    Epoch  12 Batch  238/269 - Train Accuracy:  0.774, Validation Accuracy:  0.773, Loss:  0.258
    Epoch  12 Batch  239/269 - Train Accuracy:  0.777, Validation Accuracy:  0.773, Loss:  0.262
    Epoch  12 Batch  240/269 - Train Accuracy:  0.779, Validation Accuracy:  0.774, Loss:  0.239
    Epoch  12 Batch  241/269 - Train Accuracy:  0.765, Validation Accuracy:  0.775, Loss:  0.270
    Epoch  12 Batch  242/269 - Train Accuracy:  0.749, Validation Accuracy:  0.777, Loss:  0.265
    Epoch  12 Batch  243/269 - Train Accuracy:  0.789, Validation Accuracy:  0.777, Loss:  0.253
    Epoch  12 Batch  244/269 - Train Accuracy:  0.762, Validation Accuracy:  0.777, Loss:  0.269
    Epoch  12 Batch  245/269 - Train Accuracy:  0.763, Validation Accuracy:  0.776, Loss:  0.277
    Epoch  12 Batch  246/269 - Train Accuracy:  0.750, Validation Accuracy:  0.778, Loss:  0.268
    Epoch  12 Batch  247/269 - Train Accuracy:  0.771, Validation Accuracy:  0.775, Loss:  0.272
    Epoch  12 Batch  248/269 - Train Accuracy:  0.765, Validation Accuracy:  0.773, Loss:  0.257
    Epoch  12 Batch  249/269 - Train Accuracy:  0.788, Validation Accuracy:  0.774, Loss:  0.246
    Epoch  12 Batch  250/269 - Train Accuracy:  0.764, Validation Accuracy:  0.776, Loss:  0.263
    Epoch  12 Batch  251/269 - Train Accuracy:  0.781, Validation Accuracy:  0.777, Loss:  0.258
    Epoch  12 Batch  252/269 - Train Accuracy:  0.765, Validation Accuracy:  0.777, Loss:  0.270
    Epoch  12 Batch  253/269 - Train Accuracy:  0.759, Validation Accuracy:  0.776, Loss:  0.280
    Epoch  12 Batch  254/269 - Train Accuracy:  0.770, Validation Accuracy:  0.779, Loss:  0.265
    Epoch  12 Batch  255/269 - Train Accuracy:  0.786, Validation Accuracy:  0.779, Loss:  0.249
    Epoch  12 Batch  256/269 - Train Accuracy:  0.750, Validation Accuracy:  0.781, Loss:  0.273
    Epoch  12 Batch  257/269 - Train Accuracy:  0.742, Validation Accuracy:  0.779, Loss:  0.277
    Epoch  12 Batch  258/269 - Train Accuracy:  0.760, Validation Accuracy:  0.778, Loss:  0.269
    Epoch  12 Batch  259/269 - Train Accuracy:  0.779, Validation Accuracy:  0.779, Loss:  0.262
    Epoch  12 Batch  260/269 - Train Accuracy:  0.759, Validation Accuracy:  0.777, Loss:  0.281
    Epoch  12 Batch  261/269 - Train Accuracy:  0.745, Validation Accuracy:  0.773, Loss:  0.280
    Epoch  12 Batch  262/269 - Train Accuracy:  0.783, Validation Accuracy:  0.773, Loss:  0.260
    Epoch  12 Batch  263/269 - Train Accuracy:  0.776, Validation Accuracy:  0.774, Loss:  0.277
    Epoch  12 Batch  264/269 - Train Accuracy:  0.761, Validation Accuracy:  0.777, Loss:  0.281
    Epoch  12 Batch  265/269 - Train Accuracy:  0.757, Validation Accuracy:  0.777, Loss:  0.269
    Epoch  12 Batch  266/269 - Train Accuracy:  0.769, Validation Accuracy:  0.776, Loss:  0.260
    Epoch  12 Batch  267/269 - Train Accuracy:  0.771, Validation Accuracy:  0.776, Loss:  0.272
    Epoch  13 Batch    0/269 - Train Accuracy:  0.762, Validation Accuracy:  0.773, Loss:  0.279
    Epoch  13 Batch    1/269 - Train Accuracy:  0.756, Validation Accuracy:  0.774, Loss:  0.267
    Epoch  13 Batch    2/269 - Train Accuracy:  0.762, Validation Accuracy:  0.775, Loss:  0.269
    Epoch  13 Batch    3/269 - Train Accuracy:  0.765, Validation Accuracy:  0.774, Loss:  0.268
    Epoch  13 Batch    4/269 - Train Accuracy:  0.752, Validation Accuracy:  0.768, Loss:  0.282
    Epoch  13 Batch    5/269 - Train Accuracy:  0.733, Validation Accuracy:  0.769, Loss:  0.274
    Epoch  13 Batch    6/269 - Train Accuracy:  0.762, Validation Accuracy:  0.768, Loss:  0.257
    Epoch  13 Batch    7/269 - Train Accuracy:  0.774, Validation Accuracy:  0.767, Loss:  0.252
    Epoch  13 Batch    8/269 - Train Accuracy:  0.747, Validation Accuracy:  0.768, Loss:  0.280
    Epoch  13 Batch    9/269 - Train Accuracy:  0.760, Validation Accuracy:  0.768, Loss:  0.274
    Epoch  13 Batch   10/269 - Train Accuracy:  0.765, Validation Accuracy:  0.771, Loss:  0.268
    Epoch  13 Batch   11/269 - Train Accuracy:  0.769, Validation Accuracy:  0.770, Loss:  0.272
    Epoch  13 Batch   12/269 - Train Accuracy:  0.751, Validation Accuracy:  0.768, Loss:  0.277
    Epoch  13 Batch   13/269 - Train Accuracy:  0.788, Validation Accuracy:  0.773, Loss:  0.241
    Epoch  13 Batch   14/269 - Train Accuracy:  0.754, Validation Accuracy:  0.773, Loss:  0.268
    Epoch  13 Batch   15/269 - Train Accuracy:  0.752, Validation Accuracy:  0.775, Loss:  0.254
    Epoch  13 Batch   16/269 - Train Accuracy:  0.793, Validation Accuracy:  0.777, Loss:  0.263
    Epoch  13 Batch   17/269 - Train Accuracy:  0.761, Validation Accuracy:  0.776, Loss:  0.256
    Epoch  13 Batch   18/269 - Train Accuracy:  0.766, Validation Accuracy:  0.775, Loss:  0.271
    Epoch  13 Batch   19/269 - Train Accuracy:  0.784, Validation Accuracy:  0.774, Loss:  0.239
    Epoch  13 Batch   20/269 - Train Accuracy:  0.769, Validation Accuracy:  0.772, Loss:  0.272
    Epoch  13 Batch   21/269 - Train Accuracy:  0.764, Validation Accuracy:  0.772, Loss:  0.285
    Epoch  13 Batch   22/269 - Train Accuracy:  0.783, Validation Accuracy:  0.774, Loss:  0.248
    Epoch  13 Batch   23/269 - Train Accuracy:  0.773, Validation Accuracy:  0.778, Loss:  0.255
    Epoch  13 Batch   24/269 - Train Accuracy:  0.754, Validation Accuracy:  0.779, Loss:  0.270
    Epoch  13 Batch   25/269 - Train Accuracy:  0.754, Validation Accuracy:  0.781, Loss:  0.282
    Epoch  13 Batch   26/269 - Train Accuracy:  0.780, Validation Accuracy:  0.778, Loss:  0.241
    Epoch  13 Batch   27/269 - Train Accuracy:  0.756, Validation Accuracy:  0.777, Loss:  0.265
    Epoch  13 Batch   28/269 - Train Accuracy:  0.729, Validation Accuracy:  0.774, Loss:  0.291
    Epoch  13 Batch   29/269 - Train Accuracy:  0.767, Validation Accuracy:  0.771, Loss:  0.275
    Epoch  13 Batch   30/269 - Train Accuracy:  0.765, Validation Accuracy:  0.770, Loss:  0.257
    Epoch  13 Batch   31/269 - Train Accuracy:  0.775, Validation Accuracy:  0.775, Loss:  0.246
    Epoch  13 Batch   32/269 - Train Accuracy:  0.758, Validation Accuracy:  0.771, Loss:  0.250
    Epoch  13 Batch   33/269 - Train Accuracy:  0.771, Validation Accuracy:  0.773, Loss:  0.255
    Epoch  13 Batch   34/269 - Train Accuracy:  0.775, Validation Accuracy:  0.776, Loss:  0.259
    Epoch  13 Batch   35/269 - Train Accuracy:  0.771, Validation Accuracy:  0.774, Loss:  0.272
    Epoch  13 Batch   36/269 - Train Accuracy:  0.770, Validation Accuracy:  0.773, Loss:  0.257
    Epoch  13 Batch   37/269 - Train Accuracy:  0.775, Validation Accuracy:  0.774, Loss:  0.254
    Epoch  13 Batch   38/269 - Train Accuracy:  0.773, Validation Accuracy:  0.774, Loss:  0.254
    Epoch  13 Batch   39/269 - Train Accuracy:  0.772, Validation Accuracy:  0.774, Loss:  0.264
    Epoch  13 Batch   40/269 - Train Accuracy:  0.765, Validation Accuracy:  0.775, Loss:  0.271
    Epoch  13 Batch   41/269 - Train Accuracy:  0.762, Validation Accuracy:  0.772, Loss:  0.265
    Epoch  13 Batch   42/269 - Train Accuracy:  0.779, Validation Accuracy:  0.774, Loss:  0.245
    Epoch  13 Batch   43/269 - Train Accuracy:  0.766, Validation Accuracy:  0.775, Loss:  0.263
    Epoch  13 Batch   44/269 - Train Accuracy:  0.785, Validation Accuracy:  0.776, Loss:  0.256
    Epoch  13 Batch   45/269 - Train Accuracy:  0.758, Validation Accuracy:  0.775, Loss:  0.264
    Epoch  13 Batch   46/269 - Train Accuracy:  0.769, Validation Accuracy:  0.776, Loss:  0.268
    Epoch  13 Batch   47/269 - Train Accuracy:  0.796, Validation Accuracy:  0.778, Loss:  0.243
    Epoch  13 Batch   48/269 - Train Accuracy:  0.771, Validation Accuracy:  0.778, Loss:  0.250
    Epoch  13 Batch   49/269 - Train Accuracy:  0.759, Validation Accuracy:  0.777, Loss:  0.259
    Epoch  13 Batch   50/269 - Train Accuracy:  0.761, Validation Accuracy:  0.776, Loss:  0.268
    Epoch  13 Batch   51/269 - Train Accuracy:  0.766, Validation Accuracy:  0.776, Loss:  0.259
    Epoch  13 Batch   52/269 - Train Accuracy:  0.756, Validation Accuracy:  0.775, Loss:  0.246
    Epoch  13 Batch   53/269 - Train Accuracy:  0.760, Validation Accuracy:  0.771, Loss:  0.268
    Epoch  13 Batch   54/269 - Train Accuracy:  0.782, Validation Accuracy:  0.770, Loss:  0.251
    Epoch  13 Batch   55/269 - Train Accuracy:  0.776, Validation Accuracy:  0.771, Loss:  0.258
    Epoch  13 Batch   56/269 - Train Accuracy:  0.785, Validation Accuracy:  0.770, Loss:  0.252
    Epoch  13 Batch   57/269 - Train Accuracy:  0.757, Validation Accuracy:  0.773, Loss:  0.262
    Epoch  13 Batch   58/269 - Train Accuracy:  0.775, Validation Accuracy:  0.776, Loss:  0.251
    Epoch  13 Batch   59/269 - Train Accuracy:  0.792, Validation Accuracy:  0.775, Loss:  0.233
    Epoch  13 Batch   60/269 - Train Accuracy:  0.758, Validation Accuracy:  0.776, Loss:  0.250
    Epoch  13 Batch   61/269 - Train Accuracy:  0.784, Validation Accuracy:  0.779, Loss:  0.236
    Epoch  13 Batch   62/269 - Train Accuracy:  0.775, Validation Accuracy:  0.778, Loss:  0.242
    Epoch  13 Batch   63/269 - Train Accuracy:  0.769, Validation Accuracy:  0.776, Loss:  0.257
    Epoch  13 Batch   64/269 - Train Accuracy:  0.768, Validation Accuracy:  0.777, Loss:  0.248
    Epoch  13 Batch   65/269 - Train Accuracy:  0.771, Validation Accuracy:  0.777, Loss:  0.252
    Epoch  13 Batch   66/269 - Train Accuracy:  0.763, Validation Accuracy:  0.778, Loss:  0.251
    Epoch  13 Batch   67/269 - Train Accuracy:  0.757, Validation Accuracy:  0.777, Loss:  0.261
    Epoch  13 Batch   68/269 - Train Accuracy:  0.748, Validation Accuracy:  0.779, Loss:  0.263
    Epoch  13 Batch   69/269 - Train Accuracy:  0.752, Validation Accuracy:  0.775, Loss:  0.288
    Epoch  13 Batch   70/269 - Train Accuracy:  0.784, Validation Accuracy:  0.775, Loss:  0.254
    Epoch  13 Batch   71/269 - Train Accuracy:  0.767, Validation Accuracy:  0.775, Loss:  0.270
    Epoch  13 Batch   72/269 - Train Accuracy:  0.760, Validation Accuracy:  0.774, Loss:  0.257
    Epoch  13 Batch   73/269 - Train Accuracy:  0.773, Validation Accuracy:  0.776, Loss:  0.260
    Epoch  13 Batch   74/269 - Train Accuracy:  0.760, Validation Accuracy:  0.774, Loss:  0.255
    Epoch  13 Batch   75/269 - Train Accuracy:  0.750, Validation Accuracy:  0.772, Loss:  0.255
    Epoch  13 Batch   76/269 - Train Accuracy:  0.755, Validation Accuracy:  0.770, Loss:  0.259
    Epoch  13 Batch   77/269 - Train Accuracy:  0.775, Validation Accuracy:  0.769, Loss:  0.252
    Epoch  13 Batch   78/269 - Train Accuracy:  0.776, Validation Accuracy:  0.772, Loss:  0.250
    Epoch  13 Batch   79/269 - Train Accuracy:  0.766, Validation Accuracy:  0.772, Loss:  0.256
    Epoch  13 Batch   80/269 - Train Accuracy:  0.781, Validation Accuracy:  0.771, Loss:  0.251
    Epoch  13 Batch   81/269 - Train Accuracy:  0.770, Validation Accuracy:  0.774, Loss:  0.259
    Epoch  13 Batch   82/269 - Train Accuracy:  0.787, Validation Accuracy:  0.776, Loss:  0.234
    Epoch  13 Batch   83/269 - Train Accuracy:  0.766, Validation Accuracy:  0.773, Loss:  0.275
    Epoch  13 Batch   84/269 - Train Accuracy:  0.777, Validation Accuracy:  0.773, Loss:  0.257
    Epoch  13 Batch   85/269 - Train Accuracy:  0.779, Validation Accuracy:  0.772, Loss:  0.261
    Epoch  13 Batch   86/269 - Train Accuracy:  0.761, Validation Accuracy:  0.771, Loss:  0.246
    Epoch  13 Batch   87/269 - Train Accuracy:  0.753, Validation Accuracy:  0.773, Loss:  0.268
    Epoch  13 Batch   88/269 - Train Accuracy:  0.767, Validation Accuracy:  0.773, Loss:  0.252
    Epoch  13 Batch   89/269 - Train Accuracy:  0.783, Validation Accuracy:  0.774, Loss:  0.252
    Epoch  13 Batch   90/269 - Train Accuracy:  0.736, Validation Accuracy:  0.775, Loss:  0.271
    Epoch  13 Batch   91/269 - Train Accuracy:  0.770, Validation Accuracy:  0.774, Loss:  0.244
    Epoch  13 Batch   92/269 - Train Accuracy:  0.766, Validation Accuracy:  0.775, Loss:  0.251
    Epoch  13 Batch   93/269 - Train Accuracy:  0.778, Validation Accuracy:  0.775, Loss:  0.243
    Epoch  13 Batch   94/269 - Train Accuracy:  0.762, Validation Accuracy:  0.771, Loss:  0.267
    Epoch  13 Batch   95/269 - Train Accuracy:  0.758, Validation Accuracy:  0.769, Loss:  0.254
    Epoch  13 Batch   96/269 - Train Accuracy:  0.761, Validation Accuracy:  0.771, Loss:  0.259
    Epoch  13 Batch   97/269 - Train Accuracy:  0.771, Validation Accuracy:  0.771, Loss:  0.250
    Epoch  13 Batch   98/269 - Train Accuracy:  0.764, Validation Accuracy:  0.771, Loss:  0.253
    Epoch  13 Batch   99/269 - Train Accuracy:  0.773, Validation Accuracy:  0.774, Loss:  0.253
    Epoch  13 Batch  100/269 - Train Accuracy:  0.794, Validation Accuracy:  0.772, Loss:  0.249
    Epoch  13 Batch  101/269 - Train Accuracy:  0.739, Validation Accuracy:  0.774, Loss:  0.276
    Epoch  13 Batch  102/269 - Train Accuracy:  0.772, Validation Accuracy:  0.777, Loss:  0.247
    Epoch  13 Batch  103/269 - Train Accuracy:  0.776, Validation Accuracy:  0.777, Loss:  0.249
    Epoch  13 Batch  104/269 - Train Accuracy:  0.764, Validation Accuracy:  0.776, Loss:  0.250
    Epoch  13 Batch  105/269 - Train Accuracy:  0.758, Validation Accuracy:  0.776, Loss:  0.252
    Epoch  13 Batch  106/269 - Train Accuracy:  0.766, Validation Accuracy:  0.778, Loss:  0.249
    Epoch  13 Batch  107/269 - Train Accuracy:  0.752, Validation Accuracy:  0.779, Loss:  0.268
    Epoch  13 Batch  108/269 - Train Accuracy:  0.760, Validation Accuracy:  0.777, Loss:  0.249
    Epoch  13 Batch  109/269 - Train Accuracy:  0.741, Validation Accuracy:  0.775, Loss:  0.258
    Epoch  13 Batch  110/269 - Train Accuracy:  0.757, Validation Accuracy:  0.768, Loss:  0.245
    Epoch  13 Batch  111/269 - Train Accuracy:  0.757, Validation Accuracy:  0.767, Loss:  0.267
    Epoch  13 Batch  112/269 - Train Accuracy:  0.773, Validation Accuracy:  0.766, Loss:  0.254
    Epoch  13 Batch  113/269 - Train Accuracy:  0.755, Validation Accuracy:  0.765, Loss:  0.243
    Epoch  13 Batch  114/269 - Train Accuracy:  0.758, Validation Accuracy:  0.763, Loss:  0.249
    Epoch  13 Batch  115/269 - Train Accuracy:  0.740, Validation Accuracy:  0.764, Loss:  0.269
    Epoch  13 Batch  116/269 - Train Accuracy:  0.774, Validation Accuracy:  0.765, Loss:  0.255
    Epoch  13 Batch  117/269 - Train Accuracy:  0.769, Validation Accuracy:  0.772, Loss:  0.249
    Epoch  13 Batch  118/269 - Train Accuracy:  0.800, Validation Accuracy:  0.771, Loss:  0.240
    Epoch  13 Batch  119/269 - Train Accuracy:  0.775, Validation Accuracy:  0.775, Loss:  0.258
    Epoch  13 Batch  120/269 - Train Accuracy:  0.763, Validation Accuracy:  0.776, Loss:  0.261
    Epoch  13 Batch  121/269 - Train Accuracy:  0.769, Validation Accuracy:  0.777, Loss:  0.246
    Epoch  13 Batch  122/269 - Train Accuracy:  0.779, Validation Accuracy:  0.777, Loss:  0.248
    Epoch  13 Batch  123/269 - Train Accuracy:  0.761, Validation Accuracy:  0.777, Loss:  0.254
    Epoch  13 Batch  124/269 - Train Accuracy:  0.773, Validation Accuracy:  0.776, Loss:  0.246
    Epoch  13 Batch  125/269 - Train Accuracy:  0.764, Validation Accuracy:  0.777, Loss:  0.248
    Epoch  13 Batch  126/269 - Train Accuracy:  0.767, Validation Accuracy:  0.778, Loss:  0.250
    Epoch  13 Batch  127/269 - Train Accuracy:  0.751, Validation Accuracy:  0.779, Loss:  0.261
    Epoch  13 Batch  128/269 - Train Accuracy:  0.785, Validation Accuracy:  0.779, Loss:  0.250
    Epoch  13 Batch  129/269 - Train Accuracy:  0.774, Validation Accuracy:  0.774, Loss:  0.244
    Epoch  13 Batch  130/269 - Train Accuracy:  0.762, Validation Accuracy:  0.775, Loss:  0.255
    Epoch  13 Batch  131/269 - Train Accuracy:  0.757, Validation Accuracy:  0.774, Loss:  0.253
    Epoch  13 Batch  132/269 - Train Accuracy:  0.762, Validation Accuracy:  0.770, Loss:  0.259
    Epoch  13 Batch  133/269 - Train Accuracy:  0.770, Validation Accuracy:  0.770, Loss:  0.241
    Epoch  13 Batch  134/269 - Train Accuracy:  0.744, Validation Accuracy:  0.769, Loss:  0.255
    Epoch  13 Batch  135/269 - Train Accuracy:  0.743, Validation Accuracy:  0.768, Loss:  0.270
    Epoch  13 Batch  136/269 - Train Accuracy:  0.749, Validation Accuracy:  0.768, Loss:  0.267
    Epoch  13 Batch  137/269 - Train Accuracy:  0.769, Validation Accuracy:  0.771, Loss:  0.265
    Epoch  13 Batch  138/269 - Train Accuracy:  0.758, Validation Accuracy:  0.773, Loss:  0.255
    Epoch  13 Batch  139/269 - Train Accuracy:  0.777, Validation Accuracy:  0.776, Loss:  0.238
    Epoch  13 Batch  140/269 - Train Accuracy:  0.769, Validation Accuracy:  0.775, Loss:  0.261
    Epoch  13 Batch  141/269 - Train Accuracy:  0.769, Validation Accuracy:  0.775, Loss:  0.257
    Epoch  13 Batch  142/269 - Train Accuracy:  0.784, Validation Accuracy:  0.777, Loss:  0.239
    Epoch  13 Batch  143/269 - Train Accuracy:  0.785, Validation Accuracy:  0.781, Loss:  0.245
    Epoch  13 Batch  144/269 - Train Accuracy:  0.781, Validation Accuracy:  0.782, Loss:  0.230
    Epoch  13 Batch  145/269 - Train Accuracy:  0.770, Validation Accuracy:  0.779, Loss:  0.244
    Epoch  13 Batch  146/269 - Train Accuracy:  0.776, Validation Accuracy:  0.779, Loss:  0.236
    Epoch  13 Batch  147/269 - Train Accuracy:  0.781, Validation Accuracy:  0.775, Loss:  0.243
    Epoch  13 Batch  148/269 - Train Accuracy:  0.778, Validation Accuracy:  0.778, Loss:  0.249
    Epoch  13 Batch  149/269 - Train Accuracy:  0.770, Validation Accuracy:  0.779, Loss:  0.247
    Epoch  13 Batch  150/269 - Train Accuracy:  0.779, Validation Accuracy:  0.780, Loss:  0.243
    Epoch  13 Batch  151/269 - Train Accuracy:  0.788, Validation Accuracy:  0.778, Loss:  0.237
    Epoch  13 Batch  152/269 - Train Accuracy:  0.773, Validation Accuracy:  0.774, Loss:  0.250
    Epoch  13 Batch  153/269 - Train Accuracy:  0.778, Validation Accuracy:  0.772, Loss:  0.245
    Epoch  13 Batch  154/269 - Train Accuracy:  0.748, Validation Accuracy:  0.775, Loss:  0.247
    Epoch  13 Batch  155/269 - Train Accuracy:  0.776, Validation Accuracy:  0.779, Loss:  0.239
    Epoch  13 Batch  156/269 - Train Accuracy:  0.766, Validation Accuracy:  0.780, Loss:  0.256
    Epoch  13 Batch  157/269 - Train Accuracy:  0.763, Validation Accuracy:  0.784, Loss:  0.242
    Epoch  13 Batch  158/269 - Train Accuracy:  0.769, Validation Accuracy:  0.783, Loss:  0.247
    Epoch  13 Batch  159/269 - Train Accuracy:  0.770, Validation Accuracy:  0.783, Loss:  0.250
    Epoch  13 Batch  160/269 - Train Accuracy:  0.776, Validation Accuracy:  0.781, Loss:  0.236
    Epoch  13 Batch  161/269 - Train Accuracy:  0.772, Validation Accuracy:  0.783, Loss:  0.247
    Epoch  13 Batch  162/269 - Train Accuracy:  0.780, Validation Accuracy:  0.782, Loss:  0.247
    Epoch  13 Batch  163/269 - Train Accuracy:  0.779, Validation Accuracy:  0.783, Loss:  0.247
    Epoch  13 Batch  164/269 - Train Accuracy:  0.780, Validation Accuracy:  0.785, Loss:  0.240
    Epoch  13 Batch  165/269 - Train Accuracy:  0.750, Validation Accuracy:  0.782, Loss:  0.254
    Epoch  13 Batch  166/269 - Train Accuracy:  0.777, Validation Accuracy:  0.783, Loss:  0.230
    Epoch  13 Batch  167/269 - Train Accuracy:  0.777, Validation Accuracy:  0.779, Loss:  0.246
    Epoch  13 Batch  168/269 - Train Accuracy:  0.774, Validation Accuracy:  0.778, Loss:  0.248
    Epoch  13 Batch  169/269 - Train Accuracy:  0.760, Validation Accuracy:  0.779, Loss:  0.251
    Epoch  13 Batch  170/269 - Train Accuracy:  0.776, Validation Accuracy:  0.779, Loss:  0.237
    Epoch  13 Batch  171/269 - Train Accuracy:  0.771, Validation Accuracy:  0.782, Loss:  0.256
    Epoch  13 Batch  172/269 - Train Accuracy:  0.759, Validation Accuracy:  0.783, Loss:  0.260
    Epoch  13 Batch  173/269 - Train Accuracy:  0.773, Validation Accuracy:  0.779, Loss:  0.239
    Epoch  13 Batch  174/269 - Train Accuracy:  0.782, Validation Accuracy:  0.784, Loss:  0.241
    Epoch  13 Batch  175/269 - Train Accuracy:  0.778, Validation Accuracy:  0.781, Loss:  0.261
    Epoch  13 Batch  176/269 - Train Accuracy:  0.762, Validation Accuracy:  0.782, Loss:  0.258
    Epoch  13 Batch  177/269 - Train Accuracy:  0.788, Validation Accuracy:  0.781, Loss:  0.235
    Epoch  13 Batch  178/269 - Train Accuracy:  0.777, Validation Accuracy:  0.781, Loss:  0.245
    Epoch  13 Batch  179/269 - Train Accuracy:  0.777, Validation Accuracy:  0.783, Loss:  0.244
    Epoch  13 Batch  180/269 - Train Accuracy:  0.776, Validation Accuracy:  0.780, Loss:  0.240
    Epoch  13 Batch  181/269 - Train Accuracy:  0.746, Validation Accuracy:  0.779, Loss:  0.244
    Epoch  13 Batch  182/269 - Train Accuracy:  0.781, Validation Accuracy:  0.781, Loss:  0.244
    Epoch  13 Batch  183/269 - Train Accuracy:  0.808, Validation Accuracy:  0.783, Loss:  0.213
    Epoch  13 Batch  184/269 - Train Accuracy:  0.766, Validation Accuracy:  0.783, Loss:  0.251
    Epoch  13 Batch  185/269 - Train Accuracy:  0.784, Validation Accuracy:  0.784, Loss:  0.240
    Epoch  13 Batch  186/269 - Train Accuracy:  0.765, Validation Accuracy:  0.784, Loss:  0.244
    Epoch  13 Batch  187/269 - Train Accuracy:  0.777, Validation Accuracy:  0.787, Loss:  0.239
    Epoch  13 Batch  188/269 - Train Accuracy:  0.795, Validation Accuracy:  0.787, Loss:  0.231
    Epoch  13 Batch  189/269 - Train Accuracy:  0.783, Validation Accuracy:  0.788, Loss:  0.236
    Epoch  13 Batch  190/269 - Train Accuracy:  0.790, Validation Accuracy:  0.787, Loss:  0.238
    Epoch  13 Batch  191/269 - Train Accuracy:  0.780, Validation Accuracy:  0.785, Loss:  0.232
    Epoch  13 Batch  192/269 - Train Accuracy:  0.788, Validation Accuracy:  0.783, Loss:  0.238
    Epoch  13 Batch  193/269 - Train Accuracy:  0.776, Validation Accuracy:  0.783, Loss:  0.234
    Epoch  13 Batch  194/269 - Train Accuracy:  0.791, Validation Accuracy:  0.784, Loss:  0.244
    Epoch  13 Batch  195/269 - Train Accuracy:  0.776, Validation Accuracy:  0.782, Loss:  0.240
    Epoch  13 Batch  196/269 - Train Accuracy:  0.754, Validation Accuracy:  0.780, Loss:  0.246
    Epoch  13 Batch  197/269 - Train Accuracy:  0.747, Validation Accuracy:  0.778, Loss:  0.253
    Epoch  13 Batch  198/269 - Train Accuracy:  0.761, Validation Accuracy:  0.779, Loss:  0.251
    Epoch  13 Batch  199/269 - Train Accuracy:  0.767, Validation Accuracy:  0.780, Loss:  0.245
    Epoch  13 Batch  200/269 - Train Accuracy:  0.788, Validation Accuracy:  0.780, Loss:  0.249
    Epoch  13 Batch  201/269 - Train Accuracy:  0.771, Validation Accuracy:  0.780, Loss:  0.241
    Epoch  13 Batch  202/269 - Train Accuracy:  0.773, Validation Accuracy:  0.780, Loss:  0.248
    Epoch  13 Batch  203/269 - Train Accuracy:  0.773, Validation Accuracy:  0.780, Loss:  0.255
    Epoch  13 Batch  204/269 - Train Accuracy:  0.761, Validation Accuracy:  0.783, Loss:  0.252
    Epoch  13 Batch  205/269 - Train Accuracy:  0.761, Validation Accuracy:  0.784, Loss:  0.235
    Epoch  13 Batch  206/269 - Train Accuracy:  0.776, Validation Accuracy:  0.781, Loss:  0.249
    Epoch  13 Batch  207/269 - Train Accuracy:  0.779, Validation Accuracy:  0.783, Loss:  0.240
    Epoch  13 Batch  208/269 - Train Accuracy:  0.774, Validation Accuracy:  0.782, Loss:  0.250
    Epoch  13 Batch  209/269 - Train Accuracy:  0.778, Validation Accuracy:  0.784, Loss:  0.241
    Epoch  13 Batch  210/269 - Train Accuracy:  0.779, Validation Accuracy:  0.787, Loss:  0.240
    Epoch  13 Batch  211/269 - Train Accuracy:  0.778, Validation Accuracy:  0.784, Loss:  0.246
    Epoch  13 Batch  212/269 - Train Accuracy:  0.780, Validation Accuracy:  0.784, Loss:  0.247
    Epoch  13 Batch  213/269 - Train Accuracy:  0.770, Validation Accuracy:  0.785, Loss:  0.244
    Epoch  13 Batch  214/269 - Train Accuracy:  0.786, Validation Accuracy:  0.786, Loss:  0.241
    Epoch  13 Batch  215/269 - Train Accuracy:  0.794, Validation Accuracy:  0.784, Loss:  0.226
    Epoch  13 Batch  216/269 - Train Accuracy:  0.763, Validation Accuracy:  0.784, Loss:  0.269
    Epoch  13 Batch  217/269 - Train Accuracy:  0.754, Validation Accuracy:  0.785, Loss:  0.253
    Epoch  13 Batch  218/269 - Train Accuracy:  0.774, Validation Accuracy:  0.784, Loss:  0.244
    Epoch  13 Batch  219/269 - Train Accuracy:  0.788, Validation Accuracy:  0.788, Loss:  0.251
    Epoch  13 Batch  220/269 - Train Accuracy:  0.772, Validation Accuracy:  0.791, Loss:  0.233
    Epoch  13 Batch  221/269 - Train Accuracy:  0.799, Validation Accuracy:  0.789, Loss:  0.240
    Epoch  13 Batch  222/269 - Train Accuracy:  0.785, Validation Accuracy:  0.788, Loss:  0.231
    Epoch  13 Batch  223/269 - Train Accuracy:  0.781, Validation Accuracy:  0.782, Loss:  0.233
    Epoch  13 Batch  224/269 - Train Accuracy:  0.790, Validation Accuracy:  0.785, Loss:  0.250
    Epoch  13 Batch  225/269 - Train Accuracy:  0.776, Validation Accuracy:  0.783, Loss:  0.237
    Epoch  13 Batch  226/269 - Train Accuracy:  0.780, Validation Accuracy:  0.782, Loss:  0.238
    Epoch  13 Batch  227/269 - Train Accuracy:  0.807, Validation Accuracy:  0.780, Loss:  0.220
    Epoch  13 Batch  228/269 - Train Accuracy:  0.768, Validation Accuracy:  0.780, Loss:  0.234
    Epoch  13 Batch  229/269 - Train Accuracy:  0.772, Validation Accuracy:  0.783, Loss:  0.233
    Epoch  13 Batch  230/269 - Train Accuracy:  0.759, Validation Accuracy:  0.784, Loss:  0.239
    Epoch  13 Batch  231/269 - Train Accuracy:  0.752, Validation Accuracy:  0.780, Loss:  0.250
    Epoch  13 Batch  232/269 - Train Accuracy:  0.753, Validation Accuracy:  0.782, Loss:  0.254
    Epoch  13 Batch  233/269 - Train Accuracy:  0.779, Validation Accuracy:  0.781, Loss:  0.246
    Epoch  13 Batch  234/269 - Train Accuracy:  0.770, Validation Accuracy:  0.781, Loss:  0.239
    Epoch  13 Batch  235/269 - Train Accuracy:  0.804, Validation Accuracy:  0.781, Loss:  0.226
    Epoch  13 Batch  236/269 - Train Accuracy:  0.757, Validation Accuracy:  0.780, Loss:  0.234
    Epoch  13 Batch  237/269 - Train Accuracy:  0.764, Validation Accuracy:  0.783, Loss:  0.232
    Epoch  13 Batch  238/269 - Train Accuracy:  0.775, Validation Accuracy:  0.782, Loss:  0.229
    Epoch  13 Batch  239/269 - Train Accuracy:  0.791, Validation Accuracy:  0.785, Loss:  0.237
    Epoch  13 Batch  240/269 - Train Accuracy:  0.799, Validation Accuracy:  0.786, Loss:  0.211
    Epoch  13 Batch  241/269 - Train Accuracy:  0.769, Validation Accuracy:  0.789, Loss:  0.240
    Epoch  13 Batch  242/269 - Train Accuracy:  0.759, Validation Accuracy:  0.791, Loss:  0.236
    Epoch  13 Batch  243/269 - Train Accuracy:  0.795, Validation Accuracy:  0.789, Loss:  0.225
    Epoch  13 Batch  244/269 - Train Accuracy:  0.769, Validation Accuracy:  0.789, Loss:  0.243
    Epoch  13 Batch  245/269 - Train Accuracy:  0.771, Validation Accuracy:  0.787, Loss:  0.247
    Epoch  13 Batch  246/269 - Train Accuracy:  0.768, Validation Accuracy:  0.786, Loss:  0.237
    Epoch  13 Batch  247/269 - Train Accuracy:  0.781, Validation Accuracy:  0.786, Loss:  0.242
    Epoch  13 Batch  248/269 - Train Accuracy:  0.777, Validation Accuracy:  0.788, Loss:  0.234
    Epoch  13 Batch  249/269 - Train Accuracy:  0.805, Validation Accuracy:  0.787, Loss:  0.213
    Epoch  13 Batch  250/269 - Train Accuracy:  0.782, Validation Accuracy:  0.788, Loss:  0.235
    Epoch  13 Batch  251/269 - Train Accuracy:  0.794, Validation Accuracy:  0.791, Loss:  0.226
    Epoch  13 Batch  252/269 - Train Accuracy:  0.781, Validation Accuracy:  0.791, Loss:  0.236
    Epoch  13 Batch  253/269 - Train Accuracy:  0.772, Validation Accuracy:  0.787, Loss:  0.244
    Epoch  13 Batch  254/269 - Train Accuracy:  0.787, Validation Accuracy:  0.786, Loss:  0.238
    Epoch  13 Batch  255/269 - Train Accuracy:  0.793, Validation Accuracy:  0.787, Loss:  0.220
    Epoch  13 Batch  256/269 - Train Accuracy:  0.756, Validation Accuracy:  0.788, Loss:  0.242
    Epoch  13 Batch  257/269 - Train Accuracy:  0.756, Validation Accuracy:  0.790, Loss:  0.246
    Epoch  13 Batch  258/269 - Train Accuracy:  0.768, Validation Accuracy:  0.789, Loss:  0.239
    Epoch  13 Batch  259/269 - Train Accuracy:  0.795, Validation Accuracy:  0.789, Loss:  0.234
    Epoch  13 Batch  260/269 - Train Accuracy:  0.772, Validation Accuracy:  0.787, Loss:  0.255
    Epoch  13 Batch  261/269 - Train Accuracy:  0.757, Validation Accuracy:  0.787, Loss:  0.251
    Epoch  13 Batch  262/269 - Train Accuracy:  0.795, Validation Accuracy:  0.786, Loss:  0.228
    Epoch  13 Batch  263/269 - Train Accuracy:  0.777, Validation Accuracy:  0.786, Loss:  0.242
    Epoch  13 Batch  264/269 - Train Accuracy:  0.772, Validation Accuracy:  0.789, Loss:  0.244
    Epoch  13 Batch  265/269 - Train Accuracy:  0.768, Validation Accuracy:  0.786, Loss:  0.239
    Epoch  13 Batch  266/269 - Train Accuracy:  0.777, Validation Accuracy:  0.783, Loss:  0.230
    Epoch  13 Batch  267/269 - Train Accuracy:  0.788, Validation Accuracy:  0.784, Loss:  0.245
    Epoch  14 Batch    0/269 - Train Accuracy:  0.767, Validation Accuracy:  0.783, Loss:  0.246
    Epoch  14 Batch    1/269 - Train Accuracy:  0.769, Validation Accuracy:  0.779, Loss:  0.238
    Epoch  14 Batch    2/269 - Train Accuracy:  0.772, Validation Accuracy:  0.778, Loss:  0.242
    Epoch  14 Batch    3/269 - Train Accuracy:  0.772, Validation Accuracy:  0.778, Loss:  0.240
    Epoch  14 Batch    4/269 - Train Accuracy:  0.751, Validation Accuracy:  0.774, Loss:  0.249
    Epoch  14 Batch    5/269 - Train Accuracy:  0.756, Validation Accuracy:  0.775, Loss:  0.243
    Epoch  14 Batch    6/269 - Train Accuracy:  0.762, Validation Accuracy:  0.774, Loss:  0.229
    Epoch  14 Batch    7/269 - Train Accuracy:  0.785, Validation Accuracy:  0.776, Loss:  0.227
    Epoch  14 Batch    8/269 - Train Accuracy:  0.765, Validation Accuracy:  0.776, Loss:  0.250
    Epoch  14 Batch    9/269 - Train Accuracy:  0.773, Validation Accuracy:  0.776, Loss:  0.241
    Epoch  14 Batch   10/269 - Train Accuracy:  0.769, Validation Accuracy:  0.777, Loss:  0.237
    Epoch  14 Batch   11/269 - Train Accuracy:  0.772, Validation Accuracy:  0.782, Loss:  0.239
    Epoch  14 Batch   12/269 - Train Accuracy:  0.762, Validation Accuracy:  0.789, Loss:  0.250
    Epoch  14 Batch   13/269 - Train Accuracy:  0.790, Validation Accuracy:  0.789, Loss:  0.214
    Epoch  14 Batch   14/269 - Train Accuracy:  0.779, Validation Accuracy:  0.785, Loss:  0.240
    Epoch  14 Batch   15/269 - Train Accuracy:  0.777, Validation Accuracy:  0.786, Loss:  0.226
    Epoch  14 Batch   16/269 - Train Accuracy:  0.804, Validation Accuracy:  0.787, Loss:  0.234
    Epoch  14 Batch   17/269 - Train Accuracy:  0.779, Validation Accuracy:  0.788, Loss:  0.223
    Epoch  14 Batch   18/269 - Train Accuracy:  0.775, Validation Accuracy:  0.789, Loss:  0.241
    Epoch  14 Batch   19/269 - Train Accuracy:  0.800, Validation Accuracy:  0.789, Loss:  0.218
    Epoch  14 Batch   20/269 - Train Accuracy:  0.781, Validation Accuracy:  0.785, Loss:  0.240
    Epoch  14 Batch   21/269 - Train Accuracy:  0.778, Validation Accuracy:  0.785, Loss:  0.253
    Epoch  14 Batch   22/269 - Train Accuracy:  0.786, Validation Accuracy:  0.784, Loss:  0.221
    Epoch  14 Batch   23/269 - Train Accuracy:  0.789, Validation Accuracy:  0.786, Loss:  0.231
    Epoch  14 Batch   24/269 - Train Accuracy:  0.773, Validation Accuracy:  0.788, Loss:  0.239
    Epoch  14 Batch   25/269 - Train Accuracy:  0.767, Validation Accuracy:  0.791, Loss:  0.248
    Epoch  14 Batch   26/269 - Train Accuracy:  0.790, Validation Accuracy:  0.788, Loss:  0.213
    Epoch  14 Batch   27/269 - Train Accuracy:  0.774, Validation Accuracy:  0.786, Loss:  0.229
    Epoch  14 Batch   28/269 - Train Accuracy:  0.742, Validation Accuracy:  0.779, Loss:  0.256
    Epoch  14 Batch   29/269 - Train Accuracy:  0.779, Validation Accuracy:  0.783, Loss:  0.247
    Epoch  14 Batch   30/269 - Train Accuracy:  0.782, Validation Accuracy:  0.780, Loss:  0.226
    Epoch  14 Batch   31/269 - Train Accuracy:  0.788, Validation Accuracy:  0.782, Loss:  0.219
    Epoch  14 Batch   32/269 - Train Accuracy:  0.769, Validation Accuracy:  0.779, Loss:  0.222
    Epoch  14 Batch   33/269 - Train Accuracy:  0.795, Validation Accuracy:  0.778, Loss:  0.221
    Epoch  14 Batch   34/269 - Train Accuracy:  0.782, Validation Accuracy:  0.779, Loss:  0.231
    Epoch  14 Batch   35/269 - Train Accuracy:  0.782, Validation Accuracy:  0.781, Loss:  0.243
    Epoch  14 Batch   36/269 - Train Accuracy:  0.778, Validation Accuracy:  0.782, Loss:  0.230
    Epoch  14 Batch   37/269 - Train Accuracy:  0.785, Validation Accuracy:  0.785, Loss:  0.229
    Epoch  14 Batch   38/269 - Train Accuracy:  0.785, Validation Accuracy:  0.786, Loss:  0.224
    Epoch  14 Batch   39/269 - Train Accuracy:  0.775, Validation Accuracy:  0.790, Loss:  0.237
    Epoch  14 Batch   40/269 - Train Accuracy:  0.772, Validation Accuracy:  0.783, Loss:  0.247
    Epoch  14 Batch   41/269 - Train Accuracy:  0.770, Validation Accuracy:  0.780, Loss:  0.235
    Epoch  14 Batch   42/269 - Train Accuracy:  0.790, Validation Accuracy:  0.778, Loss:  0.219
    Epoch  14 Batch   43/269 - Train Accuracy:  0.778, Validation Accuracy:  0.780, Loss:  0.234
    Epoch  14 Batch   44/269 - Train Accuracy:  0.791, Validation Accuracy:  0.782, Loss:  0.226
    Epoch  14 Batch   45/269 - Train Accuracy:  0.782, Validation Accuracy:  0.788, Loss:  0.239
    Epoch  14 Batch   46/269 - Train Accuracy:  0.787, Validation Accuracy:  0.786, Loss:  0.238
    Epoch  14 Batch   47/269 - Train Accuracy:  0.798, Validation Accuracy:  0.781, Loss:  0.216
    Epoch  14 Batch   48/269 - Train Accuracy:  0.785, Validation Accuracy:  0.782, Loss:  0.220
    Epoch  14 Batch   49/269 - Train Accuracy:  0.760, Validation Accuracy:  0.782, Loss:  0.229
    Epoch  14 Batch   50/269 - Train Accuracy:  0.767, Validation Accuracy:  0.781, Loss:  0.244
    Epoch  14 Batch   51/269 - Train Accuracy:  0.775, Validation Accuracy:  0.783, Loss:  0.229
    Epoch  14 Batch   52/269 - Train Accuracy:  0.773, Validation Accuracy:  0.784, Loss:  0.216
    Epoch  14 Batch   53/269 - Train Accuracy:  0.773, Validation Accuracy:  0.779, Loss:  0.246
    Epoch  14 Batch   54/269 - Train Accuracy:  0.795, Validation Accuracy:  0.781, Loss:  0.225
    Epoch  14 Batch   55/269 - Train Accuracy:  0.784, Validation Accuracy:  0.781, Loss:  0.230
    Epoch  14 Batch   56/269 - Train Accuracy:  0.791, Validation Accuracy:  0.786, Loss:  0.226
    Epoch  14 Batch   57/269 - Train Accuracy:  0.781, Validation Accuracy:  0.793, Loss:  0.235
    Epoch  14 Batch   58/269 - Train Accuracy:  0.782, Validation Accuracy:  0.793, Loss:  0.222
    Epoch  14 Batch   59/269 - Train Accuracy:  0.810, Validation Accuracy:  0.796, Loss:  0.208
    Epoch  14 Batch   60/269 - Train Accuracy:  0.771, Validation Accuracy:  0.793, Loss:  0.219
    Epoch  14 Batch   61/269 - Train Accuracy:  0.805, Validation Accuracy:  0.788, Loss:  0.211
    Epoch  14 Batch   62/269 - Train Accuracy:  0.801, Validation Accuracy:  0.783, Loss:  0.217
    Epoch  14 Batch   63/269 - Train Accuracy:  0.788, Validation Accuracy:  0.786, Loss:  0.230
    Epoch  14 Batch   64/269 - Train Accuracy:  0.782, Validation Accuracy:  0.790, Loss:  0.222
    Epoch  14 Batch   65/269 - Train Accuracy:  0.781, Validation Accuracy:  0.792, Loss:  0.227
    Epoch  14 Batch   66/269 - Train Accuracy:  0.775, Validation Accuracy:  0.788, Loss:  0.226
    Epoch  14 Batch   67/269 - Train Accuracy:  0.768, Validation Accuracy:  0.795, Loss:  0.233
    Epoch  14 Batch   68/269 - Train Accuracy:  0.765, Validation Accuracy:  0.788, Loss:  0.237
    Epoch  14 Batch   69/269 - Train Accuracy:  0.753, Validation Accuracy:  0.788, Loss:  0.261
    Epoch  14 Batch   70/269 - Train Accuracy:  0.792, Validation Accuracy:  0.788, Loss:  0.226
    Epoch  14 Batch   71/269 - Train Accuracy:  0.780, Validation Accuracy:  0.788, Loss:  0.246
    Epoch  14 Batch   72/269 - Train Accuracy:  0.767, Validation Accuracy:  0.790, Loss:  0.237
    Epoch  14 Batch   73/269 - Train Accuracy:  0.792, Validation Accuracy:  0.784, Loss:  0.232
    Epoch  14 Batch   74/269 - Train Accuracy:  0.772, Validation Accuracy:  0.786, Loss:  0.226
    Epoch  14 Batch   75/269 - Train Accuracy:  0.768, Validation Accuracy:  0.789, Loss:  0.229
    Epoch  14 Batch   76/269 - Train Accuracy:  0.766, Validation Accuracy:  0.793, Loss:  0.233
    Epoch  14 Batch   77/269 - Train Accuracy:  0.789, Validation Accuracy:  0.794, Loss:  0.226
    Epoch  14 Batch   78/269 - Train Accuracy:  0.792, Validation Accuracy:  0.791, Loss:  0.222
    Epoch  14 Batch   79/269 - Train Accuracy:  0.778, Validation Accuracy:  0.790, Loss:  0.233
    Epoch  14 Batch   80/269 - Train Accuracy:  0.792, Validation Accuracy:  0.791, Loss:  0.220
    Epoch  14 Batch   81/269 - Train Accuracy:  0.778, Validation Accuracy:  0.789, Loss:  0.234
    Epoch  14 Batch   82/269 - Train Accuracy:  0.797, Validation Accuracy:  0.787, Loss:  0.209
    Epoch  14 Batch   83/269 - Train Accuracy:  0.774, Validation Accuracy:  0.791, Loss:  0.241
    Epoch  14 Batch   84/269 - Train Accuracy:  0.784, Validation Accuracy:  0.794, Loss:  0.228
    Epoch  14 Batch   85/269 - Train Accuracy:  0.781, Validation Accuracy:  0.790, Loss:  0.226
    Epoch  14 Batch   86/269 - Train Accuracy:  0.777, Validation Accuracy:  0.786, Loss:  0.217
    Epoch  14 Batch   87/269 - Train Accuracy:  0.767, Validation Accuracy:  0.784, Loss:  0.239
    Epoch  14 Batch   88/269 - Train Accuracy:  0.776, Validation Accuracy:  0.786, Loss:  0.224
    Epoch  14 Batch   89/269 - Train Accuracy:  0.790, Validation Accuracy:  0.786, Loss:  0.232
    Epoch  14 Batch   90/269 - Train Accuracy:  0.753, Validation Accuracy:  0.786, Loss:  0.243
    Epoch  14 Batch   91/269 - Train Accuracy:  0.793, Validation Accuracy:  0.788, Loss:  0.218
    Epoch  14 Batch   92/269 - Train Accuracy:  0.788, Validation Accuracy:  0.785, Loss:  0.221
    Epoch  14 Batch   93/269 - Train Accuracy:  0.791, Validation Accuracy:  0.780, Loss:  0.215
    Epoch  14 Batch   94/269 - Train Accuracy:  0.782, Validation Accuracy:  0.779, Loss:  0.238
    Epoch  14 Batch   95/269 - Train Accuracy:  0.775, Validation Accuracy:  0.779, Loss:  0.228
    Epoch  14 Batch   96/269 - Train Accuracy:  0.765, Validation Accuracy:  0.787, Loss:  0.233
    Epoch  14 Batch   97/269 - Train Accuracy:  0.792, Validation Accuracy:  0.789, Loss:  0.225
    Epoch  14 Batch   98/269 - Train Accuracy:  0.780, Validation Accuracy:  0.787, Loss:  0.225
    Epoch  14 Batch   99/269 - Train Accuracy:  0.797, Validation Accuracy:  0.789, Loss:  0.225
    Epoch  14 Batch  100/269 - Train Accuracy:  0.813, Validation Accuracy:  0.788, Loss:  0.226
    Epoch  14 Batch  101/269 - Train Accuracy:  0.757, Validation Accuracy:  0.787, Loss:  0.239
    Epoch  14 Batch  102/269 - Train Accuracy:  0.776, Validation Accuracy:  0.784, Loss:  0.219
    Epoch  14 Batch  103/269 - Train Accuracy:  0.778, Validation Accuracy:  0.787, Loss:  0.224
    Epoch  14 Batch  104/269 - Train Accuracy:  0.771, Validation Accuracy:  0.787, Loss:  0.222
    Epoch  14 Batch  105/269 - Train Accuracy:  0.766, Validation Accuracy:  0.787, Loss:  0.230
    Epoch  14 Batch  106/269 - Train Accuracy:  0.777, Validation Accuracy:  0.789, Loss:  0.219
    Epoch  14 Batch  107/269 - Train Accuracy:  0.762, Validation Accuracy:  0.790, Loss:  0.238
    Epoch  14 Batch  108/269 - Train Accuracy:  0.777, Validation Accuracy:  0.789, Loss:  0.223
    Epoch  14 Batch  109/269 - Train Accuracy:  0.751, Validation Accuracy:  0.789, Loss:  0.230
    Epoch  14 Batch  110/269 - Train Accuracy:  0.779, Validation Accuracy:  0.782, Loss:  0.217
    Epoch  14 Batch  111/269 - Train Accuracy:  0.781, Validation Accuracy:  0.782, Loss:  0.238
    Epoch  14 Batch  112/269 - Train Accuracy:  0.776, Validation Accuracy:  0.781, Loss:  0.227
    Epoch  14 Batch  113/269 - Train Accuracy:  0.782, Validation Accuracy:  0.776, Loss:  0.221
    Epoch  14 Batch  114/269 - Train Accuracy:  0.775, Validation Accuracy:  0.772, Loss:  0.221
    Epoch  14 Batch  115/269 - Train Accuracy:  0.757, Validation Accuracy:  0.775, Loss:  0.237
    Epoch  14 Batch  116/269 - Train Accuracy:  0.781, Validation Accuracy:  0.779, Loss:  0.225
    Epoch  14 Batch  117/269 - Train Accuracy:  0.786, Validation Accuracy:  0.780, Loss:  0.221
    Epoch  14 Batch  118/269 - Train Accuracy:  0.809, Validation Accuracy:  0.785, Loss:  0.214
    Epoch  14 Batch  119/269 - Train Accuracy:  0.782, Validation Accuracy:  0.786, Loss:  0.231
    Epoch  14 Batch  120/269 - Train Accuracy:  0.771, Validation Accuracy:  0.789, Loss:  0.234
    Epoch  14 Batch  121/269 - Train Accuracy:  0.774, Validation Accuracy:  0.787, Loss:  0.219
    Epoch  14 Batch  122/269 - Train Accuracy:  0.776, Validation Accuracy:  0.789, Loss:  0.224
    Epoch  14 Batch  123/269 - Train Accuracy:  0.786, Validation Accuracy:  0.792, Loss:  0.227
    Epoch  14 Batch  124/269 - Train Accuracy:  0.774, Validation Accuracy:  0.788, Loss:  0.218
    Epoch  14 Batch  125/269 - Train Accuracy:  0.777, Validation Accuracy:  0.783, Loss:  0.223
    Epoch  14 Batch  126/269 - Train Accuracy:  0.781, Validation Accuracy:  0.786, Loss:  0.221
    Epoch  14 Batch  127/269 - Train Accuracy:  0.772, Validation Accuracy:  0.788, Loss:  0.226
    Epoch  14 Batch  128/269 - Train Accuracy:  0.788, Validation Accuracy:  0.787, Loss:  0.228
    Epoch  14 Batch  129/269 - Train Accuracy:  0.778, Validation Accuracy:  0.787, Loss:  0.220
    Epoch  14 Batch  130/269 - Train Accuracy:  0.770, Validation Accuracy:  0.784, Loss:  0.232
    Epoch  14 Batch  131/269 - Train Accuracy:  0.772, Validation Accuracy:  0.781, Loss:  0.227
    Epoch  14 Batch  132/269 - Train Accuracy:  0.771, Validation Accuracy:  0.783, Loss:  0.233
    Epoch  14 Batch  133/269 - Train Accuracy:  0.777, Validation Accuracy:  0.782, Loss:  0.216
    Epoch  14 Batch  134/269 - Train Accuracy:  0.752, Validation Accuracy:  0.779, Loss:  0.227
    Epoch  14 Batch  135/269 - Train Accuracy:  0.764, Validation Accuracy:  0.778, Loss:  0.239
    Epoch  14 Batch  136/269 - Train Accuracy:  0.749, Validation Accuracy:  0.782, Loss:  0.243
    Epoch  14 Batch  137/269 - Train Accuracy:  0.781, Validation Accuracy:  0.786, Loss:  0.234
    Epoch  14 Batch  138/269 - Train Accuracy:  0.777, Validation Accuracy:  0.789, Loss:  0.232
    Epoch  14 Batch  139/269 - Train Accuracy:  0.790, Validation Accuracy:  0.789, Loss:  0.214
    Epoch  14 Batch  140/269 - Train Accuracy:  0.786, Validation Accuracy:  0.788, Loss:  0.231
    Epoch  14 Batch  141/269 - Train Accuracy:  0.773, Validation Accuracy:  0.787, Loss:  0.232
    Epoch  14 Batch  142/269 - Train Accuracy:  0.795, Validation Accuracy:  0.787, Loss:  0.214
    Epoch  14 Batch  143/269 - Train Accuracy:  0.805, Validation Accuracy:  0.786, Loss:  0.218
    Epoch  14 Batch  144/269 - Train Accuracy:  0.792, Validation Accuracy:  0.789, Loss:  0.203
    Epoch  14 Batch  145/269 - Train Accuracy:  0.780, Validation Accuracy:  0.790, Loss:  0.219
    Epoch  14 Batch  146/269 - Train Accuracy:  0.779, Validation Accuracy:  0.790, Loss:  0.218
    Epoch  14 Batch  147/269 - Train Accuracy:  0.795, Validation Accuracy:  0.786, Loss:  0.219
    Epoch  14 Batch  148/269 - Train Accuracy:  0.785, Validation Accuracy:  0.787, Loss:  0.222
    Epoch  14 Batch  149/269 - Train Accuracy:  0.774, Validation Accuracy:  0.790, Loss:  0.227
    Epoch  14 Batch  150/269 - Train Accuracy:  0.784, Validation Accuracy:  0.790, Loss:  0.220
    Epoch  14 Batch  151/269 - Train Accuracy:  0.794, Validation Accuracy:  0.786, Loss:  0.218
    Epoch  14 Batch  152/269 - Train Accuracy:  0.794, Validation Accuracy:  0.787, Loss:  0.221
    Epoch  14 Batch  153/269 - Train Accuracy:  0.786, Validation Accuracy:  0.791, Loss:  0.220
    Epoch  14 Batch  154/269 - Train Accuracy:  0.771, Validation Accuracy:  0.791, Loss:  0.226
    Epoch  14 Batch  155/269 - Train Accuracy:  0.786, Validation Accuracy:  0.790, Loss:  0.213
    Epoch  14 Batch  156/269 - Train Accuracy:  0.780, Validation Accuracy:  0.791, Loss:  0.231
    Epoch  14 Batch  157/269 - Train Accuracy:  0.786, Validation Accuracy:  0.793, Loss:  0.221
    Epoch  14 Batch  158/269 - Train Accuracy:  0.785, Validation Accuracy:  0.788, Loss:  0.227
    Epoch  14 Batch  159/269 - Train Accuracy:  0.779, Validation Accuracy:  0.786, Loss:  0.221
    Epoch  14 Batch  160/269 - Train Accuracy:  0.790, Validation Accuracy:  0.787, Loss:  0.211
    Epoch  14 Batch  161/269 - Train Accuracy:  0.786, Validation Accuracy:  0.787, Loss:  0.222
    Epoch  14 Batch  162/269 - Train Accuracy:  0.780, Validation Accuracy:  0.789, Loss:  0.227
    Epoch  14 Batch  163/269 - Train Accuracy:  0.789, Validation Accuracy:  0.789, Loss:  0.221
    Epoch  14 Batch  164/269 - Train Accuracy:  0.787, Validation Accuracy:  0.789, Loss:  0.217
    Epoch  14 Batch  165/269 - Train Accuracy:  0.758, Validation Accuracy:  0.785, Loss:  0.225
    Epoch  14 Batch  166/269 - Train Accuracy:  0.792, Validation Accuracy:  0.780, Loss:  0.207
    Epoch  14 Batch  167/269 - Train Accuracy:  0.787, Validation Accuracy:  0.783, Loss:  0.219
    Epoch  14 Batch  168/269 - Train Accuracy:  0.776, Validation Accuracy:  0.787, Loss:  0.221
    Epoch  14 Batch  169/269 - Train Accuracy:  0.775, Validation Accuracy:  0.787, Loss:  0.225
    Epoch  14 Batch  170/269 - Train Accuracy:  0.771, Validation Accuracy:  0.786, Loss:  0.216
    Epoch  14 Batch  171/269 - Train Accuracy:  0.789, Validation Accuracy:  0.785, Loss:  0.228
    Epoch  14 Batch  172/269 - Train Accuracy:  0.771, Validation Accuracy:  0.784, Loss:  0.232
    Epoch  14 Batch  173/269 - Train Accuracy:  0.787, Validation Accuracy:  0.782, Loss:  0.215
    Epoch  14 Batch  174/269 - Train Accuracy:  0.792, Validation Accuracy:  0.783, Loss:  0.214
    Epoch  14 Batch  175/269 - Train Accuracy:  0.782, Validation Accuracy:  0.783, Loss:  0.235
    Epoch  14 Batch  176/269 - Train Accuracy:  0.770, Validation Accuracy:  0.785, Loss:  0.235
    Epoch  14 Batch  177/269 - Train Accuracy:  0.796, Validation Accuracy:  0.784, Loss:  0.212
    Epoch  14 Batch  178/269 - Train Accuracy:  0.776, Validation Accuracy:  0.785, Loss:  0.223
    Epoch  14 Batch  179/269 - Train Accuracy:  0.777, Validation Accuracy:  0.780, Loss:  0.219
    Epoch  14 Batch  180/269 - Train Accuracy:  0.785, Validation Accuracy:  0.776, Loss:  0.217
    Epoch  14 Batch  181/269 - Train Accuracy:  0.760, Validation Accuracy:  0.775, Loss:  0.221
    Epoch  14 Batch  182/269 - Train Accuracy:  0.784, Validation Accuracy:  0.778, Loss:  0.221
    Epoch  14 Batch  183/269 - Train Accuracy:  0.809, Validation Accuracy:  0.780, Loss:  0.191
    Epoch  14 Batch  184/269 - Train Accuracy:  0.777, Validation Accuracy:  0.785, Loss:  0.228
    Epoch  14 Batch  185/269 - Train Accuracy:  0.799, Validation Accuracy:  0.786, Loss:  0.214
    Epoch  14 Batch  186/269 - Train Accuracy:  0.773, Validation Accuracy:  0.783, Loss:  0.216
    Epoch  14 Batch  187/269 - Train Accuracy:  0.788, Validation Accuracy:  0.784, Loss:  0.214
    Epoch  14 Batch  188/269 - Train Accuracy:  0.791, Validation Accuracy:  0.787, Loss:  0.213
    Epoch  14 Batch  189/269 - Train Accuracy:  0.789, Validation Accuracy:  0.787, Loss:  0.208
    Epoch  14 Batch  190/269 - Train Accuracy:  0.805, Validation Accuracy:  0.789, Loss:  0.214
    Epoch  14 Batch  191/269 - Train Accuracy:  0.795, Validation Accuracy:  0.790, Loss:  0.212
    Epoch  14 Batch  192/269 - Train Accuracy:  0.795, Validation Accuracy:  0.790, Loss:  0.215
    Epoch  14 Batch  193/269 - Train Accuracy:  0.785, Validation Accuracy:  0.786, Loss:  0.212
    Epoch  14 Batch  194/269 - Train Accuracy:  0.791, Validation Accuracy:  0.786, Loss:  0.223
    Epoch  14 Batch  195/269 - Train Accuracy:  0.787, Validation Accuracy:  0.787, Loss:  0.218
    Epoch  14 Batch  196/269 - Train Accuracy:  0.775, Validation Accuracy:  0.787, Loss:  0.216
    Epoch  14 Batch  197/269 - Train Accuracy:  0.759, Validation Accuracy:  0.781, Loss:  0.229
    Epoch  14 Batch  198/269 - Train Accuracy:  0.771, Validation Accuracy:  0.782, Loss:  0.225
    Epoch  14 Batch  199/269 - Train Accuracy:  0.782, Validation Accuracy:  0.781, Loss:  0.221
    Epoch  14 Batch  200/269 - Train Accuracy:  0.792, Validation Accuracy:  0.784, Loss:  0.224
    Epoch  14 Batch  201/269 - Train Accuracy:  0.778, Validation Accuracy:  0.782, Loss:  0.220
    Epoch  14 Batch  202/269 - Train Accuracy:  0.785, Validation Accuracy:  0.787, Loss:  0.221
    Epoch  14 Batch  203/269 - Train Accuracy:  0.778, Validation Accuracy:  0.791, Loss:  0.228
    Epoch  14 Batch  204/269 - Train Accuracy:  0.767, Validation Accuracy:  0.791, Loss:  0.229
    Epoch  14 Batch  205/269 - Train Accuracy:  0.792, Validation Accuracy:  0.794, Loss:  0.214
    Epoch  14 Batch  206/269 - Train Accuracy:  0.786, Validation Accuracy:  0.798, Loss:  0.223
    Epoch  14 Batch  207/269 - Train Accuracy:  0.787, Validation Accuracy:  0.796, Loss:  0.213
    Epoch  14 Batch  208/269 - Train Accuracy:  0.787, Validation Accuracy:  0.793, Loss:  0.224
    Epoch  14 Batch  209/269 - Train Accuracy:  0.782, Validation Accuracy:  0.791, Loss:  0.217
    Epoch  14 Batch  210/269 - Train Accuracy:  0.788, Validation Accuracy:  0.796, Loss:  0.213
    Epoch  14 Batch  211/269 - Train Accuracy:  0.789, Validation Accuracy:  0.792, Loss:  0.219
    Epoch  14 Batch  212/269 - Train Accuracy:  0.788, Validation Accuracy:  0.791, Loss:  0.222
    Epoch  14 Batch  213/269 - Train Accuracy:  0.778, Validation Accuracy:  0.792, Loss:  0.218
    Epoch  14 Batch  214/269 - Train Accuracy:  0.800, Validation Accuracy:  0.791, Loss:  0.215
    Epoch  14 Batch  215/269 - Train Accuracy:  0.815, Validation Accuracy:  0.792, Loss:  0.200
    Epoch  14 Batch  216/269 - Train Accuracy:  0.772, Validation Accuracy:  0.794, Loss:  0.245
    Epoch  14 Batch  217/269 - Train Accuracy:  0.765, Validation Accuracy:  0.792, Loss:  0.225
    Epoch  14 Batch  218/269 - Train Accuracy:  0.783, Validation Accuracy:  0.790, Loss:  0.216
    Epoch  14 Batch  219/269 - Train Accuracy:  0.791, Validation Accuracy:  0.788, Loss:  0.226
    Epoch  14 Batch  220/269 - Train Accuracy:  0.776, Validation Accuracy:  0.787, Loss:  0.210
    Epoch  14 Batch  221/269 - Train Accuracy:  0.799, Validation Accuracy:  0.790, Loss:  0.215
    Epoch  14 Batch  222/269 - Train Accuracy:  0.795, Validation Accuracy:  0.788, Loss:  0.207
    Epoch  14 Batch  223/269 - Train Accuracy:  0.793, Validation Accuracy:  0.790, Loss:  0.211
    Epoch  14 Batch  224/269 - Train Accuracy:  0.797, Validation Accuracy:  0.787, Loss:  0.221
    Epoch  14 Batch  225/269 - Train Accuracy:  0.793, Validation Accuracy:  0.793, Loss:  0.216
    Epoch  14 Batch  226/269 - Train Accuracy:  0.784, Validation Accuracy:  0.791, Loss:  0.217
    Epoch  14 Batch  227/269 - Train Accuracy:  0.810, Validation Accuracy:  0.790, Loss:  0.200
    Epoch  14 Batch  228/269 - Train Accuracy:  0.779, Validation Accuracy:  0.792, Loss:  0.211
    Epoch  14 Batch  229/269 - Train Accuracy:  0.787, Validation Accuracy:  0.798, Loss:  0.207
    Epoch  14 Batch  230/269 - Train Accuracy:  0.780, Validation Accuracy:  0.794, Loss:  0.217
    Epoch  14 Batch  231/269 - Train Accuracy:  0.765, Validation Accuracy:  0.797, Loss:  0.224
    Epoch  14 Batch  232/269 - Train Accuracy:  0.772, Validation Accuracy:  0.795, Loss:  0.229
    Epoch  14 Batch  233/269 - Train Accuracy:  0.803, Validation Accuracy:  0.793, Loss:  0.222
    Epoch  14 Batch  234/269 - Train Accuracy:  0.785, Validation Accuracy:  0.792, Loss:  0.218
    Epoch  14 Batch  235/269 - Train Accuracy:  0.816, Validation Accuracy:  0.793, Loss:  0.202
    Epoch  14 Batch  236/269 - Train Accuracy:  0.768, Validation Accuracy:  0.787, Loss:  0.214
    Epoch  14 Batch  237/269 - Train Accuracy:  0.783, Validation Accuracy:  0.790, Loss:  0.209
    Epoch  14 Batch  238/269 - Train Accuracy:  0.796, Validation Accuracy:  0.789, Loss:  0.209
    Epoch  14 Batch  239/269 - Train Accuracy:  0.795, Validation Accuracy:  0.790, Loss:  0.210
    Epoch  14 Batch  240/269 - Train Accuracy:  0.821, Validation Accuracy:  0.788, Loss:  0.192
    Epoch  14 Batch  241/269 - Train Accuracy:  0.796, Validation Accuracy:  0.788, Loss:  0.213
    Epoch  14 Batch  242/269 - Train Accuracy:  0.776, Validation Accuracy:  0.788, Loss:  0.213
    Epoch  14 Batch  243/269 - Train Accuracy:  0.796, Validation Accuracy:  0.787, Loss:  0.202
    Epoch  14 Batch  244/269 - Train Accuracy:  0.794, Validation Accuracy:  0.788, Loss:  0.217
    Epoch  14 Batch  245/269 - Train Accuracy:  0.781, Validation Accuracy:  0.787, Loss:  0.220
    Epoch  14 Batch  246/269 - Train Accuracy:  0.787, Validation Accuracy:  0.790, Loss:  0.216
    Epoch  14 Batch  247/269 - Train Accuracy:  0.792, Validation Accuracy:  0.786, Loss:  0.218
    Epoch  14 Batch  248/269 - Train Accuracy:  0.789, Validation Accuracy:  0.790, Loss:  0.209
    Epoch  14 Batch  249/269 - Train Accuracy:  0.816, Validation Accuracy:  0.793, Loss:  0.193
    Epoch  14 Batch  250/269 - Train Accuracy:  0.796, Validation Accuracy:  0.789, Loss:  0.212
    Epoch  14 Batch  251/269 - Train Accuracy:  0.807, Validation Accuracy:  0.788, Loss:  0.205
    Epoch  14 Batch  252/269 - Train Accuracy:  0.789, Validation Accuracy:  0.785, Loss:  0.212
    Epoch  14 Batch  253/269 - Train Accuracy:  0.780, Validation Accuracy:  0.795, Loss:  0.223
    Epoch  14 Batch  254/269 - Train Accuracy:  0.809, Validation Accuracy:  0.793, Loss:  0.211
    Epoch  14 Batch  255/269 - Train Accuracy:  0.805, Validation Accuracy:  0.795, Loss:  0.201
    Epoch  14 Batch  256/269 - Train Accuracy:  0.757, Validation Accuracy:  0.797, Loss:  0.221
    Epoch  14 Batch  257/269 - Train Accuracy:  0.772, Validation Accuracy:  0.796, Loss:  0.223
    Epoch  14 Batch  258/269 - Train Accuracy:  0.790, Validation Accuracy:  0.794, Loss:  0.212
    Epoch  14 Batch  259/269 - Train Accuracy:  0.816, Validation Accuracy:  0.793, Loss:  0.214
    Epoch  14 Batch  260/269 - Train Accuracy:  0.785, Validation Accuracy:  0.790, Loss:  0.224
    Epoch  14 Batch  261/269 - Train Accuracy:  0.775, Validation Accuracy:  0.793, Loss:  0.225
    Epoch  14 Batch  262/269 - Train Accuracy:  0.798, Validation Accuracy:  0.795, Loss:  0.208
    Epoch  14 Batch  263/269 - Train Accuracy:  0.795, Validation Accuracy:  0.796, Loss:  0.223
    Epoch  14 Batch  264/269 - Train Accuracy:  0.787, Validation Accuracy:  0.799, Loss:  0.222
    Epoch  14 Batch  265/269 - Train Accuracy:  0.784, Validation Accuracy:  0.799, Loss:  0.213
    Epoch  14 Batch  266/269 - Train Accuracy:  0.802, Validation Accuracy:  0.790, Loss:  0.209
    Epoch  14 Batch  267/269 - Train Accuracy:  0.807, Validation Accuracy:  0.789, Loss:  0.217
    Epoch  15 Batch    0/269 - Train Accuracy:  0.787, Validation Accuracy:  0.789, Loss:  0.222
    Epoch  15 Batch    1/269 - Train Accuracy:  0.779, Validation Accuracy:  0.788, Loss:  0.215
    Epoch  15 Batch    2/269 - Train Accuracy:  0.785, Validation Accuracy:  0.788, Loss:  0.219
    Epoch  15 Batch    3/269 - Train Accuracy:  0.786, Validation Accuracy:  0.793, Loss:  0.213
    Epoch  15 Batch    4/269 - Train Accuracy:  0.764, Validation Accuracy:  0.793, Loss:  0.224
    Epoch  15 Batch    5/269 - Train Accuracy:  0.774, Validation Accuracy:  0.791, Loss:  0.219
    Epoch  15 Batch    6/269 - Train Accuracy:  0.774, Validation Accuracy:  0.794, Loss:  0.207
    Epoch  15 Batch    7/269 - Train Accuracy:  0.806, Validation Accuracy:  0.793, Loss:  0.207
    Epoch  15 Batch    8/269 - Train Accuracy:  0.789, Validation Accuracy:  0.795, Loss:  0.225
    Epoch  15 Batch    9/269 - Train Accuracy:  0.808, Validation Accuracy:  0.798, Loss:  0.220
    Epoch  15 Batch   10/269 - Train Accuracy:  0.793, Validation Accuracy:  0.794, Loss:  0.211
    Epoch  15 Batch   11/269 - Train Accuracy:  0.782, Validation Accuracy:  0.794, Loss:  0.212
    Epoch  15 Batch   12/269 - Train Accuracy:  0.790, Validation Accuracy:  0.796, Loss:  0.225
    Epoch  15 Batch   13/269 - Train Accuracy:  0.810, Validation Accuracy:  0.795, Loss:  0.191
    Epoch  15 Batch   14/269 - Train Accuracy:  0.798, Validation Accuracy:  0.797, Loss:  0.216
    Epoch  15 Batch   15/269 - Train Accuracy:  0.798, Validation Accuracy:  0.793, Loss:  0.201
    Epoch  15 Batch   16/269 - Train Accuracy:  0.805, Validation Accuracy:  0.796, Loss:  0.211
    Epoch  15 Batch   17/269 - Train Accuracy:  0.801, Validation Accuracy:  0.799, Loss:  0.203
    Epoch  15 Batch   18/269 - Train Accuracy:  0.786, Validation Accuracy:  0.800, Loss:  0.217
    Epoch  15 Batch   19/269 - Train Accuracy:  0.799, Validation Accuracy:  0.801, Loss:  0.192
    Epoch  15 Batch   20/269 - Train Accuracy:  0.792, Validation Accuracy:  0.800, Loss:  0.216
    Epoch  15 Batch   21/269 - Train Accuracy:  0.786, Validation Accuracy:  0.801, Loss:  0.235
    Epoch  15 Batch   22/269 - Train Accuracy:  0.801, Validation Accuracy:  0.798, Loss:  0.199
    Epoch  15 Batch   23/269 - Train Accuracy:  0.804, Validation Accuracy:  0.804, Loss:  0.205
    Epoch  15 Batch   24/269 - Train Accuracy:  0.792, Validation Accuracy:  0.800, Loss:  0.214
    Epoch  15 Batch   25/269 - Train Accuracy:  0.784, Validation Accuracy:  0.796, Loss:  0.226
    Epoch  15 Batch   26/269 - Train Accuracy:  0.794, Validation Accuracy:  0.797, Loss:  0.196
    Epoch  15 Batch   27/269 - Train Accuracy:  0.783, Validation Accuracy:  0.800, Loss:  0.205
    Epoch  15 Batch   28/269 - Train Accuracy:  0.772, Validation Accuracy:  0.798, Loss:  0.230
    Epoch  15 Batch   29/269 - Train Accuracy:  0.800, Validation Accuracy:  0.799, Loss:  0.221
    Epoch  15 Batch   30/269 - Train Accuracy:  0.808, Validation Accuracy:  0.799, Loss:  0.206
    Epoch  15 Batch   31/269 - Train Accuracy:  0.808, Validation Accuracy:  0.800, Loss:  0.196
    Epoch  15 Batch   32/269 - Train Accuracy:  0.784, Validation Accuracy:  0.800, Loss:  0.199
    Epoch  15 Batch   33/269 - Train Accuracy:  0.814, Validation Accuracy:  0.805, Loss:  0.198
    Epoch  15 Batch   34/269 - Train Accuracy:  0.797, Validation Accuracy:  0.802, Loss:  0.208
    Epoch  15 Batch   35/269 - Train Accuracy:  0.792, Validation Accuracy:  0.800, Loss:  0.222
    Epoch  15 Batch   36/269 - Train Accuracy:  0.787, Validation Accuracy:  0.800, Loss:  0.207
    Epoch  15 Batch   37/269 - Train Accuracy:  0.797, Validation Accuracy:  0.796, Loss:  0.202
    Epoch  15 Batch   38/269 - Train Accuracy:  0.807, Validation Accuracy:  0.796, Loss:  0.205
    Epoch  15 Batch   39/269 - Train Accuracy:  0.801, Validation Accuracy:  0.793, Loss:  0.212
    Epoch  15 Batch   40/269 - Train Accuracy:  0.791, Validation Accuracy:  0.797, Loss:  0.216
    Epoch  15 Batch   41/269 - Train Accuracy:  0.783, Validation Accuracy:  0.796, Loss:  0.214
    Epoch  15 Batch   42/269 - Train Accuracy:  0.806, Validation Accuracy:  0.800, Loss:  0.192
    Epoch  15 Batch   43/269 - Train Accuracy:  0.799, Validation Accuracy:  0.799, Loss:  0.209
    Epoch  15 Batch   44/269 - Train Accuracy:  0.791, Validation Accuracy:  0.791, Loss:  0.205
    Epoch  15 Batch   45/269 - Train Accuracy:  0.797, Validation Accuracy:  0.792, Loss:  0.210
    Epoch  15 Batch   46/269 - Train Accuracy:  0.798, Validation Accuracy:  0.796, Loss:  0.217
    Epoch  15 Batch   47/269 - Train Accuracy:  0.814, Validation Accuracy:  0.790, Loss:  0.195
    Epoch  15 Batch   48/269 - Train Accuracy:  0.807, Validation Accuracy:  0.794, Loss:  0.201
    Epoch  15 Batch   49/269 - Train Accuracy:  0.775, Validation Accuracy:  0.797, Loss:  0.205
    Epoch  15 Batch   50/269 - Train Accuracy:  0.776, Validation Accuracy:  0.800, Loss:  0.221
    Epoch  15 Batch   51/269 - Train Accuracy:  0.795, Validation Accuracy:  0.802, Loss:  0.207
    Epoch  15 Batch   52/269 - Train Accuracy:  0.794, Validation Accuracy:  0.808, Loss:  0.194
    Epoch  15 Batch   53/269 - Train Accuracy:  0.796, Validation Accuracy:  0.800, Loss:  0.222
    Epoch  15 Batch   54/269 - Train Accuracy:  0.823, Validation Accuracy:  0.798, Loss:  0.202
    Epoch  15 Batch   55/269 - Train Accuracy:  0.804, Validation Accuracy:  0.797, Loss:  0.208
    Epoch  15 Batch   56/269 - Train Accuracy:  0.806, Validation Accuracy:  0.799, Loss:  0.206
    Epoch  15 Batch   57/269 - Train Accuracy:  0.803, Validation Accuracy:  0.801, Loss:  0.210
    Epoch  15 Batch   58/269 - Train Accuracy:  0.799, Validation Accuracy:  0.803, Loss:  0.200
    Epoch  15 Batch   59/269 - Train Accuracy:  0.836, Validation Accuracy:  0.804, Loss:  0.183
    Epoch  15 Batch   60/269 - Train Accuracy:  0.801, Validation Accuracy:  0.803, Loss:  0.201
    Epoch  15 Batch   61/269 - Train Accuracy:  0.821, Validation Accuracy:  0.805, Loss:  0.189
    Epoch  15 Batch   62/269 - Train Accuracy:  0.829, Validation Accuracy:  0.805, Loss:  0.199
    Epoch  15 Batch   63/269 - Train Accuracy:  0.810, Validation Accuracy:  0.808, Loss:  0.212
    Epoch  15 Batch   64/269 - Train Accuracy:  0.796, Validation Accuracy:  0.809, Loss:  0.198
    Epoch  15 Batch   65/269 - Train Accuracy:  0.787, Validation Accuracy:  0.804, Loss:  0.206
    Epoch  15 Batch   66/269 - Train Accuracy:  0.799, Validation Accuracy:  0.804, Loss:  0.202
    Epoch  15 Batch   67/269 - Train Accuracy:  0.785, Validation Accuracy:  0.809, Loss:  0.211
    Epoch  15 Batch   68/269 - Train Accuracy:  0.790, Validation Accuracy:  0.810, Loss:  0.217
    Epoch  15 Batch   69/269 - Train Accuracy:  0.772, Validation Accuracy:  0.811, Loss:  0.238
    Epoch  15 Batch   70/269 - Train Accuracy:  0.820, Validation Accuracy:  0.807, Loss:  0.202
    Epoch  15 Batch   71/269 - Train Accuracy:  0.800, Validation Accuracy:  0.806, Loss:  0.222
    Epoch  15 Batch   72/269 - Train Accuracy:  0.790, Validation Accuracy:  0.804, Loss:  0.213
    Epoch  15 Batch   73/269 - Train Accuracy:  0.805, Validation Accuracy:  0.804, Loss:  0.208
    Epoch  15 Batch   74/269 - Train Accuracy:  0.793, Validation Accuracy:  0.802, Loss:  0.204
    Epoch  15 Batch   75/269 - Train Accuracy:  0.796, Validation Accuracy:  0.808, Loss:  0.205
    Epoch  15 Batch   76/269 - Train Accuracy:  0.799, Validation Accuracy:  0.812, Loss:  0.205
    Epoch  15 Batch   77/269 - Train Accuracy:  0.811, Validation Accuracy:  0.810, Loss:  0.200
    Epoch  15 Batch   78/269 - Train Accuracy:  0.812, Validation Accuracy:  0.811, Loss:  0.204
    Epoch  15 Batch   79/269 - Train Accuracy:  0.800, Validation Accuracy:  0.809, Loss:  0.206
    Epoch  15 Batch   80/269 - Train Accuracy:  0.808, Validation Accuracy:  0.801, Loss:  0.195
    Epoch  15 Batch   81/269 - Train Accuracy:  0.790, Validation Accuracy:  0.799, Loss:  0.209
    Epoch  15 Batch   82/269 - Train Accuracy:  0.814, Validation Accuracy:  0.806, Loss:  0.187
    Epoch  15 Batch   83/269 - Train Accuracy:  0.794, Validation Accuracy:  0.803, Loss:  0.219
    Epoch  15 Batch   84/269 - Train Accuracy:  0.799, Validation Accuracy:  0.799, Loss:  0.198
    Epoch  15 Batch   85/269 - Train Accuracy:  0.804, Validation Accuracy:  0.799, Loss:  0.205
    Epoch  15 Batch   86/269 - Train Accuracy:  0.813, Validation Accuracy:  0.804, Loss:  0.197
    Epoch  15 Batch   87/269 - Train Accuracy:  0.783, Validation Accuracy:  0.810, Loss:  0.212
    Epoch  15 Batch   88/269 - Train Accuracy:  0.801, Validation Accuracy:  0.805, Loss:  0.206
    Epoch  15 Batch   89/269 - Train Accuracy:  0.822, Validation Accuracy:  0.802, Loss:  0.202
    Epoch  15 Batch   90/269 - Train Accuracy:  0.777, Validation Accuracy:  0.800, Loss:  0.217
    Epoch  15 Batch   91/269 - Train Accuracy:  0.826, Validation Accuracy:  0.797, Loss:  0.195
    Epoch  15 Batch   92/269 - Train Accuracy:  0.808, Validation Accuracy:  0.801, Loss:  0.197
    Epoch  15 Batch   93/269 - Train Accuracy:  0.813, Validation Accuracy:  0.807, Loss:  0.189
    Epoch  15 Batch   94/269 - Train Accuracy:  0.805, Validation Accuracy:  0.802, Loss:  0.209
    Epoch  15 Batch   95/269 - Train Accuracy:  0.798, Validation Accuracy:  0.798, Loss:  0.199
    Epoch  15 Batch   96/269 - Train Accuracy:  0.793, Validation Accuracy:  0.794, Loss:  0.212
    Epoch  15 Batch   97/269 - Train Accuracy:  0.824, Validation Accuracy:  0.797, Loss:  0.198
    Epoch  15 Batch   98/269 - Train Accuracy:  0.805, Validation Accuracy:  0.803, Loss:  0.203
    Epoch  15 Batch   99/269 - Train Accuracy:  0.812, Validation Accuracy:  0.809, Loss:  0.205
    Epoch  15 Batch  100/269 - Train Accuracy:  0.826, Validation Accuracy:  0.810, Loss:  0.201
    Epoch  15 Batch  101/269 - Train Accuracy:  0.791, Validation Accuracy:  0.812, Loss:  0.220
    Epoch  15 Batch  102/269 - Train Accuracy:  0.799, Validation Accuracy:  0.804, Loss:  0.198
    Epoch  15 Batch  103/269 - Train Accuracy:  0.810, Validation Accuracy:  0.805, Loss:  0.200
    Epoch  15 Batch  104/269 - Train Accuracy:  0.796, Validation Accuracy:  0.803, Loss:  0.200
    Epoch  15 Batch  105/269 - Train Accuracy:  0.800, Validation Accuracy:  0.812, Loss:  0.211
    Epoch  15 Batch  106/269 - Train Accuracy:  0.801, Validation Accuracy:  0.812, Loss:  0.194
    Epoch  15 Batch  107/269 - Train Accuracy:  0.801, Validation Accuracy:  0.810, Loss:  0.210
    Epoch  15 Batch  108/269 - Train Accuracy:  0.813, Validation Accuracy:  0.814, Loss:  0.200
    Epoch  15 Batch  109/269 - Train Accuracy:  0.785, Validation Accuracy:  0.812, Loss:  0.201
    Epoch  15 Batch  110/269 - Train Accuracy:  0.805, Validation Accuracy:  0.814, Loss:  0.191
    Epoch  15 Batch  111/269 - Train Accuracy:  0.817, Validation Accuracy:  0.811, Loss:  0.211
    Epoch  15 Batch  112/269 - Train Accuracy:  0.794, Validation Accuracy:  0.809, Loss:  0.205
    Epoch  15 Batch  113/269 - Train Accuracy:  0.809, Validation Accuracy:  0.804, Loss:  0.196
    Epoch  15 Batch  114/269 - Train Accuracy:  0.804, Validation Accuracy:  0.804, Loss:  0.199
    Epoch  15 Batch  115/269 - Train Accuracy:  0.784, Validation Accuracy:  0.805, Loss:  0.211
    Epoch  15 Batch  116/269 - Train Accuracy:  0.827, Validation Accuracy:  0.808, Loss:  0.201
    Epoch  15 Batch  117/269 - Train Accuracy:  0.823, Validation Accuracy:  0.816, Loss:  0.200
    Epoch  15 Batch  118/269 - Train Accuracy:  0.838, Validation Accuracy:  0.809, Loss:  0.191
    Epoch  15 Batch  119/269 - Train Accuracy:  0.804, Validation Accuracy:  0.805, Loss:  0.204
    Epoch  15 Batch  120/269 - Train Accuracy:  0.811, Validation Accuracy:  0.818, Loss:  0.208
    Epoch  15 Batch  121/269 - Train Accuracy:  0.803, Validation Accuracy:  0.818, Loss:  0.195
    Epoch  15 Batch  122/269 - Train Accuracy:  0.807, Validation Accuracy:  0.821, Loss:  0.199
    Epoch  15 Batch  123/269 - Train Accuracy:  0.808, Validation Accuracy:  0.810, Loss:  0.204
    Epoch  15 Batch  124/269 - Train Accuracy:  0.817, Validation Accuracy:  0.812, Loss:  0.194
    Epoch  15 Batch  125/269 - Train Accuracy:  0.811, Validation Accuracy:  0.811, Loss:  0.199
    Epoch  15 Batch  126/269 - Train Accuracy:  0.804, Validation Accuracy:  0.818, Loss:  0.197
    Epoch  15 Batch  127/269 - Train Accuracy:  0.801, Validation Accuracy:  0.817, Loss:  0.205
    Epoch  15 Batch  128/269 - Train Accuracy:  0.815, Validation Accuracy:  0.814, Loss:  0.201
    Epoch  15 Batch  129/269 - Train Accuracy:  0.809, Validation Accuracy:  0.808, Loss:  0.195
    Epoch  15 Batch  130/269 - Train Accuracy:  0.810, Validation Accuracy:  0.812, Loss:  0.208
    Epoch  15 Batch  131/269 - Train Accuracy:  0.800, Validation Accuracy:  0.815, Loss:  0.204
    Epoch  15 Batch  132/269 - Train Accuracy:  0.799, Validation Accuracy:  0.813, Loss:  0.207
    Epoch  15 Batch  133/269 - Train Accuracy:  0.825, Validation Accuracy:  0.815, Loss:  0.195
    Epoch  15 Batch  134/269 - Train Accuracy:  0.794, Validation Accuracy:  0.815, Loss:  0.205
    Epoch  15 Batch  135/269 - Train Accuracy:  0.797, Validation Accuracy:  0.812, Loss:  0.212
    Epoch  15 Batch  136/269 - Train Accuracy:  0.798, Validation Accuracy:  0.816, Loss:  0.212
    Epoch  15 Batch  137/269 - Train Accuracy:  0.810, Validation Accuracy:  0.810, Loss:  0.215
    Epoch  15 Batch  138/269 - Train Accuracy:  0.806, Validation Accuracy:  0.815, Loss:  0.204
    Epoch  15 Batch  139/269 - Train Accuracy:  0.821, Validation Accuracy:  0.808, Loss:  0.197
    Epoch  15 Batch  140/269 - Train Accuracy:  0.821, Validation Accuracy:  0.808, Loss:  0.211
    Epoch  15 Batch  141/269 - Train Accuracy:  0.804, Validation Accuracy:  0.808, Loss:  0.211
    Epoch  15 Batch  142/269 - Train Accuracy:  0.819, Validation Accuracy:  0.812, Loss:  0.192
    Epoch  15 Batch  143/269 - Train Accuracy:  0.828, Validation Accuracy:  0.826, Loss:  0.196
    Epoch  15 Batch  144/269 - Train Accuracy:  0.840, Validation Accuracy:  0.821, Loss:  0.181
    Epoch  15 Batch  145/269 - Train Accuracy:  0.802, Validation Accuracy:  0.810, Loss:  0.193
    Epoch  15 Batch  146/269 - Train Accuracy:  0.807, Validation Accuracy:  0.820, Loss:  0.193
    Epoch  15 Batch  147/269 - Train Accuracy:  0.829, Validation Accuracy:  0.824, Loss:  0.196
    Epoch  15 Batch  148/269 - Train Accuracy:  0.822, Validation Accuracy:  0.835, Loss:  0.197
    Epoch  15 Batch  149/269 - Train Accuracy:  0.812, Validation Accuracy:  0.830, Loss:  0.202
    Epoch  15 Batch  150/269 - Train Accuracy:  0.809, Validation Accuracy:  0.827, Loss:  0.199
    Epoch  15 Batch  151/269 - Train Accuracy:  0.822, Validation Accuracy:  0.825, Loss:  0.198
    Epoch  15 Batch  152/269 - Train Accuracy:  0.814, Validation Accuracy:  0.822, Loss:  0.201
    Epoch  15 Batch  153/269 - Train Accuracy:  0.832, Validation Accuracy:  0.816, Loss:  0.196
    Epoch  15 Batch  154/269 - Train Accuracy:  0.819, Validation Accuracy:  0.823, Loss:  0.197
    Epoch  15 Batch  155/269 - Train Accuracy:  0.820, Validation Accuracy:  0.828, Loss:  0.188
    Epoch  15 Batch  156/269 - Train Accuracy:  0.817, Validation Accuracy:  0.827, Loss:  0.210
    Epoch  15 Batch  157/269 - Train Accuracy:  0.805, Validation Accuracy:  0.824, Loss:  0.195
    Epoch  15 Batch  158/269 - Train Accuracy:  0.815, Validation Accuracy:  0.820, Loss:  0.198
    Epoch  15 Batch  159/269 - Train Accuracy:  0.809, Validation Accuracy:  0.813, Loss:  0.194
    Epoch  15 Batch  160/269 - Train Accuracy:  0.822, Validation Accuracy:  0.817, Loss:  0.187
    Epoch  15 Batch  161/269 - Train Accuracy:  0.806, Validation Accuracy:  0.813, Loss:  0.192
    Epoch  15 Batch  162/269 - Train Accuracy:  0.816, Validation Accuracy:  0.821, Loss:  0.198
    Epoch  15 Batch  163/269 - Train Accuracy:  0.821, Validation Accuracy:  0.829, Loss:  0.196
    Epoch  15 Batch  164/269 - Train Accuracy:  0.829, Validation Accuracy:  0.832, Loss:  0.193
    Epoch  15 Batch  165/269 - Train Accuracy:  0.805, Validation Accuracy:  0.827, Loss:  0.199
    Epoch  15 Batch  166/269 - Train Accuracy:  0.821, Validation Accuracy:  0.823, Loss:  0.189
    Epoch  15 Batch  167/269 - Train Accuracy:  0.820, Validation Accuracy:  0.831, Loss:  0.193
    Epoch  15 Batch  168/269 - Train Accuracy:  0.819, Validation Accuracy:  0.831, Loss:  0.198
    Epoch  15 Batch  169/269 - Train Accuracy:  0.812, Validation Accuracy:  0.835, Loss:  0.196
    Epoch  15 Batch  170/269 - Train Accuracy:  0.832, Validation Accuracy:  0.840, Loss:  0.192
    Epoch  15 Batch  171/269 - Train Accuracy:  0.835, Validation Accuracy:  0.838, Loss:  0.202
    Epoch  15 Batch  172/269 - Train Accuracy:  0.824, Validation Accuracy:  0.839, Loss:  0.205
    Epoch  15 Batch  173/269 - Train Accuracy:  0.832, Validation Accuracy:  0.836, Loss:  0.190
    Epoch  15 Batch  174/269 - Train Accuracy:  0.835, Validation Accuracy:  0.833, Loss:  0.189
    Epoch  15 Batch  175/269 - Train Accuracy:  0.822, Validation Accuracy:  0.837, Loss:  0.210
    Epoch  15 Batch  176/269 - Train Accuracy:  0.822, Validation Accuracy:  0.854, Loss:  0.204
    Epoch  15 Batch  177/269 - Train Accuracy:  0.847, Validation Accuracy:  0.847, Loss:  0.185
    Epoch  15 Batch  178/269 - Train Accuracy:  0.838, Validation Accuracy:  0.841, Loss:  0.195
    Epoch  15 Batch  179/269 - Train Accuracy:  0.810, Validation Accuracy:  0.831, Loss:  0.195
    Epoch  15 Batch  180/269 - Train Accuracy:  0.833, Validation Accuracy:  0.836, Loss:  0.186
    Epoch  15 Batch  181/269 - Train Accuracy:  0.827, Validation Accuracy:  0.844, Loss:  0.199
    Epoch  15 Batch  182/269 - Train Accuracy:  0.858, Validation Accuracy:  0.842, Loss:  0.191
    Epoch  15 Batch  183/269 - Train Accuracy:  0.849, Validation Accuracy:  0.831, Loss:  0.169
    Epoch  15 Batch  184/269 - Train Accuracy:  0.822, Validation Accuracy:  0.821, Loss:  0.200
    Epoch  15 Batch  185/269 - Train Accuracy:  0.840, Validation Accuracy:  0.828, Loss:  0.185
    Epoch  15 Batch  186/269 - Train Accuracy:  0.832, Validation Accuracy:  0.839, Loss:  0.191
    Epoch  15 Batch  187/269 - Train Accuracy:  0.825, Validation Accuracy:  0.837, Loss:  0.184
    Epoch  15 Batch  188/269 - Train Accuracy:  0.825, Validation Accuracy:  0.833, Loss:  0.184
    Epoch  15 Batch  189/269 - Train Accuracy:  0.825, Validation Accuracy:  0.830, Loss:  0.184
    Epoch  15 Batch  190/269 - Train Accuracy:  0.833, Validation Accuracy:  0.830, Loss:  0.191
    Epoch  15 Batch  191/269 - Train Accuracy:  0.828, Validation Accuracy:  0.844, Loss:  0.186
    Epoch  15 Batch  192/269 - Train Accuracy:  0.844, Validation Accuracy:  0.844, Loss:  0.192
    Epoch  15 Batch  193/269 - Train Accuracy:  0.826, Validation Accuracy:  0.830, Loss:  0.184
    Epoch  15 Batch  194/269 - Train Accuracy:  0.817, Validation Accuracy:  0.817, Loss:  0.197
    Epoch  15 Batch  195/269 - Train Accuracy:  0.818, Validation Accuracy:  0.822, Loss:  0.192
    Epoch  15 Batch  196/269 - Train Accuracy:  0.826, Validation Accuracy:  0.831, Loss:  0.193
    Epoch  15 Batch  197/269 - Train Accuracy:  0.816, Validation Accuracy:  0.836, Loss:  0.200
    Epoch  15 Batch  198/269 - Train Accuracy:  0.817, Validation Accuracy:  0.824, Loss:  0.199
    Epoch  15 Batch  199/269 - Train Accuracy:  0.830, Validation Accuracy:  0.825, Loss:  0.196
    Epoch  15 Batch  200/269 - Train Accuracy:  0.829, Validation Accuracy:  0.835, Loss:  0.200
    Epoch  15 Batch  201/269 - Train Accuracy:  0.818, Validation Accuracy:  0.848, Loss:  0.192
    Epoch  15 Batch  202/269 - Train Accuracy:  0.828, Validation Accuracy:  0.855, Loss:  0.195
    Epoch  15 Batch  203/269 - Train Accuracy:  0.844, Validation Accuracy:  0.842, Loss:  0.200
    Epoch  15 Batch  204/269 - Train Accuracy:  0.815, Validation Accuracy:  0.824, Loss:  0.208
    Epoch  15 Batch  205/269 - Train Accuracy:  0.848, Validation Accuracy:  0.842, Loss:  0.194
    Epoch  15 Batch  206/269 - Train Accuracy:  0.823, Validation Accuracy:  0.843, Loss:  0.202
    Epoch  15 Batch  207/269 - Train Accuracy:  0.838, Validation Accuracy:  0.840, Loss:  0.185
    Epoch  15 Batch  208/269 - Train Accuracy:  0.836, Validation Accuracy:  0.837, Loss:  0.199
    Epoch  15 Batch  209/269 - Train Accuracy:  0.830, Validation Accuracy:  0.847, Loss:  0.190
    Epoch  15 Batch  210/269 - Train Accuracy:  0.833, Validation Accuracy:  0.845, Loss:  0.192
    Epoch  15 Batch  211/269 - Train Accuracy:  0.836, Validation Accuracy:  0.834, Loss:  0.192
    Epoch  15 Batch  212/269 - Train Accuracy:  0.818, Validation Accuracy:  0.841, Loss:  0.200
    Epoch  15 Batch  213/269 - Train Accuracy:  0.832, Validation Accuracy:  0.847, Loss:  0.196
    Epoch  15 Batch  214/269 - Train Accuracy:  0.835, Validation Accuracy:  0.842, Loss:  0.191
    Epoch  15 Batch  215/269 - Train Accuracy:  0.851, Validation Accuracy:  0.836, Loss:  0.177
    Epoch  15 Batch  216/269 - Train Accuracy:  0.808, Validation Accuracy:  0.831, Loss:  0.214
    Epoch  15 Batch  217/269 - Train Accuracy:  0.818, Validation Accuracy:  0.839, Loss:  0.199
    Epoch  15 Batch  218/269 - Train Accuracy:  0.828, Validation Accuracy:  0.843, Loss:  0.188
    Epoch  15 Batch  219/269 - Train Accuracy:  0.832, Validation Accuracy:  0.845, Loss:  0.198
    Epoch  15 Batch  220/269 - Train Accuracy:  0.813, Validation Accuracy:  0.834, Loss:  0.183
    Epoch  15 Batch  221/269 - Train Accuracy:  0.827, Validation Accuracy:  0.827, Loss:  0.192
    Epoch  15 Batch  222/269 - Train Accuracy:  0.847, Validation Accuracy:  0.834, Loss:  0.183
    Epoch  15 Batch  223/269 - Train Accuracy:  0.833, Validation Accuracy:  0.845, Loss:  0.185
    Epoch  15 Batch  224/269 - Train Accuracy:  0.844, Validation Accuracy:  0.852, Loss:  0.200
    Epoch  15 Batch  225/269 - Train Accuracy:  0.837, Validation Accuracy:  0.844, Loss:  0.188
    Epoch  15 Batch  226/269 - Train Accuracy:  0.831, Validation Accuracy:  0.838, Loss:  0.189
    Epoch  15 Batch  227/269 - Train Accuracy:  0.855, Validation Accuracy:  0.847, Loss:  0.178
    Epoch  15 Batch  228/269 - Train Accuracy:  0.843, Validation Accuracy:  0.859, Loss:  0.184
    Epoch  15 Batch  229/269 - Train Accuracy:  0.839, Validation Accuracy:  0.853, Loss:  0.180
    Epoch  15 Batch  230/269 - Train Accuracy:  0.831, Validation Accuracy:  0.849, Loss:  0.190
    Epoch  15 Batch  231/269 - Train Accuracy:  0.808, Validation Accuracy:  0.847, Loss:  0.194
    Epoch  15 Batch  232/269 - Train Accuracy:  0.829, Validation Accuracy:  0.842, Loss:  0.201
    Epoch  15 Batch  233/269 - Train Accuracy:  0.856, Validation Accuracy:  0.840, Loss:  0.189
    Epoch  15 Batch  234/269 - Train Accuracy:  0.852, Validation Accuracy:  0.850, Loss:  0.190
    Epoch  15 Batch  235/269 - Train Accuracy:  0.857, Validation Accuracy:  0.845, Loss:  0.179
    Epoch  15 Batch  236/269 - Train Accuracy:  0.820, Validation Accuracy:  0.830, Loss:  0.187
    Epoch  15 Batch  237/269 - Train Accuracy:  0.845, Validation Accuracy:  0.842, Loss:  0.185
    Epoch  15 Batch  238/269 - Train Accuracy:  0.863, Validation Accuracy:  0.850, Loss:  0.180
    Epoch  15 Batch  239/269 - Train Accuracy:  0.861, Validation Accuracy:  0.848, Loss:  0.187
    Epoch  15 Batch  240/269 - Train Accuracy:  0.860, Validation Accuracy:  0.840, Loss:  0.166
    Epoch  15 Batch  241/269 - Train Accuracy:  0.835, Validation Accuracy:  0.845, Loss:  0.191
    Epoch  15 Batch  242/269 - Train Accuracy:  0.856, Validation Accuracy:  0.845, Loss:  0.186
    Epoch  15 Batch  243/269 - Train Accuracy:  0.855, Validation Accuracy:  0.845, Loss:  0.177
    Epoch  15 Batch  244/269 - Train Accuracy:  0.844, Validation Accuracy:  0.835, Loss:  0.194
    Epoch  15 Batch  245/269 - Train Accuracy:  0.820, Validation Accuracy:  0.834, Loss:  0.192
    Epoch  15 Batch  246/269 - Train Accuracy:  0.831, Validation Accuracy:  0.839, Loss:  0.188
    Epoch  15 Batch  247/269 - Train Accuracy:  0.840, Validation Accuracy:  0.841, Loss:  0.187
    Epoch  15 Batch  248/269 - Train Accuracy:  0.834, Validation Accuracy:  0.837, Loss:  0.183
    Epoch  15 Batch  249/269 - Train Accuracy:  0.863, Validation Accuracy:  0.839, Loss:  0.170
    Epoch  15 Batch  250/269 - Train Accuracy:  0.838, Validation Accuracy:  0.842, Loss:  0.183
    Epoch  15 Batch  251/269 - Train Accuracy:  0.860, Validation Accuracy:  0.852, Loss:  0.176
    Epoch  15 Batch  252/269 - Train Accuracy:  0.846, Validation Accuracy:  0.855, Loss:  0.184
    Epoch  15 Batch  253/269 - Train Accuracy:  0.841, Validation Accuracy:  0.851, Loss:  0.197
    Epoch  15 Batch  254/269 - Train Accuracy:  0.852, Validation Accuracy:  0.856, Loss:  0.185
    Epoch  15 Batch  255/269 - Train Accuracy:  0.864, Validation Accuracy:  0.850, Loss:  0.172
    Epoch  15 Batch  256/269 - Train Accuracy:  0.823, Validation Accuracy:  0.853, Loss:  0.190
    Epoch  15 Batch  257/269 - Train Accuracy:  0.827, Validation Accuracy:  0.860, Loss:  0.191
    Epoch  15 Batch  258/269 - Train Accuracy:  0.849, Validation Accuracy:  0.866, Loss:  0.193
    Epoch  15 Batch  259/269 - Train Accuracy:  0.858, Validation Accuracy:  0.858, Loss:  0.186
    Epoch  15 Batch  260/269 - Train Accuracy:  0.839, Validation Accuracy:  0.857, Loss:  0.196
    Epoch  15 Batch  261/269 - Train Accuracy:  0.826, Validation Accuracy:  0.846, Loss:  0.197
    Epoch  15 Batch  262/269 - Train Accuracy:  0.853, Validation Accuracy:  0.850, Loss:  0.179
    Epoch  15 Batch  263/269 - Train Accuracy:  0.844, Validation Accuracy:  0.849, Loss:  0.195
    Epoch  15 Batch  264/269 - Train Accuracy:  0.829, Validation Accuracy:  0.854, Loss:  0.199
    Epoch  15 Batch  265/269 - Train Accuracy:  0.834, Validation Accuracy:  0.851, Loss:  0.187
    Epoch  15 Batch  266/269 - Train Accuracy:  0.858, Validation Accuracy:  0.851, Loss:  0.183
    Epoch  15 Batch  267/269 - Train Accuracy:  0.849, Validation Accuracy:  0.845, Loss:  0.191
    Epoch  16 Batch    0/269 - Train Accuracy:  0.840, Validation Accuracy:  0.847, Loss:  0.201
    Epoch  16 Batch    1/269 - Train Accuracy:  0.829, Validation Accuracy:  0.840, Loss:  0.184
    Epoch  16 Batch    2/269 - Train Accuracy:  0.839, Validation Accuracy:  0.839, Loss:  0.197
    Epoch  16 Batch    3/269 - Train Accuracy:  0.839, Validation Accuracy:  0.843, Loss:  0.184
    Epoch  16 Batch    4/269 - Train Accuracy:  0.837, Validation Accuracy:  0.861, Loss:  0.199
    Epoch  16 Batch    5/269 - Train Accuracy:  0.839, Validation Accuracy:  0.865, Loss:  0.191
    Epoch  16 Batch    6/269 - Train Accuracy:  0.847, Validation Accuracy:  0.859, Loss:  0.185
    Epoch  16 Batch    7/269 - Train Accuracy:  0.859, Validation Accuracy:  0.856, Loss:  0.178
    Epoch  16 Batch    8/269 - Train Accuracy:  0.845, Validation Accuracy:  0.852, Loss:  0.197
    Epoch  16 Batch    9/269 - Train Accuracy:  0.850, Validation Accuracy:  0.854, Loss:  0.194
    Epoch  16 Batch   10/269 - Train Accuracy:  0.852, Validation Accuracy:  0.851, Loss:  0.183
    Epoch  16 Batch   11/269 - Train Accuracy:  0.837, Validation Accuracy:  0.851, Loss:  0.194
    Epoch  16 Batch   12/269 - Train Accuracy:  0.838, Validation Accuracy:  0.850, Loss:  0.196
    Epoch  16 Batch   13/269 - Train Accuracy:  0.847, Validation Accuracy:  0.851, Loss:  0.170
    Epoch  16 Batch   14/269 - Train Accuracy:  0.838, Validation Accuracy:  0.850, Loss:  0.190
    Epoch  16 Batch   15/269 - Train Accuracy:  0.840, Validation Accuracy:  0.854, Loss:  0.171
    Epoch  16 Batch   16/269 - Train Accuracy:  0.855, Validation Accuracy:  0.855, Loss:  0.181
    Epoch  16 Batch   17/269 - Train Accuracy:  0.846, Validation Accuracy:  0.858, Loss:  0.178
    Epoch  16 Batch   18/269 - Train Accuracy:  0.839, Validation Accuracy:  0.853, Loss:  0.190
    Epoch  16 Batch   19/269 - Train Accuracy:  0.848, Validation Accuracy:  0.844, Loss:  0.168
    Epoch  16 Batch   20/269 - Train Accuracy:  0.857, Validation Accuracy:  0.850, Loss:  0.188
    Epoch  16 Batch   21/269 - Train Accuracy:  0.833, Validation Accuracy:  0.854, Loss:  0.202
    Epoch  16 Batch   22/269 - Train Accuracy:  0.870, Validation Accuracy:  0.857, Loss:  0.172
    Epoch  16 Batch   23/269 - Train Accuracy:  0.855, Validation Accuracy:  0.858, Loss:  0.180
    Epoch  16 Batch   24/269 - Train Accuracy:  0.846, Validation Accuracy:  0.858, Loss:  0.184
    Epoch  16 Batch   25/269 - Train Accuracy:  0.841, Validation Accuracy:  0.850, Loss:  0.198
    Epoch  16 Batch   26/269 - Train Accuracy:  0.852, Validation Accuracy:  0.847, Loss:  0.168
    Epoch  16 Batch   27/269 - Train Accuracy:  0.839, Validation Accuracy:  0.847, Loss:  0.177
    Epoch  16 Batch   28/269 - Train Accuracy:  0.830, Validation Accuracy:  0.855, Loss:  0.199
    Epoch  16 Batch   29/269 - Train Accuracy:  0.851, Validation Accuracy:  0.855, Loss:  0.190
    Epoch  16 Batch   30/269 - Train Accuracy:  0.837, Validation Accuracy:  0.853, Loss:  0.178
    Epoch  16 Batch   31/269 - Train Accuracy:  0.860, Validation Accuracy:  0.857, Loss:  0.171
    Epoch  16 Batch   32/269 - Train Accuracy:  0.836, Validation Accuracy:  0.860, Loss:  0.172
    Epoch  16 Batch   33/269 - Train Accuracy:  0.859, Validation Accuracy:  0.857, Loss:  0.172
    Epoch  16 Batch   34/269 - Train Accuracy:  0.853, Validation Accuracy:  0.861, Loss:  0.175
    Epoch  16 Batch   35/269 - Train Accuracy:  0.843, Validation Accuracy:  0.864, Loss:  0.192
    Epoch  16 Batch   36/269 - Train Accuracy:  0.840, Validation Accuracy:  0.862, Loss:  0.179
    Epoch  16 Batch   37/269 - Train Accuracy:  0.864, Validation Accuracy:  0.858, Loss:  0.178
    Epoch  16 Batch   38/269 - Train Accuracy:  0.853, Validation Accuracy:  0.849, Loss:  0.175
    Epoch  16 Batch   39/269 - Train Accuracy:  0.846, Validation Accuracy:  0.850, Loss:  0.186
    Epoch  16 Batch   40/269 - Train Accuracy:  0.839, Validation Accuracy:  0.852, Loss:  0.190
    Epoch  16 Batch   41/269 - Train Accuracy:  0.844, Validation Accuracy:  0.851, Loss:  0.184
    Epoch  16 Batch   42/269 - Train Accuracy:  0.852, Validation Accuracy:  0.841, Loss:  0.168
    Epoch  16 Batch   43/269 - Train Accuracy:  0.857, Validation Accuracy:  0.846, Loss:  0.179
    Epoch  16 Batch   44/269 - Train Accuracy:  0.856, Validation Accuracy:  0.857, Loss:  0.181
    Epoch  16 Batch   45/269 - Train Accuracy:  0.854, Validation Accuracy:  0.856, Loss:  0.184
    Epoch  16 Batch   46/269 - Train Accuracy:  0.855, Validation Accuracy:  0.863, Loss:  0.182
    Epoch  16 Batch   47/269 - Train Accuracy:  0.863, Validation Accuracy:  0.865, Loss:  0.168
    Epoch  16 Batch   48/269 - Train Accuracy:  0.853, Validation Accuracy:  0.864, Loss:  0.173
    Epoch  16 Batch   49/269 - Train Accuracy:  0.851, Validation Accuracy:  0.861, Loss:  0.178
    Epoch  16 Batch   50/269 - Train Accuracy:  0.835, Validation Accuracy:  0.863, Loss:  0.193
    Epoch  16 Batch   51/269 - Train Accuracy:  0.841, Validation Accuracy:  0.869, Loss:  0.177
    Epoch  16 Batch   52/269 - Train Accuracy:  0.847, Validation Accuracy:  0.856, Loss:  0.170
    Epoch  16 Batch   53/269 - Train Accuracy:  0.839, Validation Accuracy:  0.857, Loss:  0.190
    Epoch  16 Batch   54/269 - Train Accuracy:  0.859, Validation Accuracy:  0.868, Loss:  0.173
    Epoch  16 Batch   55/269 - Train Accuracy:  0.860, Validation Accuracy:  0.870, Loss:  0.179
    Epoch  16 Batch   56/269 - Train Accuracy:  0.849, Validation Accuracy:  0.866, Loss:  0.178
    Epoch  16 Batch   57/269 - Train Accuracy:  0.855, Validation Accuracy:  0.861, Loss:  0.184
    Epoch  16 Batch   58/269 - Train Accuracy:  0.858, Validation Accuracy:  0.865, Loss:  0.173
    Epoch  16 Batch   59/269 - Train Accuracy:  0.880, Validation Accuracy:  0.865, Loss:  0.158
    Epoch  16 Batch   60/269 - Train Accuracy:  0.868, Validation Accuracy:  0.858, Loss:  0.170
    Epoch  16 Batch   61/269 - Train Accuracy:  0.867, Validation Accuracy:  0.854, Loss:  0.164
    Epoch  16 Batch   62/269 - Train Accuracy:  0.858, Validation Accuracy:  0.845, Loss:  0.168
    Epoch  16 Batch   63/269 - Train Accuracy:  0.856, Validation Accuracy:  0.852, Loss:  0.181
    Epoch  16 Batch   64/269 - Train Accuracy:  0.856, Validation Accuracy:  0.852, Loss:  0.166
    Epoch  16 Batch   65/269 - Train Accuracy:  0.827, Validation Accuracy:  0.843, Loss:  0.177
    Epoch  16 Batch   66/269 - Train Accuracy:  0.848, Validation Accuracy:  0.857, Loss:  0.176
    Epoch  16 Batch   67/269 - Train Accuracy:  0.841, Validation Accuracy:  0.849, Loss:  0.182
    Epoch  16 Batch   68/269 - Train Accuracy:  0.837, Validation Accuracy:  0.852, Loss:  0.186
    Epoch  16 Batch   69/269 - Train Accuracy:  0.821, Validation Accuracy:  0.849, Loss:  0.202
    Epoch  16 Batch   70/269 - Train Accuracy:  0.859, Validation Accuracy:  0.856, Loss:  0.173
    Epoch  16 Batch   71/269 - Train Accuracy:  0.855, Validation Accuracy:  0.854, Loss:  0.192
    Epoch  16 Batch   72/269 - Train Accuracy:  0.847, Validation Accuracy:  0.854, Loss:  0.184
    Epoch  16 Batch   73/269 - Train Accuracy:  0.866, Validation Accuracy:  0.857, Loss:  0.180
    Epoch  16 Batch   74/269 - Train Accuracy:  0.849, Validation Accuracy:  0.853, Loss:  0.173
    Epoch  16 Batch   75/269 - Train Accuracy:  0.833, Validation Accuracy:  0.852, Loss:  0.174
    Epoch  16 Batch   76/269 - Train Accuracy:  0.846, Validation Accuracy:  0.862, Loss:  0.184
    Epoch  16 Batch   77/269 - Train Accuracy:  0.860, Validation Accuracy:  0.849, Loss:  0.175
    Epoch  16 Batch   78/269 - Train Accuracy:  0.860, Validation Accuracy:  0.861, Loss:  0.174
    Epoch  16 Batch   79/269 - Train Accuracy:  0.828, Validation Accuracy:  0.851, Loss:  0.177
    Epoch  16 Batch   80/269 - Train Accuracy:  0.868, Validation Accuracy:  0.844, Loss:  0.175
    Epoch  16 Batch   81/269 - Train Accuracy:  0.845, Validation Accuracy:  0.848, Loss:  0.178
    Epoch  16 Batch   82/269 - Train Accuracy:  0.857, Validation Accuracy:  0.847, Loss:  0.160
    Epoch  16 Batch   83/269 - Train Accuracy:  0.845, Validation Accuracy:  0.844, Loss:  0.189
    Epoch  16 Batch   84/269 - Train Accuracy:  0.854, Validation Accuracy:  0.847, Loss:  0.174
    Epoch  16 Batch   85/269 - Train Accuracy:  0.851, Validation Accuracy:  0.852, Loss:  0.172
    Epoch  16 Batch   86/269 - Train Accuracy:  0.864, Validation Accuracy:  0.847, Loss:  0.167
    Epoch  16 Batch   87/269 - Train Accuracy:  0.839, Validation Accuracy:  0.852, Loss:  0.186
    Epoch  16 Batch   88/269 - Train Accuracy:  0.852, Validation Accuracy:  0.859, Loss:  0.176
    Epoch  16 Batch   89/269 - Train Accuracy:  0.872, Validation Accuracy:  0.859, Loss:  0.176
    Epoch  16 Batch   90/269 - Train Accuracy:  0.842, Validation Accuracy:  0.856, Loss:  0.185
    Epoch  16 Batch   91/269 - Train Accuracy:  0.862, Validation Accuracy:  0.857, Loss:  0.168
    Epoch  16 Batch   92/269 - Train Accuracy:  0.868, Validation Accuracy:  0.857, Loss:  0.172
    Epoch  16 Batch   93/269 - Train Accuracy:  0.860, Validation Accuracy:  0.854, Loss:  0.165
    Epoch  16 Batch   94/269 - Train Accuracy:  0.852, Validation Accuracy:  0.852, Loss:  0.187
    Epoch  16 Batch   95/269 - Train Accuracy:  0.855, Validation Accuracy:  0.849, Loss:  0.180
    Epoch  16 Batch   96/269 - Train Accuracy:  0.841, Validation Accuracy:  0.849, Loss:  0.184
    Epoch  16 Batch   97/269 - Train Accuracy:  0.869, Validation Accuracy:  0.851, Loss:  0.173
    Epoch  16 Batch   98/269 - Train Accuracy:  0.856, Validation Accuracy:  0.850, Loss:  0.175
    Epoch  16 Batch   99/269 - Train Accuracy:  0.855, Validation Accuracy:  0.854, Loss:  0.176
    Epoch  16 Batch  100/269 - Train Accuracy:  0.866, Validation Accuracy:  0.856, Loss:  0.171
    Epoch  16 Batch  101/269 - Train Accuracy:  0.841, Validation Accuracy:  0.852, Loss:  0.189
    Epoch  16 Batch  102/269 - Train Accuracy:  0.849, Validation Accuracy:  0.847, Loss:  0.168
    Epoch  16 Batch  103/269 - Train Accuracy:  0.859, Validation Accuracy:  0.850, Loss:  0.175
    Epoch  16 Batch  104/269 - Train Accuracy:  0.849, Validation Accuracy:  0.849, Loss:  0.169
    Epoch  16 Batch  105/269 - Train Accuracy:  0.844, Validation Accuracy:  0.855, Loss:  0.177
    Epoch  16 Batch  106/269 - Train Accuracy:  0.866, Validation Accuracy:  0.855, Loss:  0.166
    Epoch  16 Batch  107/269 - Train Accuracy:  0.844, Validation Accuracy:  0.859, Loss:  0.185
    Epoch  16 Batch  108/269 - Train Accuracy:  0.868, Validation Accuracy:  0.867, Loss:  0.172
    Epoch  16 Batch  109/269 - Train Accuracy:  0.834, Validation Accuracy:  0.866, Loss:  0.178
    Epoch  16 Batch  110/269 - Train Accuracy:  0.858, Validation Accuracy:  0.860, Loss:  0.165
    Epoch  16 Batch  111/269 - Train Accuracy:  0.854, Validation Accuracy:  0.856, Loss:  0.190
    Epoch  16 Batch  112/269 - Train Accuracy:  0.850, Validation Accuracy:  0.860, Loss:  0.177
    Epoch  16 Batch  113/269 - Train Accuracy:  0.855, Validation Accuracy:  0.852, Loss:  0.168
    Epoch  16 Batch  114/269 - Train Accuracy:  0.864, Validation Accuracy:  0.853, Loss:  0.177
    Epoch  16 Batch  115/269 - Train Accuracy:  0.839, Validation Accuracy:  0.847, Loss:  0.183
    Epoch  16 Batch  116/269 - Train Accuracy:  0.867, Validation Accuracy:  0.850, Loss:  0.174
    Epoch  16 Batch  117/269 - Train Accuracy:  0.854, Validation Accuracy:  0.864, Loss:  0.169
    Epoch  16 Batch  118/269 - Train Accuracy:  0.872, Validation Accuracy:  0.857, Loss:  0.162
    Epoch  16 Batch  119/269 - Train Accuracy:  0.845, Validation Accuracy:  0.866, Loss:  0.180
    Epoch  16 Batch  120/269 - Train Accuracy:  0.849, Validation Accuracy:  0.848, Loss:  0.173
    Epoch  16 Batch  121/269 - Train Accuracy:  0.849, Validation Accuracy:  0.853, Loss:  0.167
    Epoch  16 Batch  122/269 - Train Accuracy:  0.855, Validation Accuracy:  0.865, Loss:  0.173
    Epoch  16 Batch  123/269 - Train Accuracy:  0.849, Validation Accuracy:  0.858, Loss:  0.174
    Epoch  16 Batch  124/269 - Train Accuracy:  0.858, Validation Accuracy:  0.863, Loss:  0.165
    Epoch  16 Batch  125/269 - Train Accuracy:  0.859, Validation Accuracy:  0.868, Loss:  0.172
    Epoch  16 Batch  126/269 - Train Accuracy:  0.843, Validation Accuracy:  0.864, Loss:  0.171
    Epoch  16 Batch  127/269 - Train Accuracy:  0.853, Validation Accuracy:  0.858, Loss:  0.172
    Epoch  16 Batch  128/269 - Train Accuracy:  0.847, Validation Accuracy:  0.847, Loss:  0.168
    Epoch  16 Batch  129/269 - Train Accuracy:  0.846, Validation Accuracy:  0.854, Loss:  0.167
    Epoch  16 Batch  130/269 - Train Accuracy:  0.853, Validation Accuracy:  0.866, Loss:  0.179
    Epoch  16 Batch  131/269 - Train Accuracy:  0.824, Validation Accuracy:  0.865, Loss:  0.173
    Epoch  16 Batch  132/269 - Train Accuracy:  0.851, Validation Accuracy:  0.867, Loss:  0.180
    Epoch  16 Batch  133/269 - Train Accuracy:  0.860, Validation Accuracy:  0.873, Loss:  0.164
    Epoch  16 Batch  134/269 - Train Accuracy:  0.854, Validation Accuracy:  0.868, Loss:  0.172
    Epoch  16 Batch  135/269 - Train Accuracy:  0.853, Validation Accuracy:  0.864, Loss:  0.185
    Epoch  16 Batch  136/269 - Train Accuracy:  0.835, Validation Accuracy:  0.861, Loss:  0.181
    Epoch  16 Batch  137/269 - Train Accuracy:  0.839, Validation Accuracy:  0.861, Loss:  0.184
    Epoch  16 Batch  138/269 - Train Accuracy:  0.850, Validation Accuracy:  0.862, Loss:  0.177
    Epoch  16 Batch  139/269 - Train Accuracy:  0.861, Validation Accuracy:  0.862, Loss:  0.161
    Epoch  16 Batch  140/269 - Train Accuracy:  0.857, Validation Accuracy:  0.859, Loss:  0.177
    Epoch  16 Batch  141/269 - Train Accuracy:  0.850, Validation Accuracy:  0.864, Loss:  0.179
    Epoch  16 Batch  142/269 - Train Accuracy:  0.855, Validation Accuracy:  0.869, Loss:  0.165
    Epoch  16 Batch  143/269 - Train Accuracy:  0.866, Validation Accuracy:  0.869, Loss:  0.164
    Epoch  16 Batch  144/269 - Train Accuracy:  0.882, Validation Accuracy:  0.868, Loss:  0.152
    Epoch  16 Batch  145/269 - Train Accuracy:  0.862, Validation Accuracy:  0.864, Loss:  0.167
    Epoch  16 Batch  146/269 - Train Accuracy:  0.861, Validation Accuracy:  0.872, Loss:  0.158
    Epoch  16 Batch  147/269 - Train Accuracy:  0.865, Validation Accuracy:  0.876, Loss:  0.167
    Epoch  16 Batch  148/269 - Train Accuracy:  0.852, Validation Accuracy:  0.874, Loss:  0.166
    Epoch  16 Batch  149/269 - Train Accuracy:  0.859, Validation Accuracy:  0.869, Loss:  0.176
    Epoch  16 Batch  150/269 - Train Accuracy:  0.852, Validation Accuracy:  0.868, Loss:  0.165
    Epoch  16 Batch  151/269 - Train Accuracy:  0.865, Validation Accuracy:  0.867, Loss:  0.169
    Epoch  16 Batch  152/269 - Train Accuracy:  0.850, Validation Accuracy:  0.862, Loss:  0.173
    Epoch  16 Batch  153/269 - Train Accuracy:  0.873, Validation Accuracy:  0.865, Loss:  0.165
    Epoch  16 Batch  154/269 - Train Accuracy:  0.861, Validation Accuracy:  0.858, Loss:  0.166
    Epoch  16 Batch  155/269 - Train Accuracy:  0.857, Validation Accuracy:  0.862, Loss:  0.160
    Epoch  16 Batch  156/269 - Train Accuracy:  0.853, Validation Accuracy:  0.861, Loss:  0.173
    Epoch  16 Batch  157/269 - Train Accuracy:  0.846, Validation Accuracy:  0.852, Loss:  0.167
    Epoch  16 Batch  158/269 - Train Accuracy:  0.867, Validation Accuracy:  0.857, Loss:  0.169
    Epoch  16 Batch  159/269 - Train Accuracy:  0.847, Validation Accuracy:  0.859, Loss:  0.168
    Epoch  16 Batch  160/269 - Train Accuracy:  0.862, Validation Accuracy:  0.858, Loss:  0.161
    Epoch  16 Batch  161/269 - Train Accuracy:  0.852, Validation Accuracy:  0.855, Loss:  0.167
    Epoch  16 Batch  162/269 - Train Accuracy:  0.862, Validation Accuracy:  0.866, Loss:  0.174
    Epoch  16 Batch  163/269 - Train Accuracy:  0.868, Validation Accuracy:  0.869, Loss:  0.171
    Epoch  16 Batch  164/269 - Train Accuracy:  0.871, Validation Accuracy:  0.870, Loss:  0.162
    Epoch  16 Batch  165/269 - Train Accuracy:  0.857, Validation Accuracy:  0.870, Loss:  0.174
    Epoch  16 Batch  166/269 - Train Accuracy:  0.877, Validation Accuracy:  0.870, Loss:  0.159
    Epoch  16 Batch  167/269 - Train Accuracy:  0.870, Validation Accuracy:  0.868, Loss:  0.165
    Epoch  16 Batch  168/269 - Train Accuracy:  0.867, Validation Accuracy:  0.861, Loss:  0.170
    Epoch  16 Batch  169/269 - Train Accuracy:  0.842, Validation Accuracy:  0.868, Loss:  0.169
    Epoch  16 Batch  170/269 - Train Accuracy:  0.858, Validation Accuracy:  0.865, Loss:  0.162
    Epoch  16 Batch  171/269 - Train Accuracy:  0.872, Validation Accuracy:  0.864, Loss:  0.172
    Epoch  16 Batch  172/269 - Train Accuracy:  0.851, Validation Accuracy:  0.859, Loss:  0.177
    Epoch  16 Batch  173/269 - Train Accuracy:  0.860, Validation Accuracy:  0.856, Loss:  0.160
    Epoch  16 Batch  174/269 - Train Accuracy:  0.870, Validation Accuracy:  0.865, Loss:  0.159
    Epoch  16 Batch  175/269 - Train Accuracy:  0.848, Validation Accuracy:  0.861, Loss:  0.181
    Epoch  16 Batch  176/269 - Train Accuracy:  0.843, Validation Accuracy:  0.864, Loss:  0.177
    Epoch  16 Batch  177/269 - Train Accuracy:  0.860, Validation Accuracy:  0.858, Loss:  0.155
    Epoch  16 Batch  178/269 - Train Accuracy:  0.863, Validation Accuracy:  0.869, Loss:  0.164
    Epoch  16 Batch  179/269 - Train Accuracy:  0.857, Validation Accuracy:  0.867, Loss:  0.161
    Epoch  16 Batch  180/269 - Train Accuracy:  0.866, Validation Accuracy:  0.867, Loss:  0.162
    Epoch  16 Batch  181/269 - Train Accuracy:  0.854, Validation Accuracy:  0.871, Loss:  0.174
    Epoch  16 Batch  182/269 - Train Accuracy:  0.858, Validation Accuracy:  0.859, Loss:  0.164
    Epoch  16 Batch  183/269 - Train Accuracy:  0.883, Validation Accuracy:  0.867, Loss:  0.145
    Epoch  16 Batch  184/269 - Train Accuracy:  0.855, Validation Accuracy:  0.867, Loss:  0.169
    Epoch  16 Batch  185/269 - Train Accuracy:  0.868, Validation Accuracy:  0.861, Loss:  0.160
    Epoch  16 Batch  186/269 - Train Accuracy:  0.865, Validation Accuracy:  0.852, Loss:  0.162
    Epoch  16 Batch  187/269 - Train Accuracy:  0.861, Validation Accuracy:  0.850, Loss:  0.161
    Epoch  16 Batch  188/269 - Train Accuracy:  0.862, Validation Accuracy:  0.859, Loss:  0.158
    Epoch  16 Batch  189/269 - Train Accuracy:  0.862, Validation Accuracy:  0.859, Loss:  0.159
    Epoch  16 Batch  190/269 - Train Accuracy:  0.873, Validation Accuracy:  0.868, Loss:  0.164
    Epoch  16 Batch  191/269 - Train Accuracy:  0.847, Validation Accuracy:  0.863, Loss:  0.162
    Epoch  16 Batch  192/269 - Train Accuracy:  0.873, Validation Accuracy:  0.866, Loss:  0.164
    Epoch  16 Batch  193/269 - Train Accuracy:  0.858, Validation Accuracy:  0.869, Loss:  0.157
    Epoch  16 Batch  194/269 - Train Accuracy:  0.855, Validation Accuracy:  0.862, Loss:  0.164
    Epoch  16 Batch  195/269 - Train Accuracy:  0.857, Validation Accuracy:  0.867, Loss:  0.164
    Epoch  16 Batch  196/269 - Train Accuracy:  0.850, Validation Accuracy:  0.864, Loss:  0.162
    Epoch  16 Batch  197/269 - Train Accuracy:  0.841, Validation Accuracy:  0.867, Loss:  0.170
    Epoch  16 Batch  198/269 - Train Accuracy:  0.858, Validation Accuracy:  0.864, Loss:  0.171
    Epoch  16 Batch  199/269 - Train Accuracy:  0.853, Validation Accuracy:  0.866, Loss:  0.163
    Epoch  16 Batch  200/269 - Train Accuracy:  0.863, Validation Accuracy:  0.871, Loss:  0.171
    Epoch  16 Batch  201/269 - Train Accuracy:  0.848, Validation Accuracy:  0.873, Loss:  0.167
    Epoch  16 Batch  202/269 - Train Accuracy:  0.871, Validation Accuracy:  0.870, Loss:  0.166
    Epoch  16 Batch  203/269 - Train Accuracy:  0.871, Validation Accuracy:  0.867, Loss:  0.175
    Epoch  16 Batch  204/269 - Train Accuracy:  0.860, Validation Accuracy:  0.868, Loss:  0.176
    Epoch  16 Batch  205/269 - Train Accuracy:  0.875, Validation Accuracy:  0.861, Loss:  0.163
    Epoch  16 Batch  206/269 - Train Accuracy:  0.846, Validation Accuracy:  0.862, Loss:  0.174
    Epoch  16 Batch  207/269 - Train Accuracy:  0.863, Validation Accuracy:  0.861, Loss:  0.157
    Epoch  16 Batch  208/269 - Train Accuracy:  0.856, Validation Accuracy:  0.870, Loss:  0.172
    Epoch  16 Batch  209/269 - Train Accuracy:  0.874, Validation Accuracy:  0.865, Loss:  0.168
    Epoch  16 Batch  210/269 - Train Accuracy:  0.861, Validation Accuracy:  0.866, Loss:  0.162
    Epoch  16 Batch  211/269 - Train Accuracy:  0.865, Validation Accuracy:  0.873, Loss:  0.170
    Epoch  16 Batch  212/269 - Train Accuracy:  0.869, Validation Accuracy:  0.868, Loss:  0.168
    Epoch  16 Batch  213/269 - Train Accuracy:  0.867, Validation Accuracy:  0.870, Loss:  0.169
    Epoch  16 Batch  214/269 - Train Accuracy:  0.865, Validation Accuracy:  0.875, Loss:  0.163
    Epoch  16 Batch  215/269 - Train Accuracy:  0.881, Validation Accuracy:  0.874, Loss:  0.152
    Epoch  16 Batch  216/269 - Train Accuracy:  0.850, Validation Accuracy:  0.867, Loss:  0.192
    Epoch  16 Batch  217/269 - Train Accuracy:  0.850, Validation Accuracy:  0.859, Loss:  0.170
    Epoch  16 Batch  218/269 - Train Accuracy:  0.861, Validation Accuracy:  0.859, Loss:  0.167
    Epoch  16 Batch  219/269 - Train Accuracy:  0.858, Validation Accuracy:  0.863, Loss:  0.169
    Epoch  16 Batch  220/269 - Train Accuracy:  0.859, Validation Accuracy:  0.861, Loss:  0.155
    Epoch  16 Batch  221/269 - Train Accuracy:  0.863, Validation Accuracy:  0.866, Loss:  0.165
    Epoch  16 Batch  222/269 - Train Accuracy:  0.876, Validation Accuracy:  0.865, Loss:  0.156
    Epoch  16 Batch  223/269 - Train Accuracy:  0.864, Validation Accuracy:  0.871, Loss:  0.162
    Epoch  16 Batch  224/269 - Train Accuracy:  0.866, Validation Accuracy:  0.874, Loss:  0.171
    Epoch  16 Batch  225/269 - Train Accuracy:  0.862, Validation Accuracy:  0.877, Loss:  0.161
    Epoch  16 Batch  226/269 - Train Accuracy:  0.875, Validation Accuracy:  0.874, Loss:  0.164
    Epoch  16 Batch  227/269 - Train Accuracy:  0.886, Validation Accuracy:  0.875, Loss:  0.154
    Epoch  16 Batch  228/269 - Train Accuracy:  0.859, Validation Accuracy:  0.873, Loss:  0.157
    Epoch  16 Batch  229/269 - Train Accuracy:  0.853, Validation Accuracy:  0.874, Loss:  0.155
    Epoch  16 Batch  230/269 - Train Accuracy:  0.870, Validation Accuracy:  0.876, Loss:  0.161
    Epoch  16 Batch  231/269 - Train Accuracy:  0.852, Validation Accuracy:  0.875, Loss:  0.164
    Epoch  16 Batch  232/269 - Train Accuracy:  0.855, Validation Accuracy:  0.871, Loss:  0.165
    Epoch  16 Batch  233/269 - Train Accuracy:  0.874, Validation Accuracy:  0.858, Loss:  0.170
    Epoch  16 Batch  234/269 - Train Accuracy:  0.875, Validation Accuracy:  0.870, Loss:  0.162
    Epoch  16 Batch  235/269 - Train Accuracy:  0.884, Validation Accuracy:  0.864, Loss:  0.150
    Epoch  16 Batch  236/269 - Train Accuracy:  0.857, Validation Accuracy:  0.867, Loss:  0.159
    Epoch  16 Batch  237/269 - Train Accuracy:  0.878, Validation Accuracy:  0.874, Loss:  0.156
    Epoch  16 Batch  238/269 - Train Accuracy:  0.875, Validation Accuracy:  0.871, Loss:  0.152
    Epoch  16 Batch  239/269 - Train Accuracy:  0.873, Validation Accuracy:  0.869, Loss:  0.158
    Epoch  16 Batch  240/269 - Train Accuracy:  0.888, Validation Accuracy:  0.870, Loss:  0.142
    Epoch  16 Batch  241/269 - Train Accuracy:  0.864, Validation Accuracy:  0.870, Loss:  0.163
    Epoch  16 Batch  242/269 - Train Accuracy:  0.873, Validation Accuracy:  0.871, Loss:  0.159
    Epoch  16 Batch  243/269 - Train Accuracy:  0.884, Validation Accuracy:  0.870, Loss:  0.147
    Epoch  16 Batch  244/269 - Train Accuracy:  0.863, Validation Accuracy:  0.865, Loss:  0.162
    Epoch  16 Batch  245/269 - Train Accuracy:  0.846, Validation Accuracy:  0.864, Loss:  0.163
    Epoch  16 Batch  246/269 - Train Accuracy:  0.851, Validation Accuracy:  0.864, Loss:  0.157
    Epoch  16 Batch  247/269 - Train Accuracy:  0.875, Validation Accuracy:  0.867, Loss:  0.158
    Epoch  16 Batch  248/269 - Train Accuracy:  0.873, Validation Accuracy:  0.870, Loss:  0.155
    Epoch  16 Batch  249/269 - Train Accuracy:  0.890, Validation Accuracy:  0.875, Loss:  0.143
    Epoch  16 Batch  250/269 - Train Accuracy:  0.881, Validation Accuracy:  0.885, Loss:  0.160
    Epoch  16 Batch  251/269 - Train Accuracy:  0.877, Validation Accuracy:  0.883, Loss:  0.147
    Epoch  16 Batch  252/269 - Train Accuracy:  0.874, Validation Accuracy:  0.878, Loss:  0.159
    Epoch  16 Batch  253/269 - Train Accuracy:  0.857, Validation Accuracy:  0.878, Loss:  0.170
    Epoch  16 Batch  254/269 - Train Accuracy:  0.871, Validation Accuracy:  0.875, Loss:  0.155
    Epoch  16 Batch  255/269 - Train Accuracy:  0.885, Validation Accuracy:  0.872, Loss:  0.148
    Epoch  16 Batch  256/269 - Train Accuracy:  0.844, Validation Accuracy:  0.877, Loss:  0.158
    Epoch  16 Batch  257/269 - Train Accuracy:  0.855, Validation Accuracy:  0.877, Loss:  0.161
    Epoch  16 Batch  258/269 - Train Accuracy:  0.872, Validation Accuracy:  0.880, Loss:  0.165
    Epoch  16 Batch  259/269 - Train Accuracy:  0.881, Validation Accuracy:  0.872, Loss:  0.156
    Epoch  16 Batch  260/269 - Train Accuracy:  0.864, Validation Accuracy:  0.875, Loss:  0.167
    Epoch  16 Batch  261/269 - Train Accuracy:  0.843, Validation Accuracy:  0.871, Loss:  0.167
    Epoch  16 Batch  262/269 - Train Accuracy:  0.879, Validation Accuracy:  0.873, Loss:  0.157
    Epoch  16 Batch  263/269 - Train Accuracy:  0.860, Validation Accuracy:  0.874, Loss:  0.163
    Epoch  16 Batch  264/269 - Train Accuracy:  0.853, Validation Accuracy:  0.877, Loss:  0.164
    Epoch  16 Batch  265/269 - Train Accuracy:  0.863, Validation Accuracy:  0.869, Loss:  0.160
    Epoch  16 Batch  266/269 - Train Accuracy:  0.880, Validation Accuracy:  0.874, Loss:  0.158
    Epoch  16 Batch  267/269 - Train Accuracy:  0.869, Validation Accuracy:  0.868, Loss:  0.162
    Epoch  17 Batch    0/269 - Train Accuracy:  0.858, Validation Accuracy:  0.867, Loss:  0.169
    Epoch  17 Batch    1/269 - Train Accuracy:  0.859, Validation Accuracy:  0.865, Loss:  0.156
    Epoch  17 Batch    2/269 - Train Accuracy:  0.865, Validation Accuracy:  0.868, Loss:  0.164
    Epoch  17 Batch    3/269 - Train Accuracy:  0.870, Validation Accuracy:  0.871, Loss:  0.157
    Epoch  17 Batch    4/269 - Train Accuracy:  0.846, Validation Accuracy:  0.869, Loss:  0.168
    Epoch  17 Batch    5/269 - Train Accuracy:  0.863, Validation Accuracy:  0.873, Loss:  0.164
    Epoch  17 Batch    6/269 - Train Accuracy:  0.874, Validation Accuracy:  0.875, Loss:  0.152
    Epoch  17 Batch    7/269 - Train Accuracy:  0.880, Validation Accuracy:  0.873, Loss:  0.152
    Epoch  17 Batch    8/269 - Train Accuracy:  0.879, Validation Accuracy:  0.872, Loss:  0.162
    Epoch  17 Batch    9/269 - Train Accuracy:  0.870, Validation Accuracy:  0.871, Loss:  0.167
    Epoch  17 Batch   10/269 - Train Accuracy:  0.879, Validation Accuracy:  0.869, Loss:  0.158
    Epoch  17 Batch   11/269 - Train Accuracy:  0.863, Validation Accuracy:  0.866, Loss:  0.166
    Epoch  17 Batch   12/269 - Train Accuracy:  0.859, Validation Accuracy:  0.870, Loss:  0.168
    Epoch  17 Batch   13/269 - Train Accuracy:  0.867, Validation Accuracy:  0.865, Loss:  0.141
    Epoch  17 Batch   14/269 - Train Accuracy:  0.866, Validation Accuracy:  0.872, Loss:  0.161
    Epoch  17 Batch   15/269 - Train Accuracy:  0.873, Validation Accuracy:  0.875, Loss:  0.147
    Epoch  17 Batch   16/269 - Train Accuracy:  0.878, Validation Accuracy:  0.877, Loss:  0.154
    Epoch  17 Batch   17/269 - Train Accuracy:  0.879, Validation Accuracy:  0.869, Loss:  0.148
    Epoch  17 Batch   18/269 - Train Accuracy:  0.859, Validation Accuracy:  0.871, Loss:  0.163
    Epoch  17 Batch   19/269 - Train Accuracy:  0.879, Validation Accuracy:  0.868, Loss:  0.144
    Epoch  17 Batch   20/269 - Train Accuracy:  0.884, Validation Accuracy:  0.869, Loss:  0.162
    Epoch  17 Batch   21/269 - Train Accuracy:  0.861, Validation Accuracy:  0.874, Loss:  0.172
    Epoch  17 Batch   22/269 - Train Accuracy:  0.876, Validation Accuracy:  0.872, Loss:  0.146
    Epoch  17 Batch   23/269 - Train Accuracy:  0.872, Validation Accuracy:  0.874, Loss:  0.156
    Epoch  17 Batch   24/269 - Train Accuracy:  0.865, Validation Accuracy:  0.872, Loss:  0.156
    Epoch  17 Batch   25/269 - Train Accuracy:  0.856, Validation Accuracy:  0.870, Loss:  0.171
    Epoch  17 Batch   26/269 - Train Accuracy:  0.877, Validation Accuracy:  0.882, Loss:  0.145
    Epoch  17 Batch   27/269 - Train Accuracy:  0.862, Validation Accuracy:  0.874, Loss:  0.147
    Epoch  17 Batch   28/269 - Train Accuracy:  0.858, Validation Accuracy:  0.879, Loss:  0.172
    Epoch  17 Batch   29/269 - Train Accuracy:  0.878, Validation Accuracy:  0.873, Loss:  0.161
    Epoch  17 Batch   30/269 - Train Accuracy:  0.871, Validation Accuracy:  0.868, Loss:  0.149
    Epoch  17 Batch   31/269 - Train Accuracy:  0.886, Validation Accuracy:  0.870, Loss:  0.150
    Epoch  17 Batch   32/269 - Train Accuracy:  0.867, Validation Accuracy:  0.878, Loss:  0.145
    Epoch  17 Batch   33/269 - Train Accuracy:  0.872, Validation Accuracy:  0.877, Loss:  0.144
    Epoch  17 Batch   34/269 - Train Accuracy:  0.877, Validation Accuracy:  0.868, Loss:  0.150
    Epoch  17 Batch   35/269 - Train Accuracy:  0.858, Validation Accuracy:  0.863, Loss:  0.163
    Epoch  17 Batch   36/269 - Train Accuracy:  0.864, Validation Accuracy:  0.864, Loss:  0.155
    Epoch  17 Batch   37/269 - Train Accuracy:  0.870, Validation Accuracy:  0.872, Loss:  0.149
    Epoch  17 Batch   38/269 - Train Accuracy:  0.880, Validation Accuracy:  0.881, Loss:  0.146
    Epoch  17 Batch   39/269 - Train Accuracy:  0.874, Validation Accuracy:  0.880, Loss:  0.156
    Epoch  17 Batch   40/269 - Train Accuracy:  0.859, Validation Accuracy:  0.874, Loss:  0.159
    Epoch  17 Batch   41/269 - Train Accuracy:  0.863, Validation Accuracy:  0.876, Loss:  0.158
    Epoch  17 Batch   42/269 - Train Accuracy:  0.883, Validation Accuracy:  0.875, Loss:  0.142
    Epoch  17 Batch   43/269 - Train Accuracy:  0.880, Validation Accuracy:  0.880, Loss:  0.156
    Epoch  17 Batch   44/269 - Train Accuracy:  0.883, Validation Accuracy:  0.876, Loss:  0.157
    Epoch  17 Batch   45/269 - Train Accuracy:  0.877, Validation Accuracy:  0.882, Loss:  0.156
    Epoch  17 Batch   46/269 - Train Accuracy:  0.875, Validation Accuracy:  0.884, Loss:  0.153
    Epoch  17 Batch   47/269 - Train Accuracy:  0.885, Validation Accuracy:  0.873, Loss:  0.137
    Epoch  17 Batch   48/269 - Train Accuracy:  0.875, Validation Accuracy:  0.875, Loss:  0.149
    Epoch  17 Batch   49/269 - Train Accuracy:  0.866, Validation Accuracy:  0.881, Loss:  0.147
    Epoch  17 Batch   50/269 - Train Accuracy:  0.856, Validation Accuracy:  0.883, Loss:  0.162
    Epoch  17 Batch   51/269 - Train Accuracy:  0.868, Validation Accuracy:  0.878, Loss:  0.153
    Epoch  17 Batch   52/269 - Train Accuracy:  0.867, Validation Accuracy:  0.878, Loss:  0.138
    Epoch  17 Batch   53/269 - Train Accuracy:  0.860, Validation Accuracy:  0.876, Loss:  0.162
    Epoch  17 Batch   54/269 - Train Accuracy:  0.881, Validation Accuracy:  0.878, Loss:  0.147
    Epoch  17 Batch   55/269 - Train Accuracy:  0.883, Validation Accuracy:  0.876, Loss:  0.149
    Epoch  17 Batch   56/269 - Train Accuracy:  0.872, Validation Accuracy:  0.880, Loss:  0.150
    Epoch  17 Batch   57/269 - Train Accuracy:  0.867, Validation Accuracy:  0.883, Loss:  0.160
    Epoch  17 Batch   58/269 - Train Accuracy:  0.885, Validation Accuracy:  0.888, Loss:  0.149
    Epoch  17 Batch   59/269 - Train Accuracy:  0.899, Validation Accuracy:  0.884, Loss:  0.133
    Epoch  17 Batch   60/269 - Train Accuracy:  0.890, Validation Accuracy:  0.882, Loss:  0.146
    Epoch  17 Batch   61/269 - Train Accuracy:  0.880, Validation Accuracy:  0.880, Loss:  0.136
    Epoch  17 Batch   62/269 - Train Accuracy:  0.884, Validation Accuracy:  0.878, Loss:  0.145
    Epoch  17 Batch   63/269 - Train Accuracy:  0.881, Validation Accuracy:  0.878, Loss:  0.156
    Epoch  17 Batch   64/269 - Train Accuracy:  0.879, Validation Accuracy:  0.876, Loss:  0.141
    Epoch  17 Batch   65/269 - Train Accuracy:  0.873, Validation Accuracy:  0.880, Loss:  0.147
    Epoch  17 Batch   66/269 - Train Accuracy:  0.881, Validation Accuracy:  0.879, Loss:  0.143
    Epoch  17 Batch   67/269 - Train Accuracy:  0.869, Validation Accuracy:  0.881, Loss:  0.157
    Epoch  17 Batch   68/269 - Train Accuracy:  0.857, Validation Accuracy:  0.878, Loss:  0.156
    Epoch  17 Batch   69/269 - Train Accuracy:  0.850, Validation Accuracy:  0.880, Loss:  0.169
    Epoch  17 Batch   70/269 - Train Accuracy:  0.888, Validation Accuracy:  0.884, Loss:  0.147
    Epoch  17 Batch   71/269 - Train Accuracy:  0.873, Validation Accuracy:  0.883, Loss:  0.163
    Epoch  17 Batch   72/269 - Train Accuracy:  0.865, Validation Accuracy:  0.881, Loss:  0.154
    Epoch  17 Batch   73/269 - Train Accuracy:  0.879, Validation Accuracy:  0.887, Loss:  0.157
    Epoch  17 Batch   74/269 - Train Accuracy:  0.890, Validation Accuracy:  0.889, Loss:  0.149
    Epoch  17 Batch   75/269 - Train Accuracy:  0.875, Validation Accuracy:  0.881, Loss:  0.146
    Epoch  17 Batch   76/269 - Train Accuracy:  0.859, Validation Accuracy:  0.880, Loss:  0.150
    Epoch  17 Batch   77/269 - Train Accuracy:  0.882, Validation Accuracy:  0.877, Loss:  0.148
    Epoch  17 Batch   78/269 - Train Accuracy:  0.883, Validation Accuracy:  0.879, Loss:  0.148
    Epoch  17 Batch   79/269 - Train Accuracy:  0.864, Validation Accuracy:  0.875, Loss:  0.150
    Epoch  17 Batch   80/269 - Train Accuracy:  0.885, Validation Accuracy:  0.874, Loss:  0.145
    Epoch  17 Batch   81/269 - Train Accuracy:  0.873, Validation Accuracy:  0.873, Loss:  0.154
    Epoch  17 Batch   82/269 - Train Accuracy:  0.888, Validation Accuracy:  0.874, Loss:  0.134
    Epoch  17 Batch   83/269 - Train Accuracy:  0.867, Validation Accuracy:  0.871, Loss:  0.165
    Epoch  17 Batch   84/269 - Train Accuracy:  0.883, Validation Accuracy:  0.877, Loss:  0.148
    Epoch  17 Batch   85/269 - Train Accuracy:  0.871, Validation Accuracy:  0.876, Loss:  0.150
    Epoch  17 Batch   86/269 - Train Accuracy:  0.887, Validation Accuracy:  0.876, Loss:  0.141
    Epoch  17 Batch   87/269 - Train Accuracy:  0.867, Validation Accuracy:  0.872, Loss:  0.159
    Epoch  17 Batch   88/269 - Train Accuracy:  0.878, Validation Accuracy:  0.877, Loss:  0.150
    Epoch  17 Batch   89/269 - Train Accuracy:  0.890, Validation Accuracy:  0.873, Loss:  0.148
    Epoch  17 Batch   90/269 - Train Accuracy:  0.869, Validation Accuracy:  0.881, Loss:  0.159
    Epoch  17 Batch   91/269 - Train Accuracy:  0.880, Validation Accuracy:  0.879, Loss:  0.143
    Epoch  17 Batch   92/269 - Train Accuracy:  0.882, Validation Accuracy:  0.871, Loss:  0.145
    Epoch  17 Batch   93/269 - Train Accuracy:  0.887, Validation Accuracy:  0.875, Loss:  0.140
    Epoch  17 Batch   94/269 - Train Accuracy:  0.871, Validation Accuracy:  0.879, Loss:  0.159
    Epoch  17 Batch   95/269 - Train Accuracy:  0.887, Validation Accuracy:  0.878, Loss:  0.146
    Epoch  17 Batch   96/269 - Train Accuracy:  0.864, Validation Accuracy:  0.873, Loss:  0.154
    Epoch  17 Batch   97/269 - Train Accuracy:  0.887, Validation Accuracy:  0.873, Loss:  0.142
    Epoch  17 Batch   98/269 - Train Accuracy:  0.884, Validation Accuracy:  0.884, Loss:  0.147
    Epoch  17 Batch   99/269 - Train Accuracy:  0.885, Validation Accuracy:  0.883, Loss:  0.147
    Epoch  17 Batch  100/269 - Train Accuracy:  0.892, Validation Accuracy:  0.877, Loss:  0.149
    Epoch  17 Batch  101/269 - Train Accuracy:  0.860, Validation Accuracy:  0.875, Loss:  0.164
    Epoch  17 Batch  102/269 - Train Accuracy:  0.873, Validation Accuracy:  0.874, Loss:  0.144
    Epoch  17 Batch  103/269 - Train Accuracy:  0.895, Validation Accuracy:  0.880, Loss:  0.148
    Epoch  17 Batch  104/269 - Train Accuracy:  0.875, Validation Accuracy:  0.881, Loss:  0.148
    Epoch  17 Batch  105/269 - Train Accuracy:  0.876, Validation Accuracy:  0.880, Loss:  0.149
    Epoch  17 Batch  106/269 - Train Accuracy:  0.888, Validation Accuracy:  0.884, Loss:  0.140
    Epoch  17 Batch  107/269 - Train Accuracy:  0.884, Validation Accuracy:  0.880, Loss:  0.151
    Epoch  17 Batch  108/269 - Train Accuracy:  0.882, Validation Accuracy:  0.878, Loss:  0.143
    Epoch  17 Batch  109/269 - Train Accuracy:  0.859, Validation Accuracy:  0.884, Loss:  0.150
    Epoch  17 Batch  110/269 - Train Accuracy:  0.877, Validation Accuracy:  0.881, Loss:  0.139
    Epoch  17 Batch  111/269 - Train Accuracy:  0.875, Validation Accuracy:  0.881, Loss:  0.157
    Epoch  17 Batch  112/269 - Train Accuracy:  0.870, Validation Accuracy:  0.873, Loss:  0.151
    Epoch  17 Batch  113/269 - Train Accuracy:  0.865, Validation Accuracy:  0.874, Loss:  0.143
    Epoch  17 Batch  114/269 - Train Accuracy:  0.886, Validation Accuracy:  0.880, Loss:  0.143
    Epoch  17 Batch  115/269 - Train Accuracy:  0.867, Validation Accuracy:  0.874, Loss:  0.154
    Epoch  17 Batch  116/269 - Train Accuracy:  0.899, Validation Accuracy:  0.875, Loss:  0.147
    Epoch  17 Batch  117/269 - Train Accuracy:  0.876, Validation Accuracy:  0.877, Loss:  0.141
    Epoch  17 Batch  118/269 - Train Accuracy:  0.903, Validation Accuracy:  0.874, Loss:  0.137
    Epoch  17 Batch  119/269 - Train Accuracy:  0.869, Validation Accuracy:  0.882, Loss:  0.150
    Epoch  17 Batch  120/269 - Train Accuracy:  0.882, Validation Accuracy:  0.880, Loss:  0.148
    Epoch  17 Batch  121/269 - Train Accuracy:  0.885, Validation Accuracy:  0.884, Loss:  0.143
    Epoch  17 Batch  122/269 - Train Accuracy:  0.881, Validation Accuracy:  0.885, Loss:  0.142
    Epoch  17 Batch  123/269 - Train Accuracy:  0.874, Validation Accuracy:  0.883, Loss:  0.143
    Epoch  17 Batch  124/269 - Train Accuracy:  0.876, Validation Accuracy:  0.882, Loss:  0.135
    Epoch  17 Batch  125/269 - Train Accuracy:  0.884, Validation Accuracy:  0.882, Loss:  0.142
    Epoch  17 Batch  126/269 - Train Accuracy:  0.866, Validation Accuracy:  0.885, Loss:  0.142
    Epoch  17 Batch  127/269 - Train Accuracy:  0.877, Validation Accuracy:  0.880, Loss:  0.145
    Epoch  17 Batch  128/269 - Train Accuracy:  0.865, Validation Accuracy:  0.879, Loss:  0.148
    Epoch  17 Batch  129/269 - Train Accuracy:  0.871, Validation Accuracy:  0.881, Loss:  0.145
    Epoch  17 Batch  130/269 - Train Accuracy:  0.884, Validation Accuracy:  0.879, Loss:  0.149
    Epoch  17 Batch  131/269 - Train Accuracy:  0.861, Validation Accuracy:  0.882, Loss:  0.151
    Epoch  17 Batch  132/269 - Train Accuracy:  0.874, Validation Accuracy:  0.889, Loss:  0.152
    Epoch  17 Batch  133/269 - Train Accuracy:  0.883, Validation Accuracy:  0.882, Loss:  0.137
    Epoch  17 Batch  134/269 - Train Accuracy:  0.866, Validation Accuracy:  0.885, Loss:  0.144
    Epoch  17 Batch  135/269 - Train Accuracy:  0.883, Validation Accuracy:  0.885, Loss:  0.157
    Epoch  17 Batch  136/269 - Train Accuracy:  0.861, Validation Accuracy:  0.880, Loss:  0.153
    Epoch  17 Batch  137/269 - Train Accuracy:  0.870, Validation Accuracy:  0.878, Loss:  0.159
    Epoch  17 Batch  138/269 - Train Accuracy:  0.870, Validation Accuracy:  0.883, Loss:  0.148
    Epoch  17 Batch  139/269 - Train Accuracy:  0.881, Validation Accuracy:  0.877, Loss:  0.139
    Epoch  17 Batch  140/269 - Train Accuracy:  0.884, Validation Accuracy:  0.884, Loss:  0.156
    Epoch  17 Batch  141/269 - Train Accuracy:  0.871, Validation Accuracy:  0.884, Loss:  0.150
    Epoch  17 Batch  142/269 - Train Accuracy:  0.882, Validation Accuracy:  0.887, Loss:  0.139
    Epoch  17 Batch  143/269 - Train Accuracy:  0.892, Validation Accuracy:  0.885, Loss:  0.142
    Epoch  17 Batch  144/269 - Train Accuracy:  0.905, Validation Accuracy:  0.887, Loss:  0.126
    Epoch  17 Batch  145/269 - Train Accuracy:  0.882, Validation Accuracy:  0.881, Loss:  0.140
    Epoch  17 Batch  146/269 - Train Accuracy:  0.878, Validation Accuracy:  0.880, Loss:  0.137
    Epoch  17 Batch  147/269 - Train Accuracy:  0.881, Validation Accuracy:  0.877, Loss:  0.144
    Epoch  17 Batch  148/269 - Train Accuracy:  0.870, Validation Accuracy:  0.878, Loss:  0.141
    Epoch  17 Batch  149/269 - Train Accuracy:  0.875, Validation Accuracy:  0.879, Loss:  0.143
    Epoch  17 Batch  150/269 - Train Accuracy:  0.877, Validation Accuracy:  0.881, Loss:  0.145
    Epoch  17 Batch  151/269 - Train Accuracy:  0.879, Validation Accuracy:  0.877, Loss:  0.146
    Epoch  17 Batch  152/269 - Train Accuracy:  0.880, Validation Accuracy:  0.882, Loss:  0.148
    Epoch  17 Batch  153/269 - Train Accuracy:  0.893, Validation Accuracy:  0.893, Loss:  0.142
    Epoch  17 Batch  154/269 - Train Accuracy:  0.888, Validation Accuracy:  0.886, Loss:  0.142
    Epoch  17 Batch  155/269 - Train Accuracy:  0.879, Validation Accuracy:  0.889, Loss:  0.142
    Epoch  17 Batch  156/269 - Train Accuracy:  0.883, Validation Accuracy:  0.884, Loss:  0.150
    Epoch  17 Batch  157/269 - Train Accuracy:  0.872, Validation Accuracy:  0.886, Loss:  0.139
    Epoch  17 Batch  158/269 - Train Accuracy:  0.886, Validation Accuracy:  0.884, Loss:  0.148
    Epoch  17 Batch  159/269 - Train Accuracy:  0.862, Validation Accuracy:  0.879, Loss:  0.145
    Epoch  17 Batch  160/269 - Train Accuracy:  0.879, Validation Accuracy:  0.880, Loss:  0.140
    Epoch  17 Batch  161/269 - Train Accuracy:  0.882, Validation Accuracy:  0.885, Loss:  0.141
    Epoch  17 Batch  162/269 - Train Accuracy:  0.894, Validation Accuracy:  0.890, Loss:  0.143
    Epoch  17 Batch  163/269 - Train Accuracy:  0.889, Validation Accuracy:  0.885, Loss:  0.149
    Epoch  17 Batch  164/269 - Train Accuracy:  0.895, Validation Accuracy:  0.889, Loss:  0.139
    Epoch  17 Batch  165/269 - Train Accuracy:  0.889, Validation Accuracy:  0.880, Loss:  0.147
    Epoch  17 Batch  166/269 - Train Accuracy:  0.892, Validation Accuracy:  0.888, Loss:  0.137
    Epoch  17 Batch  167/269 - Train Accuracy:  0.877, Validation Accuracy:  0.876, Loss:  0.143
    Epoch  17 Batch  168/269 - Train Accuracy:  0.887, Validation Accuracy:  0.880, Loss:  0.145
    Epoch  17 Batch  169/269 - Train Accuracy:  0.870, Validation Accuracy:  0.881, Loss:  0.145
    Epoch  17 Batch  170/269 - Train Accuracy:  0.880, Validation Accuracy:  0.879, Loss:  0.137
    Epoch  17 Batch  171/269 - Train Accuracy:  0.891, Validation Accuracy:  0.877, Loss:  0.145
    Epoch  17 Batch  172/269 - Train Accuracy:  0.872, Validation Accuracy:  0.874, Loss:  0.152
    Epoch  17 Batch  173/269 - Train Accuracy:  0.888, Validation Accuracy:  0.873, Loss:  0.136
    Epoch  17 Batch  174/269 - Train Accuracy:  0.895, Validation Accuracy:  0.871, Loss:  0.138
    Epoch  17 Batch  175/269 - Train Accuracy:  0.870, Validation Accuracy:  0.882, Loss:  0.154
    Epoch  17 Batch  176/269 - Train Accuracy:  0.864, Validation Accuracy:  0.879, Loss:  0.152
    Epoch  17 Batch  177/269 - Train Accuracy:  0.887, Validation Accuracy:  0.885, Loss:  0.137
    Epoch  17 Batch  178/269 - Train Accuracy:  0.894, Validation Accuracy:  0.886, Loss:  0.132
    Epoch  17 Batch  179/269 - Train Accuracy:  0.882, Validation Accuracy:  0.889, Loss:  0.139
    Epoch  17 Batch  180/269 - Train Accuracy:  0.891, Validation Accuracy:  0.888, Loss:  0.140
    Epoch  17 Batch  181/269 - Train Accuracy:  0.879, Validation Accuracy:  0.893, Loss:  0.144
    Epoch  17 Batch  182/269 - Train Accuracy:  0.892, Validation Accuracy:  0.890, Loss:  0.142
    Epoch  17 Batch  183/269 - Train Accuracy:  0.903, Validation Accuracy:  0.890, Loss:  0.122
    Epoch  17 Batch  184/269 - Train Accuracy:  0.887, Validation Accuracy:  0.890, Loss:  0.144
    Epoch  17 Batch  185/269 - Train Accuracy:  0.898, Validation Accuracy:  0.890, Loss:  0.137
    Epoch  17 Batch  186/269 - Train Accuracy:  0.885, Validation Accuracy:  0.882, Loss:  0.132
    Epoch  17 Batch  187/269 - Train Accuracy:  0.884, Validation Accuracy:  0.878, Loss:  0.141
    Epoch  17 Batch  188/269 - Train Accuracy:  0.890, Validation Accuracy:  0.879, Loss:  0.133
    Epoch  17 Batch  189/269 - Train Accuracy:  0.890, Validation Accuracy:  0.886, Loss:  0.136
    Epoch  17 Batch  190/269 - Train Accuracy:  0.896, Validation Accuracy:  0.890, Loss:  0.136
    Epoch  17 Batch  191/269 - Train Accuracy:  0.872, Validation Accuracy:  0.889, Loss:  0.136
    Epoch  17 Batch  192/269 - Train Accuracy:  0.886, Validation Accuracy:  0.894, Loss:  0.143
    Epoch  17 Batch  193/269 - Train Accuracy:  0.882, Validation Accuracy:  0.892, Loss:  0.129
    Epoch  17 Batch  194/269 - Train Accuracy:  0.883, Validation Accuracy:  0.891, Loss:  0.146
    Epoch  17 Batch  195/269 - Train Accuracy:  0.878, Validation Accuracy:  0.891, Loss:  0.132
    Epoch  17 Batch  196/269 - Train Accuracy:  0.878, Validation Accuracy:  0.887, Loss:  0.132
    Epoch  17 Batch  197/269 - Train Accuracy:  0.879, Validation Accuracy:  0.888, Loss:  0.146
    Epoch  17 Batch  198/269 - Train Accuracy:  0.882, Validation Accuracy:  0.881, Loss:  0.143
    Epoch  17 Batch  199/269 - Train Accuracy:  0.890, Validation Accuracy:  0.890, Loss:  0.143
    Epoch  17 Batch  200/269 - Train Accuracy:  0.886, Validation Accuracy:  0.882, Loss:  0.142
    Epoch  17 Batch  201/269 - Train Accuracy:  0.878, Validation Accuracy:  0.889, Loss:  0.139
    Epoch  17 Batch  202/269 - Train Accuracy:  0.875, Validation Accuracy:  0.889, Loss:  0.138
    Epoch  17 Batch  203/269 - Train Accuracy:  0.893, Validation Accuracy:  0.885, Loss:  0.147
    Epoch  17 Batch  204/269 - Train Accuracy:  0.874, Validation Accuracy:  0.887, Loss:  0.146
    Epoch  17 Batch  205/269 - Train Accuracy:  0.896, Validation Accuracy:  0.883, Loss:  0.133
    Epoch  17 Batch  206/269 - Train Accuracy:  0.872, Validation Accuracy:  0.882, Loss:  0.142
    Epoch  17 Batch  207/269 - Train Accuracy:  0.886, Validation Accuracy:  0.880, Loss:  0.134
    Epoch  17 Batch  208/269 - Train Accuracy:  0.881, Validation Accuracy:  0.890, Loss:  0.143
    Epoch  17 Batch  209/269 - Train Accuracy:  0.893, Validation Accuracy:  0.892, Loss:  0.138
    Epoch  17 Batch  210/269 - Train Accuracy:  0.888, Validation Accuracy:  0.892, Loss:  0.137
    Epoch  17 Batch  211/269 - Train Accuracy:  0.888, Validation Accuracy:  0.894, Loss:  0.138
    Epoch  17 Batch  212/269 - Train Accuracy:  0.884, Validation Accuracy:  0.895, Loss:  0.139
    Epoch  17 Batch  213/269 - Train Accuracy:  0.883, Validation Accuracy:  0.893, Loss:  0.134
    Epoch  17 Batch  214/269 - Train Accuracy:  0.883, Validation Accuracy:  0.891, Loss:  0.133
    Epoch  17 Batch  215/269 - Train Accuracy:  0.890, Validation Accuracy:  0.886, Loss:  0.123
    Epoch  17 Batch  216/269 - Train Accuracy:  0.866, Validation Accuracy:  0.888, Loss:  0.157
    Epoch  17 Batch  217/269 - Train Accuracy:  0.871, Validation Accuracy:  0.887, Loss:  0.144
    Epoch  17 Batch  218/269 - Train Accuracy:  0.894, Validation Accuracy:  0.892, Loss:  0.136
    Epoch  17 Batch  219/269 - Train Accuracy:  0.890, Validation Accuracy:  0.886, Loss:  0.141
    Epoch  17 Batch  220/269 - Train Accuracy:  0.883, Validation Accuracy:  0.891, Loss:  0.134
    Epoch  17 Batch  221/269 - Train Accuracy:  0.881, Validation Accuracy:  0.888, Loss:  0.137
    Epoch  17 Batch  222/269 - Train Accuracy:  0.900, Validation Accuracy:  0.889, Loss:  0.130
    Epoch  17 Batch  223/269 - Train Accuracy:  0.882, Validation Accuracy:  0.889, Loss:  0.125
    Epoch  17 Batch  224/269 - Train Accuracy:  0.892, Validation Accuracy:  0.897, Loss:  0.140
    Epoch  17 Batch  225/269 - Train Accuracy:  0.882, Validation Accuracy:  0.895, Loss:  0.131
    Epoch  17 Batch  226/269 - Train Accuracy:  0.894, Validation Accuracy:  0.889, Loss:  0.135
    Epoch  17 Batch  227/269 - Train Accuracy:  0.904, Validation Accuracy:  0.887, Loss:  0.129
    Epoch  17 Batch  228/269 - Train Accuracy:  0.880, Validation Accuracy:  0.886, Loss:  0.131
    Epoch  17 Batch  229/269 - Train Accuracy:  0.880, Validation Accuracy:  0.888, Loss:  0.131
    Epoch  17 Batch  230/269 - Train Accuracy:  0.884, Validation Accuracy:  0.885, Loss:  0.137
    Epoch  17 Batch  231/269 - Train Accuracy:  0.880, Validation Accuracy:  0.888, Loss:  0.141
    Epoch  17 Batch  232/269 - Train Accuracy:  0.879, Validation Accuracy:  0.890, Loss:  0.135
    Epoch  17 Batch  233/269 - Train Accuracy:  0.899, Validation Accuracy:  0.885, Loss:  0.140
    Epoch  17 Batch  234/269 - Train Accuracy:  0.888, Validation Accuracy:  0.887, Loss:  0.136
    Epoch  17 Batch  235/269 - Train Accuracy:  0.901, Validation Accuracy:  0.880, Loss:  0.120
    Epoch  17 Batch  236/269 - Train Accuracy:  0.881, Validation Accuracy:  0.888, Loss:  0.129
    Epoch  17 Batch  237/269 - Train Accuracy:  0.889, Validation Accuracy:  0.885, Loss:  0.131
    Epoch  17 Batch  238/269 - Train Accuracy:  0.890, Validation Accuracy:  0.893, Loss:  0.129
    Epoch  17 Batch  239/269 - Train Accuracy:  0.891, Validation Accuracy:  0.890, Loss:  0.131
    Epoch  17 Batch  240/269 - Train Accuracy:  0.909, Validation Accuracy:  0.888, Loss:  0.119
    Epoch  17 Batch  241/269 - Train Accuracy:  0.880, Validation Accuracy:  0.893, Loss:  0.144
    Epoch  17 Batch  242/269 - Train Accuracy:  0.893, Validation Accuracy:  0.890, Loss:  0.139
    Epoch  17 Batch  243/269 - Train Accuracy:  0.892, Validation Accuracy:  0.880, Loss:  0.123
    Epoch  17 Batch  244/269 - Train Accuracy:  0.877, Validation Accuracy:  0.885, Loss:  0.140
    Epoch  17 Batch  245/269 - Train Accuracy:  0.864, Validation Accuracy:  0.888, Loss:  0.144
    Epoch  17 Batch  246/269 - Train Accuracy:  0.876, Validation Accuracy:  0.884, Loss:  0.141
    Epoch  17 Batch  247/269 - Train Accuracy:  0.882, Validation Accuracy:  0.877, Loss:  0.135
    Epoch  17 Batch  248/269 - Train Accuracy:  0.892, Validation Accuracy:  0.885, Loss:  0.130
    Epoch  17 Batch  249/269 - Train Accuracy:  0.902, Validation Accuracy:  0.886, Loss:  0.119
    Epoch  17 Batch  250/269 - Train Accuracy:  0.894, Validation Accuracy:  0.896, Loss:  0.132
    Epoch  17 Batch  251/269 - Train Accuracy:  0.892, Validation Accuracy:  0.882, Loss:  0.122
    Epoch  17 Batch  252/269 - Train Accuracy:  0.888, Validation Accuracy:  0.892, Loss:  0.133
    Epoch  17 Batch  253/269 - Train Accuracy:  0.871, Validation Accuracy:  0.894, Loss:  0.143
    Epoch  17 Batch  254/269 - Train Accuracy:  0.890, Validation Accuracy:  0.892, Loss:  0.133
    Epoch  17 Batch  255/269 - Train Accuracy:  0.897, Validation Accuracy:  0.891, Loss:  0.124
    Epoch  17 Batch  256/269 - Train Accuracy:  0.874, Validation Accuracy:  0.894, Loss:  0.131
    Epoch  17 Batch  257/269 - Train Accuracy:  0.869, Validation Accuracy:  0.895, Loss:  0.138
    Epoch  17 Batch  258/269 - Train Accuracy:  0.886, Validation Accuracy:  0.892, Loss:  0.136
    Epoch  17 Batch  259/269 - Train Accuracy:  0.902, Validation Accuracy:  0.894, Loss:  0.132
    Epoch  17 Batch  260/269 - Train Accuracy:  0.884, Validation Accuracy:  0.895, Loss:  0.138
    Epoch  17 Batch  261/269 - Train Accuracy:  0.877, Validation Accuracy:  0.895, Loss:  0.136
    Epoch  17 Batch  262/269 - Train Accuracy:  0.896, Validation Accuracy:  0.894, Loss:  0.128
    Epoch  17 Batch  263/269 - Train Accuracy:  0.879, Validation Accuracy:  0.897, Loss:  0.137
    Epoch  17 Batch  264/269 - Train Accuracy:  0.876, Validation Accuracy:  0.894, Loss:  0.135
    Epoch  17 Batch  265/269 - Train Accuracy:  0.881, Validation Accuracy:  0.891, Loss:  0.130
    Epoch  17 Batch  266/269 - Train Accuracy:  0.896, Validation Accuracy:  0.889, Loss:  0.124
    Epoch  17 Batch  267/269 - Train Accuracy:  0.893, Validation Accuracy:  0.889, Loss:  0.138
    Epoch  18 Batch    0/269 - Train Accuracy:  0.888, Validation Accuracy:  0.888, Loss:  0.141
    Epoch  18 Batch    1/269 - Train Accuracy:  0.885, Validation Accuracy:  0.889, Loss:  0.129
    Epoch  18 Batch    2/269 - Train Accuracy:  0.885, Validation Accuracy:  0.888, Loss:  0.139
    Epoch  18 Batch    3/269 - Train Accuracy:  0.889, Validation Accuracy:  0.890, Loss:  0.128
    Epoch  18 Batch    4/269 - Train Accuracy:  0.879, Validation Accuracy:  0.891, Loss:  0.141
    Epoch  18 Batch    5/269 - Train Accuracy:  0.877, Validation Accuracy:  0.889, Loss:  0.138
    Epoch  18 Batch    6/269 - Train Accuracy:  0.899, Validation Accuracy:  0.893, Loss:  0.128
    Epoch  18 Batch    7/269 - Train Accuracy:  0.895, Validation Accuracy:  0.893, Loss:  0.128
    Epoch  18 Batch    8/269 - Train Accuracy:  0.902, Validation Accuracy:  0.892, Loss:  0.138
    Epoch  18 Batch    9/269 - Train Accuracy:  0.885, Validation Accuracy:  0.893, Loss:  0.141
    Epoch  18 Batch   10/269 - Train Accuracy:  0.901, Validation Accuracy:  0.893, Loss:  0.134
    Epoch  18 Batch   11/269 - Train Accuracy:  0.894, Validation Accuracy:  0.896, Loss:  0.138
    Epoch  18 Batch   12/269 - Train Accuracy:  0.888, Validation Accuracy:  0.899, Loss:  0.142
    Epoch  18 Batch   13/269 - Train Accuracy:  0.883, Validation Accuracy:  0.897, Loss:  0.120
    Epoch  18 Batch   14/269 - Train Accuracy:  0.886, Validation Accuracy:  0.891, Loss:  0.133
    Epoch  18 Batch   15/269 - Train Accuracy:  0.897, Validation Accuracy:  0.899, Loss:  0.121
    Epoch  18 Batch   16/269 - Train Accuracy:  0.889, Validation Accuracy:  0.899, Loss:  0.134
    Epoch  18 Batch   17/269 - Train Accuracy:  0.899, Validation Accuracy:  0.895, Loss:  0.124
    Epoch  18 Batch   18/269 - Train Accuracy:  0.878, Validation Accuracy:  0.892, Loss:  0.135
    Epoch  18 Batch   19/269 - Train Accuracy:  0.895, Validation Accuracy:  0.895, Loss:  0.118
    Epoch  18 Batch   20/269 - Train Accuracy:  0.897, Validation Accuracy:  0.898, Loss:  0.135
    Epoch  18 Batch   21/269 - Train Accuracy:  0.870, Validation Accuracy:  0.893, Loss:  0.141
    Epoch  18 Batch   22/269 - Train Accuracy:  0.900, Validation Accuracy:  0.895, Loss:  0.122
    Epoch  18 Batch   23/269 - Train Accuracy:  0.897, Validation Accuracy:  0.899, Loss:  0.127
    Epoch  18 Batch   24/269 - Train Accuracy:  0.885, Validation Accuracy:  0.892, Loss:  0.131
    Epoch  18 Batch   25/269 - Train Accuracy:  0.873, Validation Accuracy:  0.890, Loss:  0.144
    Epoch  18 Batch   26/269 - Train Accuracy:  0.889, Validation Accuracy:  0.888, Loss:  0.126
    Epoch  18 Batch   27/269 - Train Accuracy:  0.885, Validation Accuracy:  0.896, Loss:  0.128
    Epoch  18 Batch   28/269 - Train Accuracy:  0.868, Validation Accuracy:  0.883, Loss:  0.143
    Epoch  18 Batch   29/269 - Train Accuracy:  0.889, Validation Accuracy:  0.889, Loss:  0.145
    Epoch  18 Batch   30/269 - Train Accuracy:  0.884, Validation Accuracy:  0.889, Loss:  0.123
    Epoch  18 Batch   31/269 - Train Accuracy:  0.899, Validation Accuracy:  0.892, Loss:  0.125
    Epoch  18 Batch   32/269 - Train Accuracy:  0.877, Validation Accuracy:  0.890, Loss:  0.119
    Epoch  18 Batch   33/269 - Train Accuracy:  0.890, Validation Accuracy:  0.893, Loss:  0.122
    Epoch  18 Batch   34/269 - Train Accuracy:  0.892, Validation Accuracy:  0.888, Loss:  0.122
    Epoch  18 Batch   35/269 - Train Accuracy:  0.885, Validation Accuracy:  0.888, Loss:  0.139
    Epoch  18 Batch   36/269 - Train Accuracy:  0.884, Validation Accuracy:  0.892, Loss:  0.129
    Epoch  18 Batch   37/269 - Train Accuracy:  0.895, Validation Accuracy:  0.896, Loss:  0.124
    Epoch  18 Batch   38/269 - Train Accuracy:  0.890, Validation Accuracy:  0.891, Loss:  0.122
    Epoch  18 Batch   39/269 - Train Accuracy:  0.891, Validation Accuracy:  0.886, Loss:  0.129
    Epoch  18 Batch   40/269 - Train Accuracy:  0.882, Validation Accuracy:  0.892, Loss:  0.133
    Epoch  18 Batch   41/269 - Train Accuracy:  0.884, Validation Accuracy:  0.891, Loss:  0.132
    Epoch  18 Batch   42/269 - Train Accuracy:  0.907, Validation Accuracy:  0.894, Loss:  0.116
    Epoch  18 Batch   43/269 - Train Accuracy:  0.902, Validation Accuracy:  0.893, Loss:  0.129
    Epoch  18 Batch   44/269 - Train Accuracy:  0.899, Validation Accuracy:  0.903, Loss:  0.129
    Epoch  18 Batch   45/269 - Train Accuracy:  0.898, Validation Accuracy:  0.896, Loss:  0.135
    Epoch  18 Batch   46/269 - Train Accuracy:  0.887, Validation Accuracy:  0.891, Loss:  0.126
    Epoch  18 Batch   47/269 - Train Accuracy:  0.899, Validation Accuracy:  0.892, Loss:  0.118
    Epoch  18 Batch   48/269 - Train Accuracy:  0.890, Validation Accuracy:  0.893, Loss:  0.122
    Epoch  18 Batch   49/269 - Train Accuracy:  0.883, Validation Accuracy:  0.892, Loss:  0.123
    Epoch  18 Batch   50/269 - Train Accuracy:  0.872, Validation Accuracy:  0.887, Loss:  0.138
    Epoch  18 Batch   51/269 - Train Accuracy:  0.888, Validation Accuracy:  0.889, Loss:  0.128
    Epoch  18 Batch   52/269 - Train Accuracy:  0.873, Validation Accuracy:  0.889, Loss:  0.118
    Epoch  18 Batch   53/269 - Train Accuracy:  0.882, Validation Accuracy:  0.893, Loss:  0.133
    Epoch  18 Batch   54/269 - Train Accuracy:  0.897, Validation Accuracy:  0.894, Loss:  0.122
    Epoch  18 Batch   55/269 - Train Accuracy:  0.893, Validation Accuracy:  0.893, Loss:  0.122
    Epoch  18 Batch   56/269 - Train Accuracy:  0.884, Validation Accuracy:  0.898, Loss:  0.127
    Epoch  18 Batch   57/269 - Train Accuracy:  0.882, Validation Accuracy:  0.893, Loss:  0.128
    Epoch  18 Batch   58/269 - Train Accuracy:  0.907, Validation Accuracy:  0.898, Loss:  0.120
    Epoch  18 Batch   59/269 - Train Accuracy:  0.913, Validation Accuracy:  0.893, Loss:  0.111
    Epoch  18 Batch   60/269 - Train Accuracy:  0.901, Validation Accuracy:  0.895, Loss:  0.123
    Epoch  18 Batch   61/269 - Train Accuracy:  0.905, Validation Accuracy:  0.899, Loss:  0.115
    Epoch  18 Batch   62/269 - Train Accuracy:  0.902, Validation Accuracy:  0.900, Loss:  0.130
    Epoch  18 Batch   63/269 - Train Accuracy:  0.890, Validation Accuracy:  0.894, Loss:  0.131
    Epoch  18 Batch   64/269 - Train Accuracy:  0.889, Validation Accuracy:  0.897, Loss:  0.120
    Epoch  18 Batch   65/269 - Train Accuracy:  0.891, Validation Accuracy:  0.896, Loss:  0.123
    Epoch  18 Batch   66/269 - Train Accuracy:  0.895, Validation Accuracy:  0.894, Loss:  0.124
    Epoch  18 Batch   67/269 - Train Accuracy:  0.892, Validation Accuracy:  0.887, Loss:  0.132
    Epoch  18 Batch   68/269 - Train Accuracy:  0.873, Validation Accuracy:  0.890, Loss:  0.131
    Epoch  18 Batch   69/269 - Train Accuracy:  0.869, Validation Accuracy:  0.886, Loss:  0.145
    Epoch  18 Batch   70/269 - Train Accuracy:  0.906, Validation Accuracy:  0.897, Loss:  0.125
    Epoch  18 Batch   71/269 - Train Accuracy:  0.894, Validation Accuracy:  0.897, Loss:  0.135
    Epoch  18 Batch   72/269 - Train Accuracy:  0.880, Validation Accuracy:  0.902, Loss:  0.133
    Epoch  18 Batch   73/269 - Train Accuracy:  0.890, Validation Accuracy:  0.889, Loss:  0.131
    Epoch  18 Batch   74/269 - Train Accuracy:  0.903, Validation Accuracy:  0.896, Loss:  0.130
    Epoch  18 Batch   75/269 - Train Accuracy:  0.888, Validation Accuracy:  0.894, Loss:  0.123
    Epoch  18 Batch   76/269 - Train Accuracy:  0.868, Validation Accuracy:  0.886, Loss:  0.127
    Epoch  18 Batch   77/269 - Train Accuracy:  0.896, Validation Accuracy:  0.891, Loss:  0.123
    Epoch  18 Batch   78/269 - Train Accuracy:  0.896, Validation Accuracy:  0.897, Loss:  0.124
    Epoch  18 Batch   79/269 - Train Accuracy:  0.887, Validation Accuracy:  0.887, Loss:  0.126
    Epoch  18 Batch   80/269 - Train Accuracy:  0.893, Validation Accuracy:  0.883, Loss:  0.122
    Epoch  18 Batch   81/269 - Train Accuracy:  0.882, Validation Accuracy:  0.889, Loss:  0.128
    Epoch  18 Batch   82/269 - Train Accuracy:  0.903, Validation Accuracy:  0.890, Loss:  0.114
    Epoch  18 Batch   83/269 - Train Accuracy:  0.890, Validation Accuracy:  0.893, Loss:  0.137
    Epoch  18 Batch   84/269 - Train Accuracy:  0.895, Validation Accuracy:  0.897, Loss:  0.124
    Epoch  18 Batch   85/269 - Train Accuracy:  0.884, Validation Accuracy:  0.897, Loss:  0.127
    Epoch  18 Batch   86/269 - Train Accuracy:  0.897, Validation Accuracy:  0.895, Loss:  0.118
    Epoch  18 Batch   87/269 - Train Accuracy:  0.880, Validation Accuracy:  0.895, Loss:  0.127
    Epoch  18 Batch   88/269 - Train Accuracy:  0.884, Validation Accuracy:  0.897, Loss:  0.129
    Epoch  18 Batch   89/269 - Train Accuracy:  0.896, Validation Accuracy:  0.898, Loss:  0.125
    Epoch  18 Batch   90/269 - Train Accuracy:  0.890, Validation Accuracy:  0.897, Loss:  0.130
    Epoch  18 Batch   91/269 - Train Accuracy:  0.890, Validation Accuracy:  0.892, Loss:  0.116
    Epoch  18 Batch   92/269 - Train Accuracy:  0.898, Validation Accuracy:  0.892, Loss:  0.119
    Epoch  18 Batch   93/269 - Train Accuracy:  0.894, Validation Accuracy:  0.892, Loss:  0.116
    Epoch  18 Batch   94/269 - Train Accuracy:  0.886, Validation Accuracy:  0.892, Loss:  0.136
    Epoch  18 Batch   95/269 - Train Accuracy:  0.895, Validation Accuracy:  0.890, Loss:  0.121
    Epoch  18 Batch   96/269 - Train Accuracy:  0.874, Validation Accuracy:  0.894, Loss:  0.133
    Epoch  18 Batch   97/269 - Train Accuracy:  0.900, Validation Accuracy:  0.902, Loss:  0.121
    Epoch  18 Batch   98/269 - Train Accuracy:  0.900, Validation Accuracy:  0.894, Loss:  0.125
    Epoch  18 Batch   99/269 - Train Accuracy:  0.889, Validation Accuracy:  0.892, Loss:  0.125
    Epoch  18 Batch  100/269 - Train Accuracy:  0.898, Validation Accuracy:  0.889, Loss:  0.127
    Epoch  18 Batch  101/269 - Train Accuracy:  0.887, Validation Accuracy:  0.894, Loss:  0.144
    Epoch  18 Batch  102/269 - Train Accuracy:  0.890, Validation Accuracy:  0.893, Loss:  0.121
    Epoch  18 Batch  103/269 - Train Accuracy:  0.902, Validation Accuracy:  0.895, Loss:  0.124
    Epoch  18 Batch  104/269 - Train Accuracy:  0.891, Validation Accuracy:  0.895, Loss:  0.119
    Epoch  18 Batch  105/269 - Train Accuracy:  0.891, Validation Accuracy:  0.893, Loss:  0.129
    Epoch  18 Batch  106/269 - Train Accuracy:  0.891, Validation Accuracy:  0.894, Loss:  0.113
    Epoch  18 Batch  107/269 - Train Accuracy:  0.893, Validation Accuracy:  0.899, Loss:  0.133
    Epoch  18 Batch  108/269 - Train Accuracy:  0.900, Validation Accuracy:  0.889, Loss:  0.120
    Epoch  18 Batch  109/269 - Train Accuracy:  0.876, Validation Accuracy:  0.893, Loss:  0.132
    Epoch  18 Batch  110/269 - Train Accuracy:  0.886, Validation Accuracy:  0.889, Loss:  0.116
    Epoch  18 Batch  111/269 - Train Accuracy:  0.896, Validation Accuracy:  0.891, Loss:  0.141
    Epoch  18 Batch  112/269 - Train Accuracy:  0.882, Validation Accuracy:  0.885, Loss:  0.123
    Epoch  18 Batch  113/269 - Train Accuracy:  0.881, Validation Accuracy:  0.893, Loss:  0.123
    Epoch  18 Batch  114/269 - Train Accuracy:  0.887, Validation Accuracy:  0.889, Loss:  0.123
    Epoch  18 Batch  115/269 - Train Accuracy:  0.878, Validation Accuracy:  0.892, Loss:  0.129
    Epoch  18 Batch  116/269 - Train Accuracy:  0.902, Validation Accuracy:  0.896, Loss:  0.126
    Epoch  18 Batch  117/269 - Train Accuracy:  0.888, Validation Accuracy:  0.897, Loss:  0.120
    Epoch  18 Batch  118/269 - Train Accuracy:  0.912, Validation Accuracy:  0.897, Loss:  0.117
    Epoch  18 Batch  119/269 - Train Accuracy:  0.887, Validation Accuracy:  0.897, Loss:  0.126
    Epoch  18 Batch  120/269 - Train Accuracy:  0.891, Validation Accuracy:  0.899, Loss:  0.128
    Epoch  18 Batch  121/269 - Train Accuracy:  0.891, Validation Accuracy:  0.896, Loss:  0.116
    Epoch  18 Batch  122/269 - Train Accuracy:  0.890, Validation Accuracy:  0.901, Loss:  0.126
    Epoch  18 Batch  123/269 - Train Accuracy:  0.891, Validation Accuracy:  0.899, Loss:  0.121
    Epoch  18 Batch  124/269 - Train Accuracy:  0.900, Validation Accuracy:  0.902, Loss:  0.118
    Epoch  18 Batch  125/269 - Train Accuracy:  0.898, Validation Accuracy:  0.899, Loss:  0.122
    Epoch  18 Batch  126/269 - Train Accuracy:  0.880, Validation Accuracy:  0.899, Loss:  0.123
    Epoch  18 Batch  127/269 - Train Accuracy:  0.882, Validation Accuracy:  0.899, Loss:  0.124
    Epoch  18 Batch  128/269 - Train Accuracy:  0.891, Validation Accuracy:  0.894, Loss:  0.126
    Epoch  18 Batch  129/269 - Train Accuracy:  0.885, Validation Accuracy:  0.893, Loss:  0.120
    Epoch  18 Batch  130/269 - Train Accuracy:  0.894, Validation Accuracy:  0.893, Loss:  0.127
    Epoch  18 Batch  131/269 - Train Accuracy:  0.877, Validation Accuracy:  0.900, Loss:  0.124
    Epoch  18 Batch  132/269 - Train Accuracy:  0.880, Validation Accuracy:  0.902, Loss:  0.127
    Epoch  18 Batch  133/269 - Train Accuracy:  0.896, Validation Accuracy:  0.903, Loss:  0.116
    Epoch  18 Batch  134/269 - Train Accuracy:  0.881, Validation Accuracy:  0.902, Loss:  0.126
    Epoch  18 Batch  135/269 - Train Accuracy:  0.888, Validation Accuracy:  0.904, Loss:  0.135
    Epoch  18 Batch  136/269 - Train Accuracy:  0.874, Validation Accuracy:  0.899, Loss:  0.128
    Epoch  18 Batch  137/269 - Train Accuracy:  0.881, Validation Accuracy:  0.900, Loss:  0.132
    Epoch  18 Batch  138/269 - Train Accuracy:  0.878, Validation Accuracy:  0.900, Loss:  0.120
    Epoch  18 Batch  139/269 - Train Accuracy:  0.892, Validation Accuracy:  0.906, Loss:  0.112
    Epoch  18 Batch  140/269 - Train Accuracy:  0.892, Validation Accuracy:  0.900, Loss:  0.131
    Epoch  18 Batch  141/269 - Train Accuracy:  0.891, Validation Accuracy:  0.900, Loss:  0.125
    Epoch  18 Batch  142/269 - Train Accuracy:  0.888, Validation Accuracy:  0.899, Loss:  0.118
    Epoch  18 Batch  143/269 - Train Accuracy:  0.902, Validation Accuracy:  0.896, Loss:  0.113
    Epoch  18 Batch  144/269 - Train Accuracy:  0.898, Validation Accuracy:  0.895, Loss:  0.112
    Epoch  18 Batch  145/269 - Train Accuracy:  0.892, Validation Accuracy:  0.898, Loss:  0.118
    Epoch  18 Batch  146/269 - Train Accuracy:  0.890, Validation Accuracy:  0.901, Loss:  0.122
    Epoch  18 Batch  147/269 - Train Accuracy:  0.893, Validation Accuracy:  0.898, Loss:  0.118
    Epoch  18 Batch  148/269 - Train Accuracy:  0.897, Validation Accuracy:  0.893, Loss:  0.124
    Epoch  18 Batch  149/269 - Train Accuracy:  0.883, Validation Accuracy:  0.892, Loss:  0.124
    Epoch  18 Batch  150/269 - Train Accuracy:  0.895, Validation Accuracy:  0.899, Loss:  0.123
    Epoch  18 Batch  151/269 - Train Accuracy:  0.898, Validation Accuracy:  0.897, Loss:  0.124
    Epoch  18 Batch  152/269 - Train Accuracy:  0.887, Validation Accuracy:  0.901, Loss:  0.122
    Epoch  18 Batch  153/269 - Train Accuracy:  0.906, Validation Accuracy:  0.897, Loss:  0.111
    Epoch  18 Batch  154/269 - Train Accuracy:  0.904, Validation Accuracy:  0.895, Loss:  0.121
    Epoch  18 Batch  155/269 - Train Accuracy:  0.895, Validation Accuracy:  0.892, Loss:  0.115
    Epoch  18 Batch  156/269 - Train Accuracy:  0.885, Validation Accuracy:  0.893, Loss:  0.132
    Epoch  18 Batch  157/269 - Train Accuracy:  0.882, Validation Accuracy:  0.899, Loss:  0.119
    Epoch  18 Batch  158/269 - Train Accuracy:  0.892, Validation Accuracy:  0.903, Loss:  0.122
    Epoch  18 Batch  159/269 - Train Accuracy:  0.891, Validation Accuracy:  0.901, Loss:  0.120
    Epoch  18 Batch  160/269 - Train Accuracy:  0.893, Validation Accuracy:  0.904, Loss:  0.116
    Epoch  18 Batch  161/269 - Train Accuracy:  0.896, Validation Accuracy:  0.900, Loss:  0.121
    Epoch  18 Batch  162/269 - Train Accuracy:  0.907, Validation Accuracy:  0.903, Loss:  0.125
    Epoch  18 Batch  163/269 - Train Accuracy:  0.905, Validation Accuracy:  0.900, Loss:  0.117
    Epoch  18 Batch  164/269 - Train Accuracy:  0.906, Validation Accuracy:  0.902, Loss:  0.118
    Epoch  18 Batch  165/269 - Train Accuracy:  0.895, Validation Accuracy:  0.904, Loss:  0.120
    Epoch  18 Batch  166/269 - Train Accuracy:  0.901, Validation Accuracy:  0.902, Loss:  0.117
    Epoch  18 Batch  167/269 - Train Accuracy:  0.886, Validation Accuracy:  0.887, Loss:  0.122
    Epoch  18 Batch  168/269 - Train Accuracy:  0.900, Validation Accuracy:  0.894, Loss:  0.127
    Epoch  18 Batch  169/269 - Train Accuracy:  0.872, Validation Accuracy:  0.892, Loss:  0.123
    Epoch  18 Batch  170/269 - Train Accuracy:  0.891, Validation Accuracy:  0.899, Loss:  0.116
    Epoch  18 Batch  171/269 - Train Accuracy:  0.890, Validation Accuracy:  0.881, Loss:  0.122
    Epoch  18 Batch  172/269 - Train Accuracy:  0.886, Validation Accuracy:  0.893, Loss:  0.132
    Epoch  18 Batch  173/269 - Train Accuracy:  0.895, Validation Accuracy:  0.894, Loss:  0.113
    Epoch  18 Batch  174/269 - Train Accuracy:  0.910, Validation Accuracy:  0.892, Loss:  0.118
    Epoch  18 Batch  175/269 - Train Accuracy:  0.872, Validation Accuracy:  0.892, Loss:  0.136
    Epoch  18 Batch  176/269 - Train Accuracy:  0.883, Validation Accuracy:  0.896, Loss:  0.132
    Epoch  18 Batch  177/269 - Train Accuracy:  0.898, Validation Accuracy:  0.897, Loss:  0.112
    Epoch  18 Batch  178/269 - Train Accuracy:  0.907, Validation Accuracy:  0.900, Loss:  0.119
    Epoch  18 Batch  179/269 - Train Accuracy:  0.891, Validation Accuracy:  0.895, Loss:  0.116
    Epoch  18 Batch  180/269 - Train Accuracy:  0.904, Validation Accuracy:  0.895, Loss:  0.114
    Epoch  18 Batch  181/269 - Train Accuracy:  0.895, Validation Accuracy:  0.900, Loss:  0.122
    Epoch  18 Batch  182/269 - Train Accuracy:  0.904, Validation Accuracy:  0.902, Loss:  0.117
    Epoch  18 Batch  183/269 - Train Accuracy:  0.913, Validation Accuracy:  0.898, Loss:  0.099
    Epoch  18 Batch  184/269 - Train Accuracy:  0.891, Validation Accuracy:  0.899, Loss:  0.121
    Epoch  18 Batch  185/269 - Train Accuracy:  0.908, Validation Accuracy:  0.901, Loss:  0.114
    Epoch  18 Batch  186/269 - Train Accuracy:  0.907, Validation Accuracy:  0.898, Loss:  0.117
    Epoch  18 Batch  187/269 - Train Accuracy:  0.903, Validation Accuracy:  0.897, Loss:  0.114
    Epoch  18 Batch  188/269 - Train Accuracy:  0.909, Validation Accuracy:  0.901, Loss:  0.113
    Epoch  18 Batch  189/269 - Train Accuracy:  0.904, Validation Accuracy:  0.897, Loss:  0.114
    Epoch  18 Batch  190/269 - Train Accuracy:  0.903, Validation Accuracy:  0.903, Loss:  0.115
    Epoch  18 Batch  191/269 - Train Accuracy:  0.882, Validation Accuracy:  0.903, Loss:  0.117
    Epoch  18 Batch  192/269 - Train Accuracy:  0.897, Validation Accuracy:  0.903, Loss:  0.114
    Epoch  18 Batch  193/269 - Train Accuracy:  0.900, Validation Accuracy:  0.905, Loss:  0.110
    Epoch  18 Batch  194/269 - Train Accuracy:  0.899, Validation Accuracy:  0.902, Loss:  0.119
    Epoch  18 Batch  195/269 - Train Accuracy:  0.892, Validation Accuracy:  0.898, Loss:  0.110
    Epoch  18 Batch  196/269 - Train Accuracy:  0.891, Validation Accuracy:  0.904, Loss:  0.120
    Epoch  18 Batch  197/269 - Train Accuracy:  0.893, Validation Accuracy:  0.905, Loss:  0.120
    Epoch  18 Batch  198/269 - Train Accuracy:  0.899, Validation Accuracy:  0.909, Loss:  0.123
    Epoch  18 Batch  199/269 - Train Accuracy:  0.894, Validation Accuracy:  0.901, Loss:  0.116
    Epoch  18 Batch  200/269 - Train Accuracy:  0.889, Validation Accuracy:  0.901, Loss:  0.124
    Epoch  18 Batch  201/269 - Train Accuracy:  0.883, Validation Accuracy:  0.890, Loss:  0.121
    Epoch  18 Batch  202/269 - Train Accuracy:  0.892, Validation Accuracy:  0.899, Loss:  0.123
    Epoch  18 Batch  203/269 - Train Accuracy:  0.896, Validation Accuracy:  0.898, Loss:  0.124
    Epoch  18 Batch  204/269 - Train Accuracy:  0.892, Validation Accuracy:  0.907, Loss:  0.123
    Epoch  18 Batch  205/269 - Train Accuracy:  0.898, Validation Accuracy:  0.905, Loss:  0.110
    Epoch  18 Batch  206/269 - Train Accuracy:  0.889, Validation Accuracy:  0.902, Loss:  0.127
    Epoch  18 Batch  207/269 - Train Accuracy:  0.887, Validation Accuracy:  0.892, Loss:  0.111
    Epoch  18 Batch  208/269 - Train Accuracy:  0.892, Validation Accuracy:  0.894, Loss:  0.127
    Epoch  18 Batch  209/269 - Train Accuracy:  0.905, Validation Accuracy:  0.898, Loss:  0.115
    Epoch  18 Batch  210/269 - Train Accuracy:  0.894, Validation Accuracy:  0.897, Loss:  0.118
    Epoch  18 Batch  211/269 - Train Accuracy:  0.902, Validation Accuracy:  0.902, Loss:  0.116
    Epoch  18 Batch  212/269 - Train Accuracy:  0.896, Validation Accuracy:  0.900, Loss:  0.126
    Epoch  18 Batch  213/269 - Train Accuracy:  0.893, Validation Accuracy:  0.904, Loss:  0.118
    Epoch  18 Batch  214/269 - Train Accuracy:  0.891, Validation Accuracy:  0.902, Loss:  0.117
    Epoch  18 Batch  215/269 - Train Accuracy:  0.902, Validation Accuracy:  0.901, Loss:  0.113
    Epoch  18 Batch  216/269 - Train Accuracy:  0.876, Validation Accuracy:  0.901, Loss:  0.141
    Epoch  18 Batch  217/269 - Train Accuracy:  0.878, Validation Accuracy:  0.900, Loss:  0.123
    Epoch  18 Batch  218/269 - Train Accuracy:  0.895, Validation Accuracy:  0.901, Loss:  0.121
    Epoch  18 Batch  219/269 - Train Accuracy:  0.903, Validation Accuracy:  0.897, Loss:  0.129
    Epoch  18 Batch  220/269 - Train Accuracy:  0.896, Validation Accuracy:  0.901, Loss:  0.113
    Epoch  18 Batch  221/269 - Train Accuracy:  0.889, Validation Accuracy:  0.903, Loss:  0.118
    Epoch  18 Batch  222/269 - Train Accuracy:  0.912, Validation Accuracy:  0.900, Loss:  0.117
    Epoch  18 Batch  223/269 - Train Accuracy:  0.887, Validation Accuracy:  0.899, Loss:  0.110
    Epoch  18 Batch  224/269 - Train Accuracy:  0.907, Validation Accuracy:  0.902, Loss:  0.127
    Epoch  18 Batch  225/269 - Train Accuracy:  0.889, Validation Accuracy:  0.902, Loss:  0.110
    Epoch  18 Batch  226/269 - Train Accuracy:  0.909, Validation Accuracy:  0.907, Loss:  0.121
    Epoch  18 Batch  227/269 - Train Accuracy:  0.904, Validation Accuracy:  0.902, Loss:  0.115
    Epoch  18 Batch  228/269 - Train Accuracy:  0.888, Validation Accuracy:  0.907, Loss:  0.110
    Epoch  18 Batch  229/269 - Train Accuracy:  0.890, Validation Accuracy:  0.900, Loss:  0.112
    Epoch  18 Batch  230/269 - Train Accuracy:  0.899, Validation Accuracy:  0.904, Loss:  0.119
    Epoch  18 Batch  231/269 - Train Accuracy:  0.891, Validation Accuracy:  0.902, Loss:  0.120
    Epoch  18 Batch  232/269 - Train Accuracy:  0.892, Validation Accuracy:  0.901, Loss:  0.116
    Epoch  18 Batch  233/269 - Train Accuracy:  0.912, Validation Accuracy:  0.903, Loss:  0.120
    Epoch  18 Batch  234/269 - Train Accuracy:  0.898, Validation Accuracy:  0.903, Loss:  0.116
    Epoch  18 Batch  235/269 - Train Accuracy:  0.907, Validation Accuracy:  0.900, Loss:  0.106
    Epoch  18 Batch  236/269 - Train Accuracy:  0.890, Validation Accuracy:  0.904, Loss:  0.113
    Epoch  18 Batch  237/269 - Train Accuracy:  0.906, Validation Accuracy:  0.897, Loss:  0.108
    Epoch  18 Batch  238/269 - Train Accuracy:  0.902, Validation Accuracy:  0.900, Loss:  0.117
    Epoch  18 Batch  239/269 - Train Accuracy:  0.906, Validation Accuracy:  0.902, Loss:  0.111
    Epoch  18 Batch  240/269 - Train Accuracy:  0.910, Validation Accuracy:  0.895, Loss:  0.100
    Epoch  18 Batch  241/269 - Train Accuracy:  0.899, Validation Accuracy:  0.909, Loss:  0.126
    Epoch  18 Batch  242/269 - Train Accuracy:  0.892, Validation Accuracy:  0.898, Loss:  0.115
    Epoch  18 Batch  243/269 - Train Accuracy:  0.903, Validation Accuracy:  0.905, Loss:  0.109
    Epoch  18 Batch  244/269 - Train Accuracy:  0.888, Validation Accuracy:  0.885, Loss:  0.116
    Epoch  18 Batch  245/269 - Train Accuracy:  0.876, Validation Accuracy:  0.900, Loss:  0.125
    Epoch  18 Batch  246/269 - Train Accuracy:  0.879, Validation Accuracy:  0.887, Loss:  0.120
    Epoch  18 Batch  247/269 - Train Accuracy:  0.898, Validation Accuracy:  0.893, Loss:  0.117
    Epoch  18 Batch  248/269 - Train Accuracy:  0.892, Validation Accuracy:  0.886, Loss:  0.107
    Epoch  18 Batch  249/269 - Train Accuracy:  0.912, Validation Accuracy:  0.893, Loss:  0.106
    Epoch  18 Batch  250/269 - Train Accuracy:  0.896, Validation Accuracy:  0.901, Loss:  0.115
    Epoch  18 Batch  251/269 - Train Accuracy:  0.906, Validation Accuracy:  0.898, Loss:  0.105
    Epoch  18 Batch  252/269 - Train Accuracy:  0.910, Validation Accuracy:  0.904, Loss:  0.106
    Epoch  18 Batch  253/269 - Train Accuracy:  0.889, Validation Accuracy:  0.898, Loss:  0.123
    Epoch  18 Batch  254/269 - Train Accuracy:  0.904, Validation Accuracy:  0.901, Loss:  0.115
    Epoch  18 Batch  255/269 - Train Accuracy:  0.916, Validation Accuracy:  0.902, Loss:  0.113
    Epoch  18 Batch  256/269 - Train Accuracy:  0.885, Validation Accuracy:  0.901, Loss:  0.116
    Epoch  18 Batch  257/269 - Train Accuracy:  0.876, Validation Accuracy:  0.903, Loss:  0.121
    Epoch  18 Batch  258/269 - Train Accuracy:  0.897, Validation Accuracy:  0.904, Loss:  0.117
    Epoch  18 Batch  259/269 - Train Accuracy:  0.908, Validation Accuracy:  0.903, Loss:  0.110
    Epoch  18 Batch  260/269 - Train Accuracy:  0.898, Validation Accuracy:  0.904, Loss:  0.125
    Epoch  18 Batch  261/269 - Train Accuracy:  0.893, Validation Accuracy:  0.900, Loss:  0.116
    Epoch  18 Batch  262/269 - Train Accuracy:  0.901, Validation Accuracy:  0.900, Loss:  0.113
    Epoch  18 Batch  263/269 - Train Accuracy:  0.886, Validation Accuracy:  0.906, Loss:  0.120
    Epoch  18 Batch  264/269 - Train Accuracy:  0.881, Validation Accuracy:  0.903, Loss:  0.123
    Epoch  18 Batch  265/269 - Train Accuracy:  0.899, Validation Accuracy:  0.899, Loss:  0.117
    Epoch  18 Batch  266/269 - Train Accuracy:  0.906, Validation Accuracy:  0.895, Loss:  0.112
    Epoch  18 Batch  267/269 - Train Accuracy:  0.890, Validation Accuracy:  0.891, Loss:  0.122
    Epoch  19 Batch    0/269 - Train Accuracy:  0.904, Validation Accuracy:  0.896, Loss:  0.124
    Epoch  19 Batch    1/269 - Train Accuracy:  0.909, Validation Accuracy:  0.902, Loss:  0.112
    Epoch  19 Batch    2/269 - Train Accuracy:  0.898, Validation Accuracy:  0.904, Loss:  0.114
    Epoch  19 Batch    3/269 - Train Accuracy:  0.907, Validation Accuracy:  0.898, Loss:  0.113
    Epoch  19 Batch    4/269 - Train Accuracy:  0.893, Validation Accuracy:  0.906, Loss:  0.121
    Epoch  19 Batch    5/269 - Train Accuracy:  0.890, Validation Accuracy:  0.900, Loss:  0.112
    Epoch  19 Batch    6/269 - Train Accuracy:  0.904, Validation Accuracy:  0.901, Loss:  0.115
    Epoch  19 Batch    7/269 - Train Accuracy:  0.906, Validation Accuracy:  0.904, Loss:  0.109
    Epoch  19 Batch    8/269 - Train Accuracy:  0.916, Validation Accuracy:  0.910, Loss:  0.116
    Epoch  19 Batch    9/269 - Train Accuracy:  0.899, Validation Accuracy:  0.902, Loss:  0.116
    Epoch  19 Batch   10/269 - Train Accuracy:  0.912, Validation Accuracy:  0.907, Loss:  0.108
    Epoch  19 Batch   11/269 - Train Accuracy:  0.900, Validation Accuracy:  0.906, Loss:  0.118
    Epoch  19 Batch   12/269 - Train Accuracy:  0.897, Validation Accuracy:  0.898, Loss:  0.121
    Epoch  19 Batch   13/269 - Train Accuracy:  0.893, Validation Accuracy:  0.897, Loss:  0.105
    Epoch  19 Batch   14/269 - Train Accuracy:  0.890, Validation Accuracy:  0.898, Loss:  0.111
    Epoch  19 Batch   15/269 - Train Accuracy:  0.910, Validation Accuracy:  0.904, Loss:  0.103
    Epoch  19 Batch   16/269 - Train Accuracy:  0.894, Validation Accuracy:  0.903, Loss:  0.113
    Epoch  19 Batch   17/269 - Train Accuracy:  0.905, Validation Accuracy:  0.903, Loss:  0.106
    Epoch  19 Batch   18/269 - Train Accuracy:  0.899, Validation Accuracy:  0.903, Loss:  0.118
    Epoch  19 Batch   19/269 - Train Accuracy:  0.915, Validation Accuracy:  0.894, Loss:  0.101
    Epoch  19 Batch   20/269 - Train Accuracy:  0.891, Validation Accuracy:  0.887, Loss:  0.111
    Epoch  19 Batch   21/269 - Train Accuracy:  0.882, Validation Accuracy:  0.900, Loss:  0.133
    Epoch  19 Batch   22/269 - Train Accuracy:  0.912, Validation Accuracy:  0.905, Loss:  0.105
    Epoch  19 Batch   23/269 - Train Accuracy:  0.902, Validation Accuracy:  0.907, Loss:  0.115
    Epoch  19 Batch   24/269 - Train Accuracy:  0.893, Validation Accuracy:  0.906, Loss:  0.110
    Epoch  19 Batch   25/269 - Train Accuracy:  0.884, Validation Accuracy:  0.903, Loss:  0.121
    Epoch  19 Batch   26/269 - Train Accuracy:  0.900, Validation Accuracy:  0.904, Loss:  0.109
    Epoch  19 Batch   27/269 - Train Accuracy:  0.896, Validation Accuracy:  0.907, Loss:  0.106
    Epoch  19 Batch   28/269 - Train Accuracy:  0.888, Validation Accuracy:  0.907, Loss:  0.127
    Epoch  19 Batch   29/269 - Train Accuracy:  0.903, Validation Accuracy:  0.902, Loss:  0.116
    Epoch  19 Batch   30/269 - Train Accuracy:  0.898, Validation Accuracy:  0.902, Loss:  0.106
    Epoch  19 Batch   31/269 - Train Accuracy:  0.906, Validation Accuracy:  0.902, Loss:  0.102
    Epoch  19 Batch   32/269 - Train Accuracy:  0.900, Validation Accuracy:  0.903, Loss:  0.106
    Epoch  19 Batch   33/269 - Train Accuracy:  0.901, Validation Accuracy:  0.898, Loss:  0.101
    Epoch  19 Batch   34/269 - Train Accuracy:  0.902, Validation Accuracy:  0.901, Loss:  0.107
    Epoch  19 Batch   35/269 - Train Accuracy:  0.895, Validation Accuracy:  0.899, Loss:  0.118
    Epoch  19 Batch   36/269 - Train Accuracy:  0.891, Validation Accuracy:  0.893, Loss:  0.105
    Epoch  19 Batch   37/269 - Train Accuracy:  0.907, Validation Accuracy:  0.905, Loss:  0.108
    Epoch  19 Batch   38/269 - Train Accuracy:  0.899, Validation Accuracy:  0.905, Loss:  0.105
    Epoch  19 Batch   39/269 - Train Accuracy:  0.909, Validation Accuracy:  0.900, Loss:  0.109
    Epoch  19 Batch   40/269 - Train Accuracy:  0.892, Validation Accuracy:  0.900, Loss:  0.116
    Epoch  19 Batch   41/269 - Train Accuracy:  0.895, Validation Accuracy:  0.899, Loss:  0.116
    Epoch  19 Batch   42/269 - Train Accuracy:  0.911, Validation Accuracy:  0.894, Loss:  0.097
    Epoch  19 Batch   43/269 - Train Accuracy:  0.916, Validation Accuracy:  0.898, Loss:  0.113
    Epoch  19 Batch   44/269 - Train Accuracy:  0.906, Validation Accuracy:  0.900, Loss:  0.106
    Epoch  19 Batch   45/269 - Train Accuracy:  0.905, Validation Accuracy:  0.897, Loss:  0.113
    Epoch  19 Batch   46/269 - Train Accuracy:  0.890, Validation Accuracy:  0.898, Loss:  0.108
    Epoch  19 Batch   47/269 - Train Accuracy:  0.908, Validation Accuracy:  0.895, Loss:  0.099
    Epoch  19 Batch   48/269 - Train Accuracy:  0.910, Validation Accuracy:  0.900, Loss:  0.104
    Epoch  19 Batch   49/269 - Train Accuracy:  0.902, Validation Accuracy:  0.901, Loss:  0.101
    Epoch  19 Batch   50/269 - Train Accuracy:  0.881, Validation Accuracy:  0.900, Loss:  0.119
    Epoch  19 Batch   51/269 - Train Accuracy:  0.909, Validation Accuracy:  0.901, Loss:  0.107
    Epoch  19 Batch   52/269 - Train Accuracy:  0.887, Validation Accuracy:  0.898, Loss:  0.099
    Epoch  19 Batch   53/269 - Train Accuracy:  0.896, Validation Accuracy:  0.901, Loss:  0.117
    Epoch  19 Batch   54/269 - Train Accuracy:  0.914, Validation Accuracy:  0.898, Loss:  0.102
    Epoch  19 Batch   55/269 - Train Accuracy:  0.903, Validation Accuracy:  0.905, Loss:  0.102
    Epoch  19 Batch   56/269 - Train Accuracy:  0.894, Validation Accuracy:  0.903, Loss:  0.105
    Epoch  19 Batch   57/269 - Train Accuracy:  0.892, Validation Accuracy:  0.908, Loss:  0.117
    Epoch  19 Batch   58/269 - Train Accuracy:  0.912, Validation Accuracy:  0.908, Loss:  0.103
    Epoch  19 Batch   59/269 - Train Accuracy:  0.920, Validation Accuracy:  0.909, Loss:  0.091
    Epoch  19 Batch   60/269 - Train Accuracy:  0.912, Validation Accuracy:  0.903, Loss:  0.102
    Epoch  19 Batch   61/269 - Train Accuracy:  0.911, Validation Accuracy:  0.903, Loss:  0.097
    Epoch  19 Batch   62/269 - Train Accuracy:  0.904, Validation Accuracy:  0.902, Loss:  0.103
    Epoch  19 Batch   63/269 - Train Accuracy:  0.903, Validation Accuracy:  0.901, Loss:  0.110
    Epoch  19 Batch   64/269 - Train Accuracy:  0.904, Validation Accuracy:  0.904, Loss:  0.106
    Epoch  19 Batch   65/269 - Train Accuracy:  0.905, Validation Accuracy:  0.908, Loss:  0.103
    Epoch  19 Batch   66/269 - Train Accuracy:  0.909, Validation Accuracy:  0.913, Loss:  0.106
    Epoch  19 Batch   67/269 - Train Accuracy:  0.902, Validation Accuracy:  0.910, Loss:  0.112
    Epoch  19 Batch   68/269 - Train Accuracy:  0.893, Validation Accuracy:  0.901, Loss:  0.116
    Epoch  19 Batch   69/269 - Train Accuracy:  0.883, Validation Accuracy:  0.901, Loss:  0.125
    Epoch  19 Batch   70/269 - Train Accuracy:  0.916, Validation Accuracy:  0.903, Loss:  0.107
    Epoch  19 Batch   71/269 - Train Accuracy:  0.902, Validation Accuracy:  0.907, Loss:  0.121
    Epoch  19 Batch   72/269 - Train Accuracy:  0.899, Validation Accuracy:  0.902, Loss:  0.115
    Epoch  19 Batch   73/269 - Train Accuracy:  0.901, Validation Accuracy:  0.908, Loss:  0.115
    Epoch  19 Batch   74/269 - Train Accuracy:  0.905, Validation Accuracy:  0.908, Loss:  0.107
    Epoch  19 Batch   75/269 - Train Accuracy:  0.910, Validation Accuracy:  0.911, Loss:  0.106
    Epoch  19 Batch   76/269 - Train Accuracy:  0.884, Validation Accuracy:  0.909, Loss:  0.106
    Epoch  19 Batch   77/269 - Train Accuracy:  0.907, Validation Accuracy:  0.904, Loss:  0.102
    Epoch  19 Batch   78/269 - Train Accuracy:  0.909, Validation Accuracy:  0.904, Loss:  0.104
    Epoch  19 Batch   79/269 - Train Accuracy:  0.899, Validation Accuracy:  0.905, Loss:  0.110
    Epoch  19 Batch   80/269 - Train Accuracy:  0.914, Validation Accuracy:  0.899, Loss:  0.103
    Epoch  19 Batch   81/269 - Train Accuracy:  0.892, Validation Accuracy:  0.899, Loss:  0.109
    Epoch  19 Batch   82/269 - Train Accuracy:  0.914, Validation Accuracy:  0.903, Loss:  0.098
    Epoch  19 Batch   83/269 - Train Accuracy:  0.898, Validation Accuracy:  0.903, Loss:  0.120
    Epoch  19 Batch   84/269 - Train Accuracy:  0.909, Validation Accuracy:  0.909, Loss:  0.105
    Epoch  19 Batch   85/269 - Train Accuracy:  0.893, Validation Accuracy:  0.903, Loss:  0.105
    Epoch  19 Batch   86/269 - Train Accuracy:  0.908, Validation Accuracy:  0.900, Loss:  0.105
    Epoch  19 Batch   87/269 - Train Accuracy:  0.896, Validation Accuracy:  0.905, Loss:  0.112
    Epoch  19 Batch   88/269 - Train Accuracy:  0.895, Validation Accuracy:  0.903, Loss:  0.109
    Epoch  19 Batch   89/269 - Train Accuracy:  0.908, Validation Accuracy:  0.905, Loss:  0.104
    Epoch  19 Batch   90/269 - Train Accuracy:  0.896, Validation Accuracy:  0.903, Loss:  0.112
    Epoch  19 Batch   91/269 - Train Accuracy:  0.913, Validation Accuracy:  0.904, Loss:  0.094
    Epoch  19 Batch   92/269 - Train Accuracy:  0.910, Validation Accuracy:  0.902, Loss:  0.102
    Epoch  19 Batch   93/269 - Train Accuracy:  0.915, Validation Accuracy:  0.911, Loss:  0.099
    Epoch  19 Batch   94/269 - Train Accuracy:  0.890, Validation Accuracy:  0.909, Loss:  0.114
    Epoch  19 Batch   95/269 - Train Accuracy:  0.912, Validation Accuracy:  0.906, Loss:  0.102
    Epoch  19 Batch   96/269 - Train Accuracy:  0.885, Validation Accuracy:  0.907, Loss:  0.112
    Epoch  19 Batch   97/269 - Train Accuracy:  0.914, Validation Accuracy:  0.908, Loss:  0.105
    Epoch  19 Batch   98/269 - Train Accuracy:  0.915, Validation Accuracy:  0.906, Loss:  0.105
    Epoch  19 Batch   99/269 - Train Accuracy:  0.902, Validation Accuracy:  0.909, Loss:  0.106
    Epoch  19 Batch  100/269 - Train Accuracy:  0.906, Validation Accuracy:  0.906, Loss:  0.107
    Epoch  19 Batch  101/269 - Train Accuracy:  0.894, Validation Accuracy:  0.905, Loss:  0.115
    Epoch  19 Batch  102/269 - Train Accuracy:  0.901, Validation Accuracy:  0.903, Loss:  0.101
    Epoch  19 Batch  103/269 - Train Accuracy:  0.911, Validation Accuracy:  0.897, Loss:  0.108
    Epoch  19 Batch  104/269 - Train Accuracy:  0.900, Validation Accuracy:  0.899, Loss:  0.103
    Epoch  19 Batch  105/269 - Train Accuracy:  0.900, Validation Accuracy:  0.903, Loss:  0.107
    Epoch  19 Batch  106/269 - Train Accuracy:  0.907, Validation Accuracy:  0.907, Loss:  0.097
    Epoch  19 Batch  107/269 - Train Accuracy:  0.914, Validation Accuracy:  0.907, Loss:  0.109
    Epoch  19 Batch  108/269 - Train Accuracy:  0.921, Validation Accuracy:  0.904, Loss:  0.103
    Epoch  19 Batch  109/269 - Train Accuracy:  0.887, Validation Accuracy:  0.901, Loss:  0.106
    Epoch  19 Batch  110/269 - Train Accuracy:  0.892, Validation Accuracy:  0.897, Loss:  0.096
    Epoch  19 Batch  111/269 - Train Accuracy:  0.904, Validation Accuracy:  0.895, Loss:  0.117
    Epoch  19 Batch  112/269 - Train Accuracy:  0.902, Validation Accuracy:  0.903, Loss:  0.109
    Epoch  19 Batch  113/269 - Train Accuracy:  0.898, Validation Accuracy:  0.901, Loss:  0.100
    Epoch  19 Batch  114/269 - Train Accuracy:  0.904, Validation Accuracy:  0.902, Loss:  0.106
    Epoch  19 Batch  115/269 - Train Accuracy:  0.895, Validation Accuracy:  0.898, Loss:  0.113
    Epoch  19 Batch  116/269 - Train Accuracy:  0.909, Validation Accuracy:  0.896, Loss:  0.107
    Epoch  19 Batch  117/269 - Train Accuracy:  0.896, Validation Accuracy:  0.899, Loss:  0.103
    Epoch  19 Batch  118/269 - Train Accuracy:  0.920, Validation Accuracy:  0.895, Loss:  0.095
    Epoch  19 Batch  119/269 - Train Accuracy:  0.896, Validation Accuracy:  0.898, Loss:  0.109
    Epoch  19 Batch  120/269 - Train Accuracy:  0.905, Validation Accuracy:  0.898, Loss:  0.104
    Epoch  19 Batch  121/269 - Train Accuracy:  0.907, Validation Accuracy:  0.906, Loss:  0.097
    Epoch  19 Batch  122/269 - Train Accuracy:  0.894, Validation Accuracy:  0.908, Loss:  0.103
    Epoch  19 Batch  123/269 - Train Accuracy:  0.905, Validation Accuracy:  0.905, Loss:  0.106
    Epoch  19 Batch  124/269 - Train Accuracy:  0.909, Validation Accuracy:  0.904, Loss:  0.099
    Epoch  19 Batch  125/269 - Train Accuracy:  0.912, Validation Accuracy:  0.899, Loss:  0.098
    Epoch  19 Batch  126/269 - Train Accuracy:  0.901, Validation Accuracy:  0.899, Loss:  0.100
    Epoch  19 Batch  127/269 - Train Accuracy:  0.903, Validation Accuracy:  0.898, Loss:  0.102
    Epoch  19 Batch  128/269 - Train Accuracy:  0.903, Validation Accuracy:  0.896, Loss:  0.105
    Epoch  19 Batch  129/269 - Train Accuracy:  0.902, Validation Accuracy:  0.896, Loss:  0.104
    Epoch  19 Batch  130/269 - Train Accuracy:  0.906, Validation Accuracy:  0.906, Loss:  0.109
    Epoch  19 Batch  131/269 - Train Accuracy:  0.882, Validation Accuracy:  0.904, Loss:  0.102
    Epoch  19 Batch  132/269 - Train Accuracy:  0.890, Validation Accuracy:  0.902, Loss:  0.112
    Epoch  19 Batch  133/269 - Train Accuracy:  0.906, Validation Accuracy:  0.906, Loss:  0.096
    Epoch  19 Batch  134/269 - Train Accuracy:  0.894, Validation Accuracy:  0.913, Loss:  0.107
    Epoch  19 Batch  135/269 - Train Accuracy:  0.909, Validation Accuracy:  0.904, Loss:  0.113
    Epoch  19 Batch  136/269 - Train Accuracy:  0.889, Validation Accuracy:  0.902, Loss:  0.116
    Epoch  19 Batch  137/269 - Train Accuracy:  0.887, Validation Accuracy:  0.905, Loss:  0.112
    Epoch  19 Batch  138/269 - Train Accuracy:  0.897, Validation Accuracy:  0.907, Loss:  0.103
    Epoch  19 Batch  139/269 - Train Accuracy:  0.905, Validation Accuracy:  0.909, Loss:  0.098
    Epoch  19 Batch  140/269 - Train Accuracy:  0.910, Validation Accuracy:  0.911, Loss:  0.114
    Epoch  19 Batch  141/269 - Train Accuracy:  0.902, Validation Accuracy:  0.908, Loss:  0.105
    Epoch  19 Batch  142/269 - Train Accuracy:  0.903, Validation Accuracy:  0.908, Loss:  0.098
    Epoch  19 Batch  143/269 - Train Accuracy:  0.909, Validation Accuracy:  0.910, Loss:  0.097
    Epoch  19 Batch  144/269 - Train Accuracy:  0.910, Validation Accuracy:  0.909, Loss:  0.086
    Epoch  19 Batch  145/269 - Train Accuracy:  0.905, Validation Accuracy:  0.904, Loss:  0.097
    Epoch  19 Batch  146/269 - Train Accuracy:  0.900, Validation Accuracy:  0.905, Loss:  0.100
    Epoch  19 Batch  147/269 - Train Accuracy:  0.909, Validation Accuracy:  0.905, Loss:  0.099
    Epoch  19 Batch  148/269 - Train Accuracy:  0.904, Validation Accuracy:  0.905, Loss:  0.102
    Epoch  19 Batch  149/269 - Train Accuracy:  0.890, Validation Accuracy:  0.903, Loss:  0.107
    Epoch  19 Batch  150/269 - Train Accuracy:  0.902, Validation Accuracy:  0.909, Loss:  0.102
    Epoch  19 Batch  151/269 - Train Accuracy:  0.904, Validation Accuracy:  0.907, Loss:  0.108
    Epoch  19 Batch  152/269 - Train Accuracy:  0.904, Validation Accuracy:  0.907, Loss:  0.102
    Epoch  19 Batch  153/269 - Train Accuracy:  0.913, Validation Accuracy:  0.909, Loss:  0.096
    Epoch  19 Batch  154/269 - Train Accuracy:  0.917, Validation Accuracy:  0.906, Loss:  0.098
    Epoch  19 Batch  155/269 - Train Accuracy:  0.897, Validation Accuracy:  0.906, Loss:  0.098
    Epoch  19 Batch  156/269 - Train Accuracy:  0.908, Validation Accuracy:  0.905, Loss:  0.103
    Epoch  19 Batch  157/269 - Train Accuracy:  0.895, Validation Accuracy:  0.904, Loss:  0.096
    Epoch  19 Batch  158/269 - Train Accuracy:  0.911, Validation Accuracy:  0.903, Loss:  0.098
    Epoch  19 Batch  159/269 - Train Accuracy:  0.903, Validation Accuracy:  0.909, Loss:  0.101
    Epoch  19 Batch  160/269 - Train Accuracy:  0.907, Validation Accuracy:  0.914, Loss:  0.097
    Epoch  19 Batch  161/269 - Train Accuracy:  0.912, Validation Accuracy:  0.913, Loss:  0.096
    Epoch  19 Batch  162/269 - Train Accuracy:  0.920, Validation Accuracy:  0.911, Loss:  0.102
    Epoch  19 Batch  163/269 - Train Accuracy:  0.914, Validation Accuracy:  0.909, Loss:  0.099
    Epoch  19 Batch  164/269 - Train Accuracy:  0.909, Validation Accuracy:  0.909, Loss:  0.095
    Epoch  19 Batch  165/269 - Train Accuracy:  0.908, Validation Accuracy:  0.908, Loss:  0.104
    Epoch  19 Batch  166/269 - Train Accuracy:  0.914, Validation Accuracy:  0.905, Loss:  0.095
    Epoch  19 Batch  167/269 - Train Accuracy:  0.913, Validation Accuracy:  0.908, Loss:  0.102
    Epoch  19 Batch  168/269 - Train Accuracy:  0.912, Validation Accuracy:  0.909, Loss:  0.098
    Epoch  19 Batch  169/269 - Train Accuracy:  0.907, Validation Accuracy:  0.909, Loss:  0.101
    Epoch  19 Batch  170/269 - Train Accuracy:  0.915, Validation Accuracy:  0.904, Loss:  0.101
    Epoch  19 Batch  171/269 - Train Accuracy:  0.917, Validation Accuracy:  0.902, Loss:  0.102
    Epoch  19 Batch  172/269 - Train Accuracy:  0.893, Validation Accuracy:  0.900, Loss:  0.107
    Epoch  19 Batch  173/269 - Train Accuracy:  0.913, Validation Accuracy:  0.907, Loss:  0.097
    Epoch  19 Batch  174/269 - Train Accuracy:  0.923, Validation Accuracy:  0.905, Loss:  0.097
    Epoch  19 Batch  175/269 - Train Accuracy:  0.900, Validation Accuracy:  0.910, Loss:  0.119
    Epoch  19 Batch  176/269 - Train Accuracy:  0.886, Validation Accuracy:  0.911, Loss:  0.109
    Epoch  19 Batch  177/269 - Train Accuracy:  0.910, Validation Accuracy:  0.909, Loss:  0.098
    Epoch  19 Batch  178/269 - Train Accuracy:  0.918, Validation Accuracy:  0.914, Loss:  0.096
    Epoch  19 Batch  179/269 - Train Accuracy:  0.911, Validation Accuracy:  0.913, Loss:  0.097
    Epoch  19 Batch  180/269 - Train Accuracy:  0.910, Validation Accuracy:  0.914, Loss:  0.093
    Epoch  19 Batch  181/269 - Train Accuracy:  0.906, Validation Accuracy:  0.908, Loss:  0.104
    Epoch  19 Batch  182/269 - Train Accuracy:  0.911, Validation Accuracy:  0.904, Loss:  0.102
    Epoch  19 Batch  183/269 - Train Accuracy:  0.928, Validation Accuracy:  0.912, Loss:  0.084
    Epoch  19 Batch  184/269 - Train Accuracy:  0.907, Validation Accuracy:  0.911, Loss:  0.103
    Epoch  19 Batch  185/269 - Train Accuracy:  0.923, Validation Accuracy:  0.908, Loss:  0.096
    Epoch  19 Batch  186/269 - Train Accuracy:  0.913, Validation Accuracy:  0.906, Loss:  0.094
    Epoch  19 Batch  187/269 - Train Accuracy:  0.910, Validation Accuracy:  0.904, Loss:  0.096
    Epoch  19 Batch  188/269 - Train Accuracy:  0.915, Validation Accuracy:  0.905, Loss:  0.100
    Epoch  19 Batch  189/269 - Train Accuracy:  0.913, Validation Accuracy:  0.903, Loss:  0.097
    Epoch  19 Batch  190/269 - Train Accuracy:  0.912, Validation Accuracy:  0.908, Loss:  0.097
    Epoch  19 Batch  191/269 - Train Accuracy:  0.894, Validation Accuracy:  0.912, Loss:  0.098
    Epoch  19 Batch  192/269 - Train Accuracy:  0.914, Validation Accuracy:  0.912, Loss:  0.098
    Epoch  19 Batch  193/269 - Train Accuracy:  0.914, Validation Accuracy:  0.910, Loss:  0.093
    Epoch  19 Batch  194/269 - Train Accuracy:  0.902, Validation Accuracy:  0.907, Loss:  0.098
    Epoch  19 Batch  195/269 - Train Accuracy:  0.911, Validation Accuracy:  0.905, Loss:  0.092
    Epoch  19 Batch  196/269 - Train Accuracy:  0.902, Validation Accuracy:  0.909, Loss:  0.096
    Epoch  19 Batch  197/269 - Train Accuracy:  0.903, Validation Accuracy:  0.909, Loss:  0.098
    Epoch  19 Batch  198/269 - Train Accuracy:  0.901, Validation Accuracy:  0.914, Loss:  0.103
    Epoch  19 Batch  199/269 - Train Accuracy:  0.913, Validation Accuracy:  0.916, Loss:  0.098
    Epoch  19 Batch  200/269 - Train Accuracy:  0.908, Validation Accuracy:  0.917, Loss:  0.100
    Epoch  19 Batch  201/269 - Train Accuracy:  0.907, Validation Accuracy:  0.913, Loss:  0.099
    Epoch  19 Batch  202/269 - Train Accuracy:  0.901, Validation Accuracy:  0.913, Loss:  0.100
    Epoch  19 Batch  203/269 - Train Accuracy:  0.908, Validation Accuracy:  0.909, Loss:  0.104
    Epoch  19 Batch  204/269 - Train Accuracy:  0.908, Validation Accuracy:  0.912, Loss:  0.099
    Epoch  19 Batch  205/269 - Train Accuracy:  0.915, Validation Accuracy:  0.914, Loss:  0.092
    Epoch  19 Batch  206/269 - Train Accuracy:  0.890, Validation Accuracy:  0.910, Loss:  0.104
    Epoch  19 Batch  207/269 - Train Accuracy:  0.899, Validation Accuracy:  0.912, Loss:  0.095
    Epoch  19 Batch  208/269 - Train Accuracy:  0.910, Validation Accuracy:  0.917, Loss:  0.101
    Epoch  19 Batch  209/269 - Train Accuracy:  0.911, Validation Accuracy:  0.916, Loss:  0.100
    Epoch  19 Batch  210/269 - Train Accuracy:  0.897, Validation Accuracy:  0.919, Loss:  0.100
    Epoch  19 Batch  211/269 - Train Accuracy:  0.913, Validation Accuracy:  0.916, Loss:  0.098
    Epoch  19 Batch  212/269 - Train Accuracy:  0.906, Validation Accuracy:  0.914, Loss:  0.104
    Epoch  19 Batch  213/269 - Train Accuracy:  0.905, Validation Accuracy:  0.907, Loss:  0.095
    Epoch  19 Batch  214/269 - Train Accuracy:  0.905, Validation Accuracy:  0.909, Loss:  0.100
    Epoch  19 Batch  215/269 - Train Accuracy:  0.907, Validation Accuracy:  0.909, Loss:  0.095
    Epoch  19 Batch  216/269 - Train Accuracy:  0.892, Validation Accuracy:  0.913, Loss:  0.119
    Epoch  19 Batch  217/269 - Train Accuracy:  0.908, Validation Accuracy:  0.913, Loss:  0.099
    Epoch  19 Batch  218/269 - Train Accuracy:  0.908, Validation Accuracy:  0.908, Loss:  0.095
    Epoch  19 Batch  219/269 - Train Accuracy:  0.918, Validation Accuracy:  0.913, Loss:  0.106
    Epoch  19 Batch  220/269 - Train Accuracy:  0.903, Validation Accuracy:  0.905, Loss:  0.096
    Epoch  19 Batch  221/269 - Train Accuracy:  0.899, Validation Accuracy:  0.900, Loss:  0.100
    Epoch  19 Batch  222/269 - Train Accuracy:  0.915, Validation Accuracy:  0.905, Loss:  0.094
    Epoch  19 Batch  223/269 - Train Accuracy:  0.910, Validation Accuracy:  0.913, Loss:  0.094
    Epoch  19 Batch  224/269 - Train Accuracy:  0.913, Validation Accuracy:  0.915, Loss:  0.106
    Epoch  19 Batch  225/269 - Train Accuracy:  0.908, Validation Accuracy:  0.914, Loss:  0.095
    Epoch  19 Batch  226/269 - Train Accuracy:  0.911, Validation Accuracy:  0.920, Loss:  0.101
    Epoch  19 Batch  227/269 - Train Accuracy:  0.921, Validation Accuracy:  0.911, Loss:  0.102
    Epoch  19 Batch  228/269 - Train Accuracy:  0.907, Validation Accuracy:  0.904, Loss:  0.097
    Epoch  19 Batch  229/269 - Train Accuracy:  0.902, Validation Accuracy:  0.905, Loss:  0.096
    Epoch  19 Batch  230/269 - Train Accuracy:  0.912, Validation Accuracy:  0.905, Loss:  0.101
    Epoch  19 Batch  231/269 - Train Accuracy:  0.902, Validation Accuracy:  0.910, Loss:  0.100
    Epoch  19 Batch  232/269 - Train Accuracy:  0.899, Validation Accuracy:  0.912, Loss:  0.099
    Epoch  19 Batch  233/269 - Train Accuracy:  0.919, Validation Accuracy:  0.904, Loss:  0.105
    Epoch  19 Batch  234/269 - Train Accuracy:  0.921, Validation Accuracy:  0.904, Loss:  0.097
    Epoch  19 Batch  235/269 - Train Accuracy:  0.919, Validation Accuracy:  0.903, Loss:  0.089
    Epoch  19 Batch  236/269 - Train Accuracy:  0.905, Validation Accuracy:  0.909, Loss:  0.092
    Epoch  19 Batch  237/269 - Train Accuracy:  0.916, Validation Accuracy:  0.915, Loss:  0.095
    Epoch  19 Batch  238/269 - Train Accuracy:  0.917, Validation Accuracy:  0.907, Loss:  0.094
    Epoch  19 Batch  239/269 - Train Accuracy:  0.916, Validation Accuracy:  0.910, Loss:  0.095
    Epoch  19 Batch  240/269 - Train Accuracy:  0.918, Validation Accuracy:  0.910, Loss:  0.086
    Epoch  19 Batch  241/269 - Train Accuracy:  0.903, Validation Accuracy:  0.911, Loss:  0.103
    Epoch  19 Batch  242/269 - Train Accuracy:  0.915, Validation Accuracy:  0.906, Loss:  0.094
    Epoch  19 Batch  243/269 - Train Accuracy:  0.915, Validation Accuracy:  0.908, Loss:  0.084
    Epoch  19 Batch  244/269 - Train Accuracy:  0.904, Validation Accuracy:  0.911, Loss:  0.095
    Epoch  19 Batch  245/269 - Train Accuracy:  0.896, Validation Accuracy:  0.905, Loss:  0.094
    Epoch  19 Batch  246/269 - Train Accuracy:  0.893, Validation Accuracy:  0.905, Loss:  0.100
    Epoch  19 Batch  247/269 - Train Accuracy:  0.910, Validation Accuracy:  0.905, Loss:  0.096
    Epoch  19 Batch  248/269 - Train Accuracy:  0.899, Validation Accuracy:  0.912, Loss:  0.089
    Epoch  19 Batch  249/269 - Train Accuracy:  0.915, Validation Accuracy:  0.911, Loss:  0.082
    Epoch  19 Batch  250/269 - Train Accuracy:  0.911, Validation Accuracy:  0.909, Loss:  0.094
    Epoch  19 Batch  251/269 - Train Accuracy:  0.927, Validation Accuracy:  0.906, Loss:  0.086
    Epoch  19 Batch  252/269 - Train Accuracy:  0.912, Validation Accuracy:  0.902, Loss:  0.086
    Epoch  19 Batch  253/269 - Train Accuracy:  0.895, Validation Accuracy:  0.910, Loss:  0.104
    Epoch  19 Batch  254/269 - Train Accuracy:  0.913, Validation Accuracy:  0.908, Loss:  0.093
    Epoch  19 Batch  255/269 - Train Accuracy:  0.927, Validation Accuracy:  0.907, Loss:  0.091
    Epoch  19 Batch  256/269 - Train Accuracy:  0.893, Validation Accuracy:  0.910, Loss:  0.098
    Epoch  19 Batch  257/269 - Train Accuracy:  0.900, Validation Accuracy:  0.911, Loss:  0.096
    Epoch  19 Batch  258/269 - Train Accuracy:  0.910, Validation Accuracy:  0.914, Loss:  0.099
    Epoch  19 Batch  259/269 - Train Accuracy:  0.919, Validation Accuracy:  0.909, Loss:  0.092
    Epoch  19 Batch  260/269 - Train Accuracy:  0.916, Validation Accuracy:  0.917, Loss:  0.106
    Epoch  19 Batch  261/269 - Train Accuracy:  0.908, Validation Accuracy:  0.917, Loss:  0.094
    Epoch  19 Batch  262/269 - Train Accuracy:  0.908, Validation Accuracy:  0.913, Loss:  0.094
    Epoch  19 Batch  263/269 - Train Accuracy:  0.903, Validation Accuracy:  0.910, Loss:  0.102
    Epoch  19 Batch  264/269 - Train Accuracy:  0.893, Validation Accuracy:  0.911, Loss:  0.104
    Epoch  19 Batch  265/269 - Train Accuracy:  0.909, Validation Accuracy:  0.913, Loss:  0.098
    Epoch  19 Batch  266/269 - Train Accuracy:  0.921, Validation Accuracy:  0.916, Loss:  0.092
    Epoch  19 Batch  267/269 - Train Accuracy:  0.915, Validation Accuracy:  0.919, Loss:  0.098
    Model Trained and Saved


### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
# Save parameters for checkpoint
helper.save_params(save_path)
```

# Checkpoint


```python
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    result = [vocab_to_int[word] if word in vocab_to_int.keys() else vocab_to_int['<UNK>'] for word in sentence.lower().split()]
    return result

```

    Tests Passed


## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'

translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
```

    Input
      Word Ids:      [207, 86, 61, 122, 132, 172, 107]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']

    Prediction
      Word Ids:      [326, 76, 334, 247, 107, 16, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      French Words: ['la', 'fruit', 'est', 'le', 'vieux', 'camion', '.', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']


## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset I'm using only has a vocabulary of 227 English words of the thousands that I use, I'm only going to see good results using these words.  For this project, I don't need a perfect translation. However, if we want to create a better translation model, we'll need better data.

We can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take me **days** to train, so it's important to have a GPU and the neural network is performing well on dataset we had.
