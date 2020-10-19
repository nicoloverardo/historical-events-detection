"""
GloVe Embeddings + bi-LSTM + CRF.
Code taken and adapted from:

https://github.com/guillaumegenthial/tf_ner/
"""

import functools
from pathlib import Path

import numpy as np
import json
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator
from tf_metrics import precision, recall, f1

class BiLSTM():

    @staticmethod
    def parse_fn(line_words, line_tags):
        # Encode in Bytes for TF
        words = [w.encode() for w in line_words.strip().split()]
        tags = [t.encode() for t in line_tags.strip().split()]
        assert len(words) == len(tags), "Words and tags lengths don't match"
        return (words, len(words)), tags

    @staticmethod
    def generator_fn(words, tags):
        with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
            for line_words, line_tags in zip(f_words, f_tags):
                yield BiLSTM.parse_fn(line_words, line_tags)

    @staticmethod
    def input_fn(words, tags, params=None, shuffle_and_repeat=False):
        params = params if params is not None else {}
        shapes = (([None], ()), [None])
        types = ((tf.string, tf.int32), tf.string)
        defaults = (('<pad>', 0), 'O')

        dataset = tf.data.Dataset.from_generator(
            functools.partial(BiLSTM.generator_fn, words, tags),
            output_shapes=shapes, output_types=types)

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

        dataset = (dataset
                   .padded_batch(params.get('batch_size', 20), shapes, defaults)
                   .prefetch(1))
        return dataset

    @staticmethod
    def model_fn(features, labels, mode, params):
        # For serving, features are a bit different
        if isinstance(features, dict):
            features = features['words'], features['nwords']

        # Read vocabs and inputs
        dropout = params['dropout']
        words, nwords = features
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        vocab_words = tf.contrib.lookup.index_table_from_file(
            params['words'], num_oov_buckets=params['num_oov_buckets'])
        with Path(params['tags']).open() as f:
            indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
            num_tags = len(indices) + 1

        # Word Embeddings
        word_ids = vocab_words.lookup(words)
        glove = np.load(params['glove'])['embeddings']  # np.array
        variable = np.vstack([glove, [[0.]*params['dim']]])
        variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
        embeddings = tf.nn.embedding_lookup(variable, word_ids)
        embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

        # LSTM
        t = tf.transpose(embeddings, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.layers.dropout(output, rate=dropout, training=training)

        # CRF
        logits = tf.layers.dense(output, num_tags)
        #crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
        crf_params = tf.compat.v1.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
        pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # Predictions
            reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
                params['tags'])
            pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
            predictions = {
                'pred_ids': pred_ids,
                'tags': pred_strings
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            # Loss
            vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
            tags = vocab_tags.lookup(labels)
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                logits, tags, nwords, crf_params)
            loss = tf.reduce_mean(-log_likelihood)

            # Metrics
            weights = tf.sequence_mask(nwords)
            metrics = {
                #'acc': tf.metrics.accuracy(tags, pred_ids, weights),
                'acc': tf.compat.v1.metrics.accuracy(tags, pred_ids, weights),
                'precision': precision(tags, pred_ids, num_tags, indices, weights),
                'recall': recall(tags, pred_ids, num_tags, indices, weights),
                'f1': f1(tags, pred_ids, num_tags, indices, weights),
            }
            for metric_name, op in metrics.items():
                #tf.summary.scalar(metric_name, op[1])
                tf.compat.v1.summary.scalar(metric_name, op[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            elif mode == tf.estimator.ModeKeys.TRAIN:
                #train_op = tf.train.AdamOptimizer().minimize(
                #    loss, global_step=tf.train.get_or_create_global_step())
                train_op = tf.compat.v1.train.AdamOptimizer().minimize(
                    loss, global_step=tf.compat.v1.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)
    
    @staticmethod
    def write_predictions(name, estimator, datadir, path):
        Path(path+'/score').mkdir(parents=True, exist_ok=True)
        with Path(path+'/score/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(BiLSTM.input_fn, BiLSTM.fwords(name, datadir), BiLSTM.ftags(name, datadir))
            golds_gen = BiLSTM.generator_fn(BiLSTM.fwords(name, datadir), BiLSTM.ftags(name, datadir))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    @staticmethod
    def fwords(name, datadir):
        return str(Path(datadir, '{}.words.txt'.format(name)))

    @staticmethod
    def ftags(name, datadir):
        return str(Path(datadir, '{}.tags.txt'.format(name)))

    @staticmethod
    def pretty_print(preds):
        for text in preds:
            words = [w[0] for w in text]
            ps = [p[1] for p in text]
            lengths = [max(len(w), len(p)) for w, p in zip(words, ps)]
            padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
            padded_preds = [p+ (l - len(p)) * ' ' for p, l in zip(ps, lengths)]
            print('words: {}'.format(' '.join(padded_words)))
            print('preds: {}'.format(' '.join(padded_preds)))

            if len(preds) > 1:
                print("\n")

    @staticmethod
    def predict_input_fn(line):
        # Words
        words = [w.encode() for w in line.strip().split()]
        nwords = len(words)

        # Wrapping in Tensors
        words = tf.constant([words], dtype=tf.string)
        nwords = tf.constant([nwords], dtype=tf.int32)

        return (words, nwords), None
    
    @staticmethod
    def predict(lines, params, datadir, modeldir):
        with Path(params).open() as f:
            params = json.load(f)

        params['words'] = str(Path(datadir, 'vocab.words.txt'))
        params['chars'] = str(Path(datadir, 'vocab.chars.txt'))
        params['tags'] = str(Path(datadir, 'vocab.tags.txt'))
        params['glove'] = str(Path(datadir, 'glove.npz'))

        estimator = tf.estimator.Estimator(BiLSTM.model_fn, modeldir, params=params)
        return [BiLSTM._do_prediction(line, estimator) for line in lines]

    @staticmethod
    def _do_prediction(line, estimator):
        predict_inpf = functools.partial(BiLSTM.predict_input_fn, line)

        result = []
        for pred in estimator.predict(predict_inpf):
            preds = [p.decode() for p in pred['tags']]
            result = list(zip(line.strip().split(), preds))
            break
        return result