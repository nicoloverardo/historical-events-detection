from pathlib import Path

import pickle
import numpy as np
from tensorflow.python.types.core import Value
from tf2crf import CRF
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

class BilstmCrf():
    def __init__(self, data_path=None, glove=None):
        self.data_path = data_path
        self.glove = glove
        self.batch_size = 64
        self.epochs = 25
        self.max_tokens = 20000
        self.output_sequence_length = 200

    def _get_data(self, path):
        with Path(path).open("r") as f:
            sentences, labels, sentence, tag = [], [], [], []

            for line in f:
                if line.strip():
                    splits = line.strip().split()

                    sentence.append(splits[0])
                    tag.append(splits[-1])
                else:
                    sentences.append(" ".join(sentence))
                    labels.append(" ".join(tag))

                    sentence, tag = [], []
        
            return sentences, labels
    
    def _get_vectorizer(self, data=None, settings=None):
        if settings is None:
            if data is None:
                raise ValueError("Not restoring vectorizer, data is needed.")

            vectorizer = TextVectorization(max_tokens=self.max_tokens,
                output_sequence_length=self.output_sequence_length, standardize=None)

            text_ds = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
            vectorizer.adapt(text_ds)
        else:
            c, w = settings

            vectorizer = TextVectorization(c['max_tokens'],
                output_sequence_length=c['output_sequence_length'], standardize=None)
            vectorizer.set_weights(w)

        return vectorizer

    def get_embeddings(self, voc=None, word_index=None, 
                       embedding_dim=300,
                       path=None, progress=True, 
                       save=True, save_name=None):
        """
        Taken and adapted from:
        https://github.com/guillaumegenthial/tf_ner/
        """

        if path is None:
            if isinstance(self.glove, str):
                path = self.glove
            else:
                raise TypeError("Provide valid glove path")

        if voc is None:
            voc = self.voc_x
        
        if word_index is None:
            word_index = self.word_index

        embeddings_index = {}
        with open(path) as f:
            for i, line in enumerate(f):
                if i % 100000 == 0 and progress:
                    print('- At line {}'.format(i))

                line = line.strip().split()

                if len(line) != 300 + 1:
                    continue

                word = line[0]
                coefs = " ".join(line[1:])
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        
        if progress:
            print("Found %s word vectors." % len(embeddings_index))

        num_tokens = len(voc) + 2
        hits = 0
        misses = 0

        embeddings = np.zeros((num_tokens, embedding_dim))
        items = tqdm(word_index.items()) if progress else word_index.items()
        for word, i in items:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings[i] = embedding_vector
                hits += 1
            else:
                misses += 1

        if progress:
            print("Converted %d words (%d misses)" % (hits, misses))

        if save:
            if save_name is None:
                np.savez_compressed(self.data_path + "gloveemb.npz", embeddings=embeddings)
            else:
                np.savez_compressed(self.data_path + save_name, embeddings=embeddings)
        
        return embeddings
    
    def create_model(self, input_size, num_labels, embeddings, lstm_size=128):
        crf = CRF()

        self.model = Sequential([
            Input(shape=(None,), dtype="int32"),
            Embedding(input_size, 300, 
                      embeddings_initializer=Constant(embeddings),
                      trainable=False),
            Bidirectional(LSTM(lstm_size, return_sequences=True)),
            Bidirectional(LSTM(lstm_size, return_sequences=True)),
            TimeDistributed(Dense(num_labels, activation=None)),
            crf
        ])
        
        self.model.compile(loss=crf.loss,
                           optimizer=Adam(),
                           metrics=crf.accuracy)
    
    def prepare_data(self):
        self.X_train, self.y_train = self._get_data(self.data_path + "train.txt")
        self.X_val, self.y_val = self._get_data(self.data_path + "dev.txt")
        self.X_test, self.y_test = self._get_data(self.data_path + "test.txt")

        self.vectorizer_x = self._get_vectorizer(self.X_train)
        self.vectorizer_y = self._get_vectorizer(self.y_train)

        self._set_voc_wordindex()

        self.X_train = self.vectorizer_x(np.array([[s] for s in self.X_train])).numpy()
        self.X_val = self.vectorizer_x(np.array([[s] for s in self.X_val])).numpy()
        self.X_test = self.vectorizer_x(np.array([[s] for s in self.X_test])).numpy()

        self.y_train = self.vectorizer_y(np.array([[s] for s in self.y_train])).numpy()
        self.y_val = self.vectorizer_y(np.array([[s] for s in self.y_val])).numpy()
        self.y_test = self.vectorizer_y(np.array([[s] for s in self.y_test])).numpy()

    def _set_voc_wordindex(self):
        self.voc_x = self.vectorizer_x.get_vocabulary()
        self.voc_y = self.vectorizer_y.get_vocabulary()
        self.word_index = dict(zip(self.voc_x, range(len(self.voc_x))))

    def train(self, embeddings=None):
        self.prepare_data()

        if embeddings is None:
            if isinstance(self.glove, str):
                embeddings = self.get_embeddings(self.voc_x, self.word_index)
            elif isinstance(self.glove, np.ndarray):
                embeddings = self.glove
            else:
                raise TypeError("Glove path/matrix not valid")

        self.create_model(input_size=len(self.voc_x) + 2,
                          num_labels=len(self.voc_y),
                          embeddings=embeddings)

        es = EarlyStopping(monitor='loss', verbose=1,
                           mode='min', patience = 2, min_delta=0.1)

        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size, 
                       epochs=self.epochs,
                       validation_data=(self.X_val, self.y_val), 
                       callbacks=[es])

    def evaluate(self, data=None, labels=None):
        if data is None:
            data = self.X_test

        if labels is None:
            labels = self.y_test
        
        _, test_acc = self.model.evaluate(data, labels)

        print('Test Accuracy: {}'.format(test_acc))

    
    def _pretty_print(self, preds):
        """
        Taken and adapted from:
        https://github.com/guillaumegenthial/tf_ner/
        """

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

    def predict(self, data, print=True):
        x = self.vectorizer_x(np.array(data)).numpy()

        preds = self.model.predict(x)
        y_index = dict(zip(range(len(self.voc_y)), self.voc_y))

        data_split = [line.strip().split() for line in data]
        plist = [[y_index[int(p)] for p in pred] for pred in preds]

        result = [list(zip(data_split[i], plist[i])) for i in range(len(plist))]

        if print:
            self._pretty_print(result)

        return result
    
    def save(self, path=None):
        if path is None:
            path = self.data_path

        Path(path).mkdir(parents=True, exist_ok=True)

        with Path(path, "vectorizers.pkl").open("wb") as f:
            pickle.dump((self.vectorizer_x.get_weights(),
                         self.vectorizer_x.get_config(), 
                         self.vectorizer_y.get_config(), 
                         self.vectorizer_y.get_weights()), f)

        self.model.save_weights(path)

    def restore_model(self, embeddings, path):
        with Path(path, "vectorizers.pkl").open("rb") as f:
            wx, cx, cy, wy = pickle.load(f)

        self.vectorizer_x = self._get_vectorizer(settings=(cx, wx))
        self.vectorizer_y = self._get_vectorizer(settings=(cy, wy))
        
        self._set_voc_wordindex()

        self.create_model(input_size=len(self.voc_x) + 2,
                          num_labels=len(self.voc_y),
                          embeddings=embeddings)

        self.model.load_weights(path)
