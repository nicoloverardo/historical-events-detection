"""
Code taken and adapted from build_glove.py
and from build_vocab.py files of:

https://github.com/guillaumegenthial/tf_ner/
"""

from collections import Counter
from pathlib import Path

import numpy as np

class GloveVocab():

    @staticmethod
    def _create_words_tags_file(input, output, idx):
        sentence = ""
        with Path(input).open("r") as input_f:
            with Path(output).open("w+") as output_f:
                for l in input_f:
                    if l.strip():
                        words = l.split()
                        t = (words[idx])
                        sentence = sentence+t+" "
                    else:
                        output_f.write(sentence+'\n')
                        sentence = ""

    @staticmethod
    def create_words_file(path, output, kind="train"):
        GloveVocab._create_words_tags_file(path, output+"/"+kind+".words.txt", 0)        

    @staticmethod
    def create_tags_file(path, output, kind="train"):
        GloveVocab._create_words_tags_file(path, output+"/"+kind+".tags.txt", 5)          
    
    @staticmethod
    def build_vocab(data_paths, path_vocab_words='vocab.words.txt', mincount=1, path_vocab_chars='vocab.chars.txt', path_vocab_tags='vocab.tags.txt'):
        vocab_words = GloveVocab.create_vocab_words(data_paths, path_vocab_words=path_vocab_words, mincount=mincount)

        GloveVocab.create_vocab_chars(vocab_words, path_vocab_chars=path_vocab_chars)
        GloveVocab.create_vocab_tags(data_paths[0], path_vocab_tags=path_vocab_tags)

    @staticmethod
    def words(name):
        return '{}.words.txt'.format(name)

    @staticmethod
    def create_vocab_words(data_paths, path_vocab_words='vocab.words.txt', mincount=1):
        """
        Get Counter of words on all the data, filter by min count, save (1. Words)
        """

        print('Build vocab words (may take a while)')

        counter_words = Counter()

        for n in data_paths:
            with Path(GloveVocab.words(n)).open() as f:
                for line in f:
                    counter_words.update(line.strip().split())

        vocab_words = {w for w, c in counter_words.items() if c >= mincount}

        with Path(path_vocab_words).open('w') as f:
            for w in sorted(list(vocab_words)):
                f.write('{}\n'.format(w))

        print('- done. Kept {} out of {}'.format(
            len(vocab_words), len(counter_words)))

        return vocab_words

    @staticmethod
    def create_vocab_chars(vocab_words, path_vocab_chars='vocab.chars.txt'):
        """ 
        Get all the characters from the vocab words (2. Chars)
        """

        print('Build vocab chars')

        vocab_chars = set()

        for w in vocab_words:
            vocab_chars.update(w)

        with Path(path_vocab_chars).open('w') as f:
            for c in sorted(list(vocab_chars)):
                f.write('{}\n'.format(c))

        print('- done. Found {} chars'.format(len(vocab_chars)))

    @staticmethod
    def tags(name):
        return '{}.tags.txt'.format(name)

    @staticmethod
    def create_vocab_tags(path, path_vocab_tags='vocab.tags.txt'):
        """
        Get all tags from the training set (3. Tags)
        """

        print('Build vocab tags (may take a while)')

        vocab_tags = set()

        with Path(GloveVocab.tags(path)).open() as f:
            for line in f:
                vocab_tags.update(line.strip().split())

        with Path(path_vocab_tags).open('w') as f:
            for t in sorted(list(vocab_tags)):
                f.write('{}\n'.format(t))

        print('- done. Found {} tags.'.format(len(vocab_tags)))

    @staticmethod
    def build_glove(path_vocab_words="vocab.words.txt", path_glove_txt="glove.840B.300d.txt", output_name='glove.npz'):
        """
        Build an np.array from some glove file and some vocab file
        """

        # Load vocab
        with Path(path_vocab_words).open() as f:
            word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}

        size_vocab = len(word_to_idx)

        # Array of zeros
        embeddings = np.zeros((size_vocab, 300))

        # Get relevant glove vectors
        found = 0

        print('Reading GloVe file (may take a while)')

        with Path(path_glove_txt).open() as f:
            for line_idx, line in enumerate(f):
                if line_idx % 100000 == 0:
                    print('- At line {}'.format(line_idx))

                line = line.strip().split()

                if len(line) != 300 + 1:
                    continue

                word = line[0]
                embedding = line[1:]

                if word in word_to_idx:
                    found += 1
                    word_idx = word_to_idx[word]
                    embeddings[word_idx] = embedding

        print('- done. Found {} vectors for {} words'.format(found, size_vocab))

        # Save np.array to file
        np.savez_compressed(output_name, embeddings=embeddings)