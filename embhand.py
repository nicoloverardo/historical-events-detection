import csv
from os import stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import spatial
from sklearn.decomposition import PCA

from tqdm import tqdm

class EmbeddingsHandler():

    @staticmethod
    def find_closest_embeddings(embeddings_dict, embedding, limit=6):
        if limit is None:
            return sorted(embeddings_dict.keys(), key=lambda w: spatial.distance.cosine(embeddings_dict[w], embedding))
        else:
            return sorted(embeddings_dict.keys(), key=lambda w: spatial.distance.cosine(embeddings_dict[w], embedding))[1:limit]

    @staticmethod
    def reduce_dim(data, random_state=42):
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(data)

    @staticmethod
    def load_glove(path, nrows=None):
        return pd.read_csv(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, na_values=None, keep_default_na=False, nrows=nrows)
    
    @staticmethod
    def vec(words, w):
        return words.loc[w].values

    @staticmethod
    def cosine_distance(emb1, emb2):
        return spatial.distance.cosine(emb1, emb2)

    @staticmethod
    def plot_words(data, start=0, end=100):
        Y = EmbeddingsHandler.reduce_dim(data)
        Y = Y[start:end]

        plt.scatter(Y[:, 0], Y[:, 1])

        for label, x, y in zip(data.index.values, Y[:, 0], Y[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
        plt.show()

    @staticmethod
    def to_dict(data, progress=False):
        idx = tqdm(range(data.shape[0])) if progress else range(data.shape[0])
        return {data.index[i]: data.iloc[i].values for i in idx}
    
    @staticmethod
    def reshape_vocab(vocab1, vocab2):
        common = (set(vocab1.keys())).intersection(set(vocab2.keys()))
        return {k: vocab1[k] for k in common}, {k: vocab2[k] for k in common}

    @staticmethod
    def rotate_emb(d1, d2):
        voc = set(d1.keys())

        A = np.array([d1[k] for k in voc])
        B = np.array([d2[k] for k in voc])

        R = scipy.linalg.orthogonal_procrustes(A, B)[0]

        if not np.linalg.det(R) > 0:
            raise ValueError("Not a proper rotation: determinant is > 0")

        return {k:v for k,v in zip(voc, np.dot(A, R))}
    
    @staticmethod
    def get_df(df):
        df_red = EmbeddingsHandler.reduce_dim(df)
        df_red = pd.DataFrame(df_red, columns=['x', 'y'])
        words = pd.DataFrame(df.index.values, columns=['word'])

        return pd.concat([words, df_red], axis=1)

    @staticmethod
    def plot_word_vectors(df1, df2, word):
        df_1 = EmbeddingsHandler.get_df(df1)
        df_2 = EmbeddingsHandler.get_df(df2)

        v1 = df_1.iloc[df_1[df_1["word"] == word].index.values[0], 1:].to_numpy()
        v2 = df_2.iloc[df_2[df_2["word"] == word].index.values[0], 1:].to_numpy()

        ax = plt.axes()
        ax.arrow(0.0, 0.0, v1[0], v1[1], head_width=0.2, head_length=0.2, label="glove", fc='k', ec='k')
        ax.arrow(0.0, 0.0, v2[0], v2[1], head_width=0.2, head_length=0.2, label="histo", fc='g', ec='g')
        plt.grid()

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        plt.show()