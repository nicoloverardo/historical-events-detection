import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import spatial
from sklearn.decomposition import PCA
from tqdm import tqdm

from text import PreProcessing


class EmbeddingsHandler():
    """
    Handles various operations for pre-trained
    embeddings txt files
    """
    
    @staticmethod
    def find_closest_embeddings(embeddings_dict, embedding, limit=6):
        if limit is None:
            return sorted(embeddings_dict.keys(),
                          key=lambda w: spatial.distance.cosine(
                              embeddings_dict[w], 
                              embedding))
        else:
            return sorted(embeddings_dict.keys(),
                          key=lambda w: spatial.distance.cosine(
                              embeddings_dict[w], 
                              embedding))[1:limit]

    @staticmethod
    def reduce_dim(data, random_state=42):
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(data)

    @staticmethod
    def load_glove(path, nrows=None):
        return pd.read_csv(path, sep=" ", index_col=0,
                           header=None, quoting=csv.QUOTE_NONE,
                           na_values=None, keep_default_na=False,
                           nrows=nrows)
    
    @staticmethod
    def vec(words, w):
        return words.loc[w].values
    
    @staticmethod
    def filter_indices(data, sort_indices=True):
        filtered_idxs = [PreProcessing.cleanText(idx) for idx in data.index.values]
        filtered_idxs = [i for i in filtered_idxs if i]

        df = data[data.index.isin(filtered_idxs)]

        if sort_indices:
            df.sort_index(inplace=True)

        return df
    
    @staticmethod
    def get_indices_intersection(df1, df2, sort_indices=True):
        inters = df1.index.intersection(df2.index)

        df1, df2 = (df1[df1.index.isin(inters)],
                    df2[df2.index.isin(inters)])

        if sort_indices:
            df1.sort_index(inplace=True)
            df2.sort_index(inplace=True)

        return (df1, df2)

    @staticmethod
    def cosine_distance(emb1, emb2):
        return spatial.distance.cosine(emb1, emb2)

    @staticmethod
    def plot_words(data, start=0, end=100):
        if not isinstance(data, list):
            data = [data]
            
        for j in data:
            Y = EmbeddingsHandler.reduce_dim(j)
            Y = Y[start:end]

            plt.scatter(Y[:, 0], Y[:, 1])

            zipped = zip(j.index.values, Y[:, 0], Y[:, 1])

            for label, x, y in zipped:
                plt.annotate(label, xy=(x, y),
                             xytext=(0, 0), textcoords="offset points")

        plt.show()

    @staticmethod
    def to_dict(data, progress=False):
        idx = tqdm(range(data.shape[0])) if progress else range(data.shape[0])
        return {data.index[i]: data.iloc[i].values for i in idx}
    
    @staticmethod
    def reshape_vocab(vocab1, vocab2):
        common = (set(vocab1.keys())).intersection(set(vocab2.keys()))

        dict1 = {k: vocab1[k] for k in common}
        dict2 = {k: vocab2[k] for k in common}

        return dict1, dict2 

    @staticmethod
    def rotate_emb(d1, d2):
        voc = set(d1.keys())

        A = np.array([d1[k] for k in voc])
        B = np.array([d2[k] for k in voc])

        R = scipy.linalg.orthogonal_procrustes(A, B)[0]

        if np.linalg.det(R) <= 0:
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
        ax = plt.axes()

        EmbeddingsHandler._draw_vector(df1, word, ax, "k")
        EmbeddingsHandler._draw_vector(df2, word, ax, "g")

        plt.grid()

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        plt.show()

    @staticmethod
    def _draw_vector(df, word, ax, color):
        df_red = EmbeddingsHandler.get_df(df)
        vec = df_red.iloc[df_red[df_red["word"] == word]
                            .index.values[0], 1:].to_numpy()

        ax.arrow(0.0, 0.0, vec[0], vec[1],
                 head_width=0.2, head_length=0.2,
                 fc=color, ec=color)
