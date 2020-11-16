# Historical events detection
Project for the Knowledge Extraction and Information Retrieval exam. The PDF report can be found [here](/Make_History_Count.pdf).

The dataset contained in the `data/histo` folder is called HISTO. You can find the official repository [here](https://github.com/dhfbk/Histo).

Full .txt word embeddings are not provided; instead, a compressed representation of them is present in the `data/histo` folder. Take a look at the [README](/wordemb/README.md) in the `wordemb` folder for more information.

## Usage
Here we report the order one should follow when looking at this work:

1. [nerhisto.ipynb](/nerhisto.ipynb): sentence tagging on the Histo dataset using a custom built BiLSTM + CRF model for TensorFlow version > 2.
2. [embcomp.ipynb](/embcomp.ipynb): comparison of  glove.840B.300d (Common Crawl) and HistoGlove (Glove trained on historical words).
3. [wikievents.ipynb](/wikievents.ipynb): retrieval and classification of 2.000 Wikipedia pages (1.000 pertaining historical events, 1.000 of other topics) using a LSTM neural network.
4. [nerwiki.ipynb](/nerwiki.ipynb): sentence tagging on the custom wikipedia dataset built at the previous step. Comparison with results from spaCy. You can watch the output of last two `displacy.render` cells [here](/render.png).

## Description
Reported from the project assignment:
### Making History Count
>Event dection in text is a challenging task consisting in finding text portions reporting the description of a real event. Historical events are a specific type of events that are framed in a historical context, reporting facts, dates, historical figures and locations. One of the challenges is that the definition of historical event itself is problematic. Another challenge is due to the fact that, when dealing with corpora of historical documents, the language itself may change dealing to shifts in semantics and style. The project aims at addressing both the tasks of studying language variations and detecting historical events. In particular, for the first issue, the goal is to sistematically compare word embeddings pre-trained on a corpus of historical texts with word embeddings pre-trained on corpora of contemporary texts, such as such as word2vec, GloVe, or BERT with the goal of studying the shift in terms or relative word vector distances between pairs of words in the historical context and in the contemporary one. Moreover, the project aims at building specific classifiers for detecting the presence of an event in the text and, optionally, extract from the event description the most relevant components (e.g., dates, historical figures, locations).
