import nltk
import string

class PreProcessing():
    @staticmethod
    def cleanText(x):
        x = x.encode("ascii", "ignore").decode()
        tokens = nltk.word_tokenize(x)
        tokens = [w.lower() for w in tokens]
        words = [word for word in tokens if word.isalpha()]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(string.punctuation)
        words = [w for w in stripped if w.isalpha() and not w in stopwords]
        words = [w for w in words if len(w) > 1 and w.isalpha()]
    
        return " ".join(words)  