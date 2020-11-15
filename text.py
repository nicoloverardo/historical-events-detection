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
    
    @staticmethod
    def small_clean(x):
        x = x.encode("ascii", "ignore").decode()
        tokens = nltk.word_tokenize(x)
        words = [word for word in tokens if word.isalnum()]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [w for w in stripped if len(w) > 1]
    
        return " ".join(words)

    @staticmethod
    def convert_for_spacyviz(data):
        sp_preds = []
        for sent in data:
            ents = []
            str_sent = ""

            for word, lbl in sent:
                if lbl is not "O":
                    start = len(str_sent)
                    end = start + len(word)
                    ents.append({"start": start, "end": end, "label": lbl})

                str_sent = str_sent + word + " "
            
            ents = PreProcessing.merge_ents(ents) if len(ents) > 0 else ents 
        
            sp_preds.append({"text": str_sent.strip() + ".",
                             "ents": ents, 
                             "title": None})

        return sp_preds

    @staticmethod
    def merge_ents(ents):
        ents_ = []
        for i, e in enumerate(ents):
            if i != 0:
                if ents[i-1]["end"] + 1 == ents[i]["start"] and \
                   ents[i-1]["label"] == ents[i]["label"]:

                    ents_[-1]["end"] = ents[i]["end"]
                else:
                    ents_.append(ents[i])
            else:
                ents_.append(ents[i])
        
        return ents_
    
    @staticmethod
    def parse_iob_data(data, nlp):
        sentences, labels, sentence, tag = [], [], [], []

        for text in data:
            doc = nlp(text)

            for sent in doc.sents:
                for word in sent:
                    sentence.append(word.text)
                    if word.ent_type_ is not "":
                        tag.append(word.ent_type_)
                    else:
                        tag.append(word.ent_iob_)
            
                sentences.append(" ".join(sentence))
                labels.append(" ".join(tag))

                sentence, tag = [], []
        return sentences, labels