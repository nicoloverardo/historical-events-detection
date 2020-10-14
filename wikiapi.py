import html
import json
import urllib
import sparql
import wikipedia


class WikiWrapper():

    @staticmethod
    def download_pages_name(q):
        result = sparql.query('http://dbpedia.org/sparql', q)

        names = []
        for item in result.fetchall():
            row = sparql.unpack_row(item)
            names.append(row[0].split("/")[-1])
        
        return names
    
    @staticmethod
    def download_summary(x):
        try:
            return wikipedia.summary(x.Name)
        except:
            return ""
    
    @staticmethod
    def get_extract(x):
        q = r"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&continue=%7C%7C&titles=" + html.escape(x.Name) + r"&converttitles=1&exchars=1000&explaintext=1"

        try:
            with urllib.request.urlopen(q) as url:
                data = json.loads(url.read().decode())

                return list(data["query"]["pages"].values())[0]["extract"]
        except:
            return ""
