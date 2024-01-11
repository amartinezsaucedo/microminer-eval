import csv
import re
import nltk
import gensim.downloader as api
import numpy as np
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from microminer_eval.utils.file import read_file


class Preprocesser:
    _lemmatizer: WordNetLemmatizer
    _v2w_model: Word2Vec
    _stop_words = ['handle', 'cancel', "title", 'parent', "cell", "bean", 'loader', 'stage', 'pressed', 'dragged',
                   'view', 'box', 'initialize', "total", "view", "image", "icon", "offset", "node", "scene", 'duration',
                   'drawer', 'nav', 'load', 'data', 'is', 'empty', 'all', "static", "cascade", 'transaction',
                   'override',
                   "join", "one", "description", "strategy", "generation", "override", 'persistence', 'generated', "io",
                   "projection", "property", "commit", "dao", "this", 'style', 'menu', "begin", "column", "translate",
                   "on", "selected", 'name', "png", "logo", 'string', 'name', "table", "exception", "contains",
                   "filter",
                   "controller", "implement", "button", "session", "hibernate", "array", "org", "save", "clear",
                   "boolean", "init", "remove", "entity", "observable", "double", "length", "alert", "action", "field",
                   "bundle", "show", "root", "list", "index", "text", "return", "wait", "lower", "true", "false",
                   "java",
                   "util", "long", "collection", "interface", "layout", "value", "valid", "is""value", "type", "model",
                   "public", "private", "id", "error", "void", "not", "int", "float", "for", "set", "catch", "try",
                   "javafx", "import", "class", "com", "package", "if", "else", 'null', "no", "delete", "add", "edit",
                   "get", "new", "open", "close", "mouse", "event", "window", "throw"]
    _class_embbedings: dict = dict()
    _class_strings: dict = dict()
    _project_path: str
    _project_name: str

    def __init__(self, project_path: str, project_name: str):
        nltk.download('wordnet')
        self._lemmatizer = WordNetLemmatizer()
        self._v2w_model = api.load('word2vec-google-news-300')
        self._project_path = project_path
        self._project_name = project_name

    def get_lemma(self, word: str) -> str:
        return self._lemmatizer.lemmatize(word)

    def preprocess(self, typed_classes: list[str]) -> list[str]:
        errors = []
        for class_name in typed_classes:
            class_lines = read_file(f"{self._project_path}/{class_name}")
            strings = []
            for string in class_lines:
                sentence = " ".join(re.findall("[a-zA-Z]+", string))
                strings.extend([w for w in sentence.split() if w not in self._stop_words])
            no_strings = [re.findall('.[^A-Z]*', string) for string in strings]
            all_words = []
            for ns in no_strings:
                all_words += ns
            treated_words = []
            embeddings = []
            w = []
            for a in all_words:
                treated_words.append(self.get_lemma(a.lower()))
            for t in treated_words:
                if t in self._v2w_model and t not in self._stop_words and len(t) > 1:
                    w.append(t)
                    embeddings.append(self._v2w_model[t])
            self._class_strings[class_name] = w
            matrix = np.matrix(embeddings)
            if len(matrix) > 1:
                mean_matrix = matrix.mean(0)
                self._class_embbedings[class_name] = np.asarray(mean_matrix).reshape(-1)
            else:
                errors.append(class_name)
        return errors

    # Static distance
    def load_call_graph(self) -> list:
        distances = []
        with open(f"{self._project_path}/{self._project_name}_call_graph.csv", newline='') as graph:
            reader = csv.reader(graph, delimiter=',')
            for row in reader:
                class1 = row[0]
                class2 = row[1]
                if class1 in self._class_embbedings and class2 in self._class_strings:
                    distances.append([class1, class2, float(row[2])])
        return distances

    def calculate_embeddings_service(self, classes: list[str]) -> float:
        total = 0
        embeddings_service = 0
        for class_name in classes:
            embeddings_class = self._class_embbedings[class_name] * len(self._class_strings[class_name])
            total += len(self._class_strings[class_name])
            embeddings_service += embeddings_class
        return embeddings_service / total
