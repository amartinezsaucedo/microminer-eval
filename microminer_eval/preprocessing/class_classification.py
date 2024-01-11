import os
import logging
import numpy as np
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModel


class ClassClassifier:
    _project_path: str = None
    _classifier = None
    _classes: dict = {}

    def __init__(self, model_path: str, project_path: str):
        self._project_path = project_path
        self._classifier = load(model_path)
        self._load_classes()

    def _load_classes(self):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", resume_download=True)
        model = AutoModel.from_pretrained("microsoft/codebert-base", resume_download=True)
        logging.info("Tokenizer and model loaded")
        for code_file in os.listdir(self._project_path):
            index = 0
            embeddings = []
            if code_file.endswith(".java"):
                with open(f"{self._project_path}/{code_file}", encoding="ISO-8859-1", errors="ignore") as file:
                    code_tokens = tokenizer.tokenize(file.read())
                    while len(code_tokens) > index * 510:
                        code_tokens = code_tokens[index * 510:(index + 1) * 510]
                        tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
                        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
                        context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
                        embeddings.append(context_embeddings[0][0].tolist())
                        index += 1
                    matrix = np.matrix(embeddings)
                    mean_matrix = matrix.mean(0)
                    arr = np.array(mean_matrix[0]).flatten()
                    name = code_file.split(".")
                    name = name[0].strip()
                    self._classes[name] = arr.tolist()
        logging.info("Classes loaded")

    def classify(self):
        X = []
        failed = []
        for n in self._classes:
            if len(self._classes[n]) > 0:
                X.append(self._classes[n])
            else:
                failed.append(n)
        predictions = self._classifier.predict(X)
        classes_list = list(self._classes)
        classification = {
            "entity": [],
            "application": [],
            "utility": []
        }
        for index, prediction in enumerate(predictions):
            classification[prediction].append(f"{classes_list[index]}.java")
        logging.info(f"{len(X)} classes classified, {len(failed)} failed")
        for class_type, classes in classification.items():
            filename = f"{self._project_path}/{class_type}.txt"
            with open(filename, "w") as file:
                for class_name in classes:
                    file.write(f"{class_name}\n")
            logging.info(f"Output saved to {filename}")
