import argparse
import logging

import numpy as np
import pandas as pd

from microminer_eval.microminer import MicroMiner
from microminer_eval.preprocessing.preprocess import preprocess

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--project-name", help="Project name", required=True, type=str)
parser.add_argument("-f", "--project-path", help="Project path", required=True, type=str)
parser.add_argument("-m", "--model_path", help="Classifier path", type=str, default="data/classifier.joblib")

arguments = parser.parse_args()

project_name = arguments.project_name
project_path = arguments.project_path
model_path = arguments.model_path

preprocess(project_path, project_name, model_path)
microminer = MicroMiner(project_path, project_name, model_path)
"""Smaller resolutions recover smaller clusters and therefore a larger number of them, while, conversely, 
larger values recover clusters containing more data points. If resolution is less than 1, the algorithm favors larger 
communities. Greater than 1 favors smaller communities """
resolution = np.linspace(0, 5, 11)  # MicroMiner default = 0.5
"""Static relationship weight"""
alpha = np.linspace(0, 1, 11)  # MicroMiner default = 0.1
"""Semantic relationship weight. in the case of a system in which components are poorly named, β could be reduced in 
favour of α to reduce the dependence on the semantic analysis in favour of the static relationships/analysis. """
beta = 1 - alpha
m_fuzzy = np.linspace(1, 5, 9)  # MicroMiner default = 3, paper states 2 though
microservice_threshold = 9  # MicroMiner default = 9, expert defined
results = []

for r in resolution:
    for a in alpha:
        for m in m_fuzzy:
            microminer.configure(r, a, 1 - a, m, microservice_threshold)
            partitions = microminer.partition()
            results.append({"alpha": a, "beta": 1-a, "m": m, "resolution": r,
                            "microservice_threshold": microservice_threshold, "partitions": partitions})

pd.DataFrame(results).to_csv(f"{project_path}/{project_name}_results.csv")

