import argparse
import logging

import numpy as np
import os
from timeit import default_timer as timer
from datetime import timedelta

from microminer_eval.microminer import MicroMiner
from microminer_eval.preprocessing.preprocess import preprocess

from ema_workbench import IntegerParameter, ScalarOutcome, Model
from ema_workbench import perform_experiments
from ema_workbench import save_results
from ema_workbench.em_framework.evaluators import Samplers
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from cdlib import evaluation
from cdlib.classes import NodeClustering
import pickle

start = timer()

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--project-name", help="Project name", required=True, type=str)
parser.add_argument("-f", "--project-path", help="Project path", required=True, type=str)
parser.add_argument("-m", "--model_path", help="Classifier path", type=str, default="data/classifier.joblib")

arguments = parser.parse_args()

project_name = arguments.project_name
project_path = arguments.project_path
model_path = arguments.model_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

call_graph_df = preprocess(project_path, project_name, model_path)  # The call graph is actually a dataframe
nx_call_graph = call_graph_df.as_nx_graph()
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
all_partitions = dict()


def convert_to_key(resolution, alpha, mfuzzy, microservice_threshold):
    return 'resolution_' + str(resolution) + '_alpha_' + str(alpha) + '_mfuzzy_' + str(mfuzzy) + '_mthreshold_' + str(
        microservice_threshold)


def ned(partitions):
    ned_sum = 0
    class_len = 0
    for cluster in partitions.values():
        size = len(cluster)
        class_len += size
        if 5 <= size <= 20:
            ned_sum += size

    ned_score = 1
    if class_len > 0 and ned_sum > 0:
        ned_score = ned_score - (ned_sum / class_len)

    return round(ned_score, 3)


def model_function(resolution, alpha, mfuzzy, microservice_threshold):
    beta = 1 - alpha
    microminer.configure(resolution / 100.0, alpha / 100.0, beta / 100.0, mfuzzy, microservice_threshold)
    partitions = microminer.partition()  # This is a dictionary with the (overlapping) clusters

    s = convert_to_key(resolution, alpha, mfuzzy, microservice_threshold)
    all_partitions[s] = partitions  # Store the partitions for later use
    n_clustering = NodeClustering(communities=list(partitions.values()), graph=nx_call_graph, overlap=True)
    modularity = evaluation.newman_girvan_modularity(nx_call_graph, n_clustering)
    ned_score = ned(partitions)
    density = evaluation.scaled_density(nx_call_graph, n_clustering)

    reference_class_set = set(list(nx_call_graph.nodes()))
    partitions_class_set = set([x for xs in partitions.values() for x in xs])
    diff_set = reference_class_set.difference(partitions_class_set)

    return {'n_partitions': len(partitions),
            'modularity': modularity.score,  # Number of clusters/partitions and modularity index as metrics
            'ned': ned_score,
            'density': density.score,
            'noise_classes': len(diff_set)  # Number of classes not included in any partition/cluster
            }


logging.info("Starting evaluation of grid of parameters...")

model = Model(project_name, function=model_function)
# specify uncertainties
model.uncertainties = [IntegerParameter('resolution', 1, 100),
                       IntegerParameter('alpha', 1, 100),
                       IntegerParameter('mfuzzy', 2, 10),
                       IntegerParameter('microservice_threshold', 2, 10)]

# specify outcomes
model.outcomes = [ScalarOutcome('n_partitions'),
                  ScalarOutcome('modularity'),
                  ScalarOutcome('ned'),
                  ScalarOutcome('density'),
                  ScalarOutcome('noise_classes')]

n_scenarios = 128
results = perform_experiments(models=model, scenarios=n_scenarios, uncertainty_sampling=Samplers.SOBOL)
filename = project_path + '/' + project_name + '_' + str(n_scenarios) + 'scenarios_nopolicies_sobol'  # .tar.gz'
save_results(results, filename + '.tar.gz')

# For Sobol analysis (later on)
uncertainties_problem = get_SALib_problem(model.uncertainties)
with open(filename + '_model.pkl', 'wb') as output:
    pickle.dump(uncertainties_problem, output)
with open(filename + '_partitions.pkl', 'wb') as output:
    pickle.dump(all_partitions, output)
with open(filename + '_graph.pkl', 'wb') as output:
    pickle.dump(nx_call_graph, output)

end = timer()
logging.info(str(timedelta(seconds=end - start)) + " secs. done")
