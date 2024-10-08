import argparse
import logging

import networkx as nx
import pandas as pd
import os
from timeit import default_timer as timer
from datetime import timedelta

from ema_workbench import IntegerParameter, ScalarOutcome, Model, RealParameter
from ema_workbench import perform_experiments
from ema_workbench import save_results
from ema_workbench.em_framework.evaluators import Samplers
from ema_workbench.em_framework.salib_samplers import get_SALib_problem

from cdlib import evaluation
from cdlib.classes import NodeClustering
import pickle

from microminer_eval.algorithm.main import partition

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

df = pd.read_csv(f"../results/{project_name}/call_graph.csv").reset_index() # The original dataframe has the from and to columns as indices
nx_call_graph = nx.from_pandas_edgelist(df, source='from', target='to', create_using=nx.Graph(), edge_attr='weight')


results = []
all_partitions = dict()


def convert_to_key(resolution, k):
    return f"resolution_{resolution:.9f}_k_{k}"



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

def model_function(resolution, k):
    partitions = partition({"resolution": resolution, "k_topics": k, "project": project_path})[0][0]

    s = convert_to_key(resolution, k)
    all_partitions[s] = partitions  # Store the partitions for later use
    n_clustering = NodeClustering(communities=list(partitions.values()), graph=nx_call_graph, overlap=True)
    modularity = evaluation.newman_girvan_modularity(nx_call_graph, n_clustering)
    ned_score = ned(partitions)
    density = evaluation.scaled_density(nx_call_graph, n_clustering)

    reference_class_set = set(list(nx_call_graph.nodes()))
    partitions_class_set = set([x for xs in partitions.values() for x in xs])
    diff_set = reference_class_set.difference(partitions_class_set)

    return {'n_partitions': float(len(partitions)),
            'modularity': modularity.score,  # Number of clusters/partitions and modularity index as metrics
            'noise_classes': float(len(diff_set)),  # Number of classes not included in any partition/cluster
            'ned': ned_score,
            'density': density.score,
            }


logging.info("Starting evaluation of grid of parameters...")

model = Model(project_name, function=model_function)
# specify uncertainties
model.uncertainties = [RealParameter('resolution', 0.1, 2),
                       IntegerParameter('k', 1, len(nx_call_graph.nodes)), ]

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