import networkx as nx
from ema_workbench import load_results
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from cdlib import evaluation
from cdlib.classes import NodeClustering
import pickle
from cdlib import algorithms
import scipy.spatial as spatial
import numpy as np
import os



def compute_parameter_distances(experiments_df, columns=None, parameters=[], metric='cosine', scaling=True):
    scaler = None
    data = experiments_df[parameters].values
    scaled_data = data
    # print(scaled_data.shape)
    if scaling:
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

    dist = pdist(scaled_data, metric=metric)
    if columns is not None:
        dist_df = pd.DataFrame(squareform(dist), columns=columns, index=columns)
    else:
        dist_df = pd.DataFrame(squareform(dist))
    # print(dist_df.shape)
    return dist_df, scaler

# Omega index indicates the similarity between two partitions
# If omega = 1, the two partitions are identical (distance = 0), and omega = 0 (distance = 1) is the opposite case
# Thus, omega works as a similarity index
def compute_omega_index(partition_i, partition_j, graph, distance=False):
    clustering_i = NodeClustering(communities=list(partition_i.values()), graph=graph, overlap=True)
    clustering_j = NodeClustering(communities=list(partition_j.values()), graph=graph, overlap=True)
    if distance:
        return 1 - evaluation.omega(clustering_i, clustering_j).score
    else:
        return evaluation.omega(clustering_i, clustering_j).score

def get_noise_classes(partition, graph):
    reference_class_set = set(list(graph.nodes()))
    partition_class_set = set([x for xs in partition.values() for x in xs])
    difference_set = reference_class_set.difference(partition_class_set)
    return difference_set

def update_partition_with_noise(partition, graph):
    noise_classes = get_noise_classes(partition, graph)
    if len(noise_classes) > 0:
        partition[-1] = list(noise_classes) # -1 is the key for the noise classes
    
    return partition

def check_experiments(distance_df, p_step, partitions_dict, sim_threshold=0.95, binary=True):
    count_experiments = []
    for exp in distance_df.columns:
        # Get experiments being close enough to target exp
        indices = distance_df.loc[distance_df[exp] <= p_step].index
        ref = partitions_dict[exp]
        ref = update_partition_with_noise(ref, java_graph)
        # Discard experiments being dissimlar to target exp
        selected_indices = []
        all_valid_indices = True
        for idx in indices:
            partition_i = partitions_dict[idx]
            partition_i = update_partition_with_noise(partition_i, java_graph)
            if compute_omega_index(ref, partition_i, java_graph) >= sim_threshold:
                selected_indices.append(idx)
            else:
                all_valid_indices = False
                break
        if all_valid_indices:
            if binary:      
                count_experiments.append(1)
            else:
                count_experiments.append(len(selected_indices))
        else:
            #print("\t", exp, "indices so far (but discarded):", len(selected_indices))
            count_experiments.append(0)
    return count_experiments


# Generate code for a binary search of a suitable parameter value within a continuous interval. 
# The stopping criterion for the search is  given by an error threshold. 
# The suitability of the parameter value is given by another function.
def binary_search(distance_df, partitions_dict, threshold=0.95, max_trials=25, freq=5, 
                  interval=(0.0,1.0), tolerance=0.0005, verbose=True):
    if verbose:
        print("threshold=", threshold, " max_trials=", max_trials, " interval=", interval)
    
    explored_exps = dict()
    left = interval[0]
    right = interval[1]
    n = 1
    best_f = None
    best_mid = None
    mid = None
    for n in tqdm(range(1, max_trials+1)):
        if (abs(right-left) >= tolerance):
            mid = (left + right) / 2
            if n % freq == 0:
                if verbose:
                    print(n, "r=", mid, "...")
            f = check_experiments(distance_df, mid, partitions_dict, sim_threshold=threshold)
            f_sum = sum(f)
            if f_sum == 0: # Mid point is not suitable yet
                right = mid
            else:
                best_f = f
                best_mid = mid
                if verbose:
                    print(n,"r=", mid, ", satisfying distance found!", "count=", f_sum, "coverage=", f_sum/len(partitions_dict))
                    selected_exps = [col for idx, col in zip(best_f, distance_df.columns) if idx == 1]
                    for exp in selected_exps:
                        explored_exps[exp] = mid
                    # print(selected_exps)
                left = mid
        else:
            if verbose:
                print("Stopping criterion reached:", tolerance, " r=", mid)
            break

    if best_f is None:
        if verbose:
            print("No suitable distance found")
        return mid, best_f, []
    else:
        selected_exps = [col for idx, col in zip(best_f, distance_df.columns) if idx == 1]
        return best_mid, sum(best_f)/len(partitions_dict), selected_exps, explored_exps


#--------------------------------------------------------

GRAPH_FILENAME = "../jpetstore/jpetstore_128scenarios_nopolicies_sobol_graph.pkl"
#GRAPH_FILENAME = "./cargo/cargo_128scenarios_nopolicies_sobol_graph.pkl"

MODEL_FILENAME = '../jpetstore/jpetstore_128scenarios_nopolicies_sobol' #.tar.gz'
#MODEL_FILENAME = './cargo/cargo_128scenarios_nopolicies_sobol' #.tar.gz'

PARTITIONS_FILENAME = "../jpetstore/jpetstore_128scenarios_nopolicies_sobol_partitions.pkl"
#PARTITIONS_FILENAME = "./cargo/cargo_128scenarios_nopolicies_sobol_partitions.pkl"

DISTANCE_FILENAME = "../jpetstore/jpetstore_parameter_distances.csv"
#DISTANCE_FILENAME = "./cargo/cargo_parameter_distances.csv"

STABLE_SOLUTIONS_FILENAME = "../jpetstore/jpetstore_stable_solutions.pkl"
#STABLE_SOLUTIONS_FILENAME = "./cargo/cargo_stable_solutions.pkl"

df = pd.read_csv(f"../results/jpetstore/call_graph.csv").reset_index() # The original dataframe has the from and to columns as indices
java_graph = nx.from_pandas_edgelist(df, source='from', target='to', create_using=nx.Graph(), edge_attr='weight')

ALL_PARAMETERS = ['k',  'resolution']

experiments_df, outcomes = load_results(MODEL_FILENAME+ '.tar.gz')
print(experiments_df.shape)
experiments_df = experiments_df[ALL_PARAMETERS].drop_duplicates(keep='first')
print(experiments_df.shape)

# Creating labels for the experiments
exp_labels = []
for idx, row in experiments_df.iterrows():
    lb = f"resolution_{row['resolution']:.9f}_k_{int(row['k'])}"
    exp_labels.append(lb)
# print(exp_labels)
experiments_df.index = exp_labels
print(experiments_df.head())

partitions_dict = None
with open(PARTITIONS_FILENAME, 'rb') as f:
     partitions_dict = pickle.load(f)
print("partitions:", len(partitions_dict))
key_0 = list(partitions_dict.keys())[10]
# print(partitions_dict[key_0])
print(key_0)

relevant_parameters = ['resolution', 'k'] # ['resolution'] # ALL_PARAMETERS
# relevant_parameters = ['resolution', 'mfuzzy']

print("Computing parameter distances ...", relevant_parameters)
distances_df, scaler = compute_parameter_distances(experiments_df, experiments_df.index, parameters=relevant_parameters,
                                        metric='euclidean', scaling=True)
print(distances_df.shape)

distances_df.to_csv(DISTANCE_FILENAME)
# distances_df = pd.read_csv(DISTANCE_FILENAME, index_col=0)

distance_np = np.tril(distances_df).flatten()
min_non_zero = np.min(distance_np[np.nonzero(distance_np)])
print("min-max distances:", min_non_zero, distance_np.max())
print(distances_df.max().max())

print()
# Test the binary search
print("Binary search ...")
# for t in range(50,96,5):
#     print("-->", (t/100.0))
#     radius, coverage, configs = binary_search(distances_df, partitions_dict, threshold=t/100.0, 
#                                           max_trials=50, tolerance=0.0001, verbose=False)
#     # print()
#     print("Max radius:", radius, "Coverage of configurations:", coverage)
#     print(len(configs), configs)

t = 0.95
print("-->", t)
radius, coverage, configs, explored_configs = binary_search(distances_df, partitions_dict, threshold=t, verbose=True, tolerance=0.005)
print("Max radius:", radius, "Coverage of configurations:", coverage)
print(len(configs), configs)
print(len(explored_configs), "explored configurations") #, explored_configs)
print()

step = np.array([radius]*len(relevant_parameters))
print("Denormalization for radius=", radius, scaler.inverse_transform(step.reshape(1, -1))[0], relevant_parameters)
    
print("done.")

# Test with a fixed radius
print()
test = 0.24 #0.12
f = check_experiments(distances_df, test, partitions_dict, sim_threshold=t)
f_sum = sum(f)
print(test, "Count:", f_sum, "Coverage:", f_sum/len(partitions_dict))
selected_exps = [col for idx, col in zip(f, distances_df.columns) if idx == 1]
for exp in selected_exps: # It adds additional configurations
    if exp not in explored_configs.keys():
        explored_configs[exp] = test
with open(STABLE_SOLUTIONS_FILENAME, 'wb') as output:
    pickle.dump(explored_configs, output)
# print(explored_configs)