import argparse

from ema_workbench import load_results
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from cdlib import evaluation
from cdlib.classes import NodeClustering
import pickle
import numpy as np


def compute_parameter_distances(experiments_df, columns=None, parameters=[], metric='cosine', scaling=True):
    scaler = None
    data = experiments_df[parameters].values
    scaled_data = data
    if scaling:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

    dist = pdist(scaled_data, metric=metric)
    if columns is not None:
        dist_df = pd.DataFrame(squareform(dist), columns=columns, index=columns)
    else:
        dist_df = pd.DataFrame(squareform(dist))
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
        partition[-1] = list(noise_classes)  # -1 is the key for the noise classes

    return partition


"""
Por cada descomposición, busco aquellas que tengan una distance menor a p_step (valor de búsqueda binaria)
Si TODAS las descomposiciones a una distancia menor a p_step son similares (mayor a sim_threshold), entonces 1
"""


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
            # print("\t", exp, "indices so far (but discarded):", len(selected_indices))
            count_experiments.append(0)
    return count_experiments


# Generate code for a binary search of a suitable parameter value within a continuous interval. 
# The stopping criterion for the search is  given by an error threshold. 
# The suitability of the parameter value is given by another function.
def binary_search(distance_df, partitions_dict, threshold=0.95, max_trials=25, freq=5,
                  interval=(0.0, 1.0), tolerance=0.0005, verbose=True):
    if verbose:
        print("threshold=", threshold, " max_trials=", max_trials, " interval=", interval)

    explored_exps = dict()
    left = interval[0]
    right = interval[1]
    n = 1
    best_f = None
    best_mid = None
    mid = None
    for n in tqdm(range(1, max_trials + 1)):
        if (abs(right - left) >= tolerance):
            mid = (left + right) / 2
            if n % freq == 0:
                if verbose:
                    print(n, "r=", mid, "...")
            f = check_experiments(distance_df, mid, partitions_dict, sim_threshold=threshold)
            f_sum = sum(f)
            if f_sum == 0:  # Mid point is not suitable yet
                right = mid
            else:
                best_f = f
                best_mid = mid
                if verbose:
                    print(n, "r=", mid, ", satisfying distance found!", "count=", f_sum, "coverage=",
                          f_sum / len(partitions_dict))
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
        return best_mid, sum(best_f) / len(partitions_dict), selected_exps, explored_exps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project-name", help="Project name", required=True, type=str)
    parser.add_argument("-f", "--project-path", help="Project path", required=True, type=str)

    arguments = parser.parse_args()

    project_name = arguments.project_name
    project_path = arguments.project_path

    GRAPH_FILENAME = f"{project_path}/{project_name}_128scenarios_nopolicies_sobol_graph.pkl"

    MODEL_FILENAME = f"{project_path}/{project_name}_128scenarios_nopolicies_sobol"  # .tar.gz

    PARTITIONS_FILENAME = f"{project_path}/{project_name}_128scenarios_nopolicies_sobol_partitions.pkl"

    DISTANCE_FILENAME = f"{project_path}/{project_name}_parameter_distances.csv"

    STABLE_SOLUTIONS_FILENAME = f"{project_path}/{project_name}_stable_solutions.pkl"

    java_graph = None
    with open(GRAPH_FILENAME, 'rb') as f:
        java_graph = pickle.load(f)

    ALL_PARAMETERS = ['k', 'resolution']

    experiments_df, outcomes = load_results(MODEL_FILENAME + '.tar.gz')
    print(experiments_df.shape)
    experiments_df = experiments_df[ALL_PARAMETERS].drop_duplicates(keep='first')
    print(experiments_df.shape)

    # Creating labels for the experiments
    exp_labels = []
    for idx, row in experiments_df.iterrows():
        lb = 'resolution_' + str(row['resolution']) + '_k_' + str(row['k'])
        exp_labels.append(lb)
    experiments_df.index = exp_labels
    print(experiments_df.head())

    partitions_dict = None
    with open(PARTITIONS_FILENAME, 'rb') as f:
        partitions_dict = pickle.load(f)
    print("partitions:", len(partitions_dict))
    key_0 = list(partitions_dict.keys())[10]
    print(key_0)

    relevant_parameters = ['resolution', 'k']  # ['resolution'] # ALL_PARAMETERS
    #  relevant_parameters = ['resolution', 'mfuzzy']

    print("Computing parameter distances ...", relevant_parameters)
    distances_df, scaler = compute_parameter_distances(experiments_df, experiments_df.index,
                                                       parameters=relevant_parameters,
                                                       metric='euclidean', scaling=True)
    print(distances_df.shape)

    distances_df.to_csv(DISTANCE_FILENAME)

    distance_np = np.tril(distances_df).flatten()
    min_non_zero = np.min(distance_np[np.nonzero(distance_np)])
    print("min-max distances:", min_non_zero, distance_np.max())
    print(distances_df.max().max())

    print()
    # Test the binary search
    print("Binary search ...")
    t = 0.95
    print("-->", t)
    radius, coverage, configs, explored_configs = binary_search(distances_df, partitions_dict, threshold=t,
                                                                verbose=True, tolerance=0.005)
    print("Max radius:", radius, "Coverage of configurations:", coverage)
    print(len(configs), configs)
    print(len(explored_configs), "explored configurations")  # , explored_configs)
    print()

    step = np.array([radius] * len(relevant_parameters))
    print("Denormalization for radius=", radius, scaler.inverse_transform(step.reshape(1, -1))[0], relevant_parameters)

    print("done.")

    # Test with a fixed radius
    print()
    test = 0.24  # 0.12
    f = check_experiments(distances_df, test, partitions_dict, sim_threshold=t)
    f_sum = sum(f)
    print(test, "Count:", f_sum, "Coverage:", f_sum / len(partitions_dict))
    selected_exps = [col for idx, col in zip(f, distances_df.columns) if idx == 1]
    for exp in selected_exps:  # It adds additional configurations
        if exp not in explored_configs.keys():
            explored_configs[exp] = test
    with open(STABLE_SOLUTIONS_FILENAME, 'wb') as output:
        pickle.dump(explored_configs, output)
