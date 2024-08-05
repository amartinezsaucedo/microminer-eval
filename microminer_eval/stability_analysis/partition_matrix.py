import argparse
from cdlib import evaluation
from cdlib.classes import NodeClustering
import pickle
from tqdm import tqdm
import pandas as pd
import os

def get_noise_classes(partition, java_graph):
    reference_class_set = set(list(java_graph.nodes()))
    partition_class_set = set([x for xs in partition.values() for x in xs])
    difference_set = reference_class_set.difference(partition_class_set)
    return difference_set


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

# This computation can take some time, particularly for large graphs with heterogeneous partitions
def compute_indices(partitions_dict, graph, distance=False, metric="omega", include_noise=False, checkpoint=None):
    print(metric, checkpoint, len(partitions_dict), graph.number_of_nodes(), include_noise, distance)

    omega_scores = dict() 
    if (checkpoint is not None) and os.path.exists(checkpoint):
        with open(checkpoint, 'rb') as f:
            omega_scores = pickle.load(f) # deserialize the dict
            print("Checkpoint loaded:", len(omega_scores), 'entries')

    for m, i in enumerate(tqdm(partitions_dict.keys())):
        if i not in omega_scores.keys():
            omega_scores[i] = dict()

            noise_classes_i = set()
            if include_noise:
                noise_classes_i = get_noise_classes(partitions_dict[i], graph)
            if len(noise_classes_i) > 0:
                partition_i = list(partitions_dict[i].values()) + [list(noise_classes_i)]
            else:
                partition_i = list(partitions_dict[i].values())
            clustering_i = NodeClustering(communities=partition_i, graph=graph, overlap=True)
        
            for n, j in enumerate(partitions_dict.keys()):
                if n < m:
                    partition_j = list(partitions_dict[j].values())
                    noise_classes_j = set()
                    if include_noise:
                        noise_classes_j = get_noise_classes(partitions_dict[j], graph)
                    if len(noise_classes_j) > 0:
                        partition_j = list(partitions_dict[j].values()) + [list(noise_classes_j)]
                    else:
                        partition_j = list(partitions_dict[j].values())

                    clustering_j = NodeClustering(communities=partition_j, graph=graph, overlap=True)
                    value = None
                    if metric == "omega":
                        value = evaluation.omega(clustering_i, clustering_j).score
                    elif metric == "mutualinfo_lfk":
                        value = evaluation.overlapping_normalized_mutual_information_LFK(clustering_i, clustering_j).score
                    elif metric == "mutualinfo_mgh":
                        value = evaluation.overlapping_normalized_mutual_information_MGH(clustering_i, clustering_j).score
                    else:
                        raise ValueError("Invalid metric")
                    if distance:
                        value = 1 - value
                elif n == m:
                    value = 1.0
                    if distance:
                        value = 1 - value

                if value is not None:
                    omega_scores[i][j] = value
        
            if checkpoint is not None: # Write the whole row
                with open(checkpoint, 'wb') as f:  # open
                    pickle.dump(omega_scores, f) # serialize the dict
                    f.close()
    
    # Complete the rest of the matrix
    for m, i in enumerate(tqdm(partitions_dict.keys())):
        for n, j in enumerate(partitions_dict.keys()):
            if n > m:
                omega_scores[i][j] = omega_scores[j][i]

    omega_scores_df = pd.DataFrame(omega_scores)
    return omega_scores_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project-name", help="Project name", required=True, type=str)
    parser.add_argument("-f", "--project-path", help="Project path", required=True, type=str)

    arguments = parser.parse_args()

    project_name = arguments.project_name
    project_path = arguments.project_path

    GRAPH_FILENAME = f"{project_path}/{project_name}_128scenarios_nopolicies_sobol_graph.pkl"
    java_graph = None
    with open(GRAPH_FILENAME, 'rb') as f:
         java_graph = pickle.load(f)

    PARTITIONS_FILENAME = f"{project_path}/{project_name}_128scenarios_nopolicies_sobol_partitions.pkl"
    partitions_dict = None
    with open(PARTITIONS_FILENAME, 'rb') as f:
         partitions_dict = pickle.load(f)
    print("partitions:", len(partitions_dict))
    key_0 = list(partitions_dict.keys())[10]
    print(partitions_dict[key_0])

    partitions_similarity_df = compute_indices(partitions_dict, java_graph, metric="omega",
                                               include_noise=False, checkpoint="temporal.pkl")
    print(partitions_similarity_df.shape)

    SIMILARITY_FILENAME = f"{project_path}/{project_name}_omega_scores.csv"

    partitions_similarity_df.to_csv(SIMILARITY_FILENAME)