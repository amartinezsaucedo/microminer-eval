import networkx as nx
from cdlib import algorithms


class TypeService:
    _graph: nx.Graph
    _distances: list[list]

    def __init__(self, classes: list[str], distances: list[list]):
        self._classes = classes
        self._graph = nx.Graph()
        self._distances = distances
        self._initialize_graph(classes)

    def _initialize_graph(self, classes: list[str]):
        for class_name in classes:
            self._graph.add_node(class_name)
        for distance in self._distances:
            if distance[0] in classes and distance[1] in classes:
                self._graph.add_edge(distance[0], distance[1], weight=distance[2])

    def get_communities(self, resolution: float) -> list:
        return algorithms.louvain(self._graph, weight='weight', resolution=resolution).communities

    def fine_tune_communities(self, community: list) -> list:
        for cluster in community:
            if len(cluster) < 2:
                community = self._fine_tune(cluster[0], community)
        return community

    def _fine_tune(self, class_name: str, community: list) -> list:
        service_score = []
        for _ in community:
            service_score.append(0)
        for distance in self._distances:
            if distance[0] == class_name and distance[1] != class_name:
                for index, cluster in enumerate(community):
                    for class_name_cluster in cluster:
                        if distance[1] == class_name_cluster:
                            service_score[index] += distance[2]
                            break
            elif distance[1] == class_name and distance[0] != class_name:
                for index, cluster in enumerate(community):
                    for class_name_cluster in cluster:
                        if distance[0] == class_name_cluster:
                            service_score[index] += distance[2]
                            break
        max_score = max(service_score)
        if max_score > 0:
            max_score_index = service_score.index(max(service_score))
            indices = []
            for i, x in enumerate(service_score):
                if x == max_score:
                    indices.append(i)
            if len(indices) == 1:
                community[max_score_index].append(class_name)
                community.remove([class_name])
        return community

