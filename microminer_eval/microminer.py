import math
import networkx as nx
from scipy import spatial
from microminer_eval.partition.preprocess_source_code import Preprocesser
from microminer_eval.partition.typed_services_identification import TypeService
from microminer_eval.utils.file import read_file


def _remove_from_array(values: list, values_to_remove: list):
    return [clazz for clazz in values if clazz not in values_to_remove]


class MicroMiner:
    _preprocesser: Preprocesser
    _project_path: str
    _project_name: str
    _model_path: str
    _classes_data: dict
    _typed_classes: list[str]
    _typed_services: dict
    _distances: list
    _resolution: float
    _alpha: float
    _beta: float
    _m: int
    _microservice_threshold: int

    def __init__(self, project_path: str, project_name: str, model_path: str):
        self._project_path = project_path
        self._project_name = project_name
        self._model_path = model_path
        self._load_classes()
        self._preprocesser = Preprocesser(project_name=self._project_name, project_path=self._project_path)
        errors = self._preprocesser.preprocess(self._typed_classes)
        self._remove_classes_with_errors(errors)
        self._distances = self._preprocesser.load_call_graph()
        self._typed_services = {
            "application": None,
            "utility": None,
            "entity": None
        }

    def _load_classes(self):
        self._classes_data = {
            "application": [],
            "utility": [],
            "entity": []
        }
        for class_type in self._classes_data.keys():
            self._classes_data[class_type] = read_file(f"{self._project_path}/{class_type}.txt")
        self._typed_classes = []
        for classes in self._classes_data.values():
            self._typed_classes.extend(classes)

    def _remove_classes_with_errors(self, errors: list[str]):
        for class_type, classes in self._classes_data.items():
            self._classes_data[class_type] = _remove_from_array(classes, errors)

    def configure(self, resolution: float, alpha: float, beta: float, m: int, microservice_threshold: int):
        self._resolution = resolution
        self._alpha = alpha
        self._beta = beta
        self._m = m
        self._microservice_threshold = microservice_threshold

    def partition(self):
        self._identify_type_service()
        return self.create_service_graph()

    def _identify_type_service(self):
        service_types_to_remove = []
        for service_type in self._typed_services.keys():
            typed_services = TypeService(self._classes_data[service_type], self._distances)
            if not typed_services.is_graph_empty():
                communities = typed_services.get_communities(self._resolution)
                communities = typed_services.fine_tune_communities(communities)
                self._typed_services[service_type] = communities
            else:
                service_types_to_remove.append(service_type)
        for service_type_to_remove in service_types_to_remove:
            self._typed_services.pop(service_type_to_remove, None)

    def create_service_graph(self):
        services = dict()
        services_embeddings = dict()
        classes_per_service = dict()
        for service_type, communities in self._typed_services.items():
            for community in communities:
                name = f"{service_type}{community[0]}"
                services[name] = community
                services_embeddings[name] = self._preprocesser.calculate_embeddings_service(community)
                for class_name in community:
                    classes_per_service[class_name] = name
        services_distance = []
        for distance in self._distances:
            class1 = distance[0]
            class2 = distance[1]
            static_distance = float(distance[2])
            if class1 in classes_per_service and class2 in classes_per_service:
                S1 = classes_per_service[class1]
                S2 = classes_per_service[class2]
                semantic_distance = self._calculate_semantic_distance(services_embeddings[S1], services_embeddings[S2])
                services_distance.append([S1, S2, static_distance, semantic_distance])
        graph = nx.Graph()
        for distances in services_distance:
            weight = self._calculate_weight(distances[3], distances[2])
            graph.add_edge(distances[0], distances[1], weight=weight)
        adjacency_matrix = []
        nodes = list(graph.nodes)
        for _ in nodes:
            array = []
            for _ in nodes:
                array.append(math.inf)
            adjacency_matrix.append(array)
        for index, node in enumerate(nodes):
            adjacencies = graph.adj[node]
            for adj_one in adjacencies:
                adj_index = nodes.index(adj_one)
                for d in services_distance:
                    if d[0] == adj_one and d[1] == node:
                        adjacency_matrix[index][adj_index] = 100 / d[2]
                        break
                    if d[1] == adj_one and d[0] == node:
                        adjacency_matrix[index][adj_index] = 100 / d[2]
                        break
        distances = self._floyd_warshall(adjacency_matrix, len(nodes))
        for index_i, node_i in enumerate(nodes):
            for index_j, node_j in enumerate(nodes):
                ds = self._calculate_semantic_distance(services_embeddings[node_i], services_embeddings[node_j])
                ds = ds * 2
                distances[index_i][index_j] = ds + float(distances[index_i][index_j])
        application_index = dict()
        for index, node in enumerate(nodes):
            if node.startswith("application"):
                application_index[node] = index
        microservices = dict()
        for index, application in enumerate(application_index):
            wij = []
            microservice = []
            for service in services[application]:
                microservice.append(service)
            for node_index, node in enumerate(nodes):
                if node != application:
                    c = self._calculate_membership_denominator(node_index, application_index[application], distances,
                                                               application_index)
                    val = 1 / c
                    wij.append(val)
                    if val * 100 > self._microservice_threshold:
                        for s in services[node]:
                            microservice.append(s)
            microservices[index] = microservice
        return microservices

    def _calculate_semantic_distance(self, value1: float, value2: float):
        return spatial.distance.cosine(value1, value2)

    def _calculate_weight(self, semantic_distance: float, static_distance: float):
        if not (semantic_distance == 0):
            semantic_distance = 1 / semantic_distance
        return (self._alpha * float(static_distance)) + (self._beta * semantic_distance)

    def _floyd_warshall(self, adjacency_matrix: list, vertices: int):
        distance = list(map(lambda x: list(map(lambda y: y, x)), adjacency_matrix))
        for k in range(vertices):
            for i in range(vertices):
                for j in range(vertices):
                    distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
        return distance

    def _calculate_membership_denominator(self, node_index: int, cluster, distances: list, application_index: dict):
        coef = 2 / (self._m - 1)
        wij = 0
        dij = distances[node_index][cluster]
        for index in application_index:
            v = dij / distances[node_index][application_index[index]]
            val = math.pow(v, coef)
            wij += val
        return wij
