import logging
import xml.etree.ElementTree as ElementTree
from pandas import DataFrame
import networkx as nx


class CallGraph:
    _weights: dict = {
        'code:StorableUnit': 100,  # Aggregation
        'action:Calls': 5,  # Method invocation
        'action:Creates': 25,  # Instantiation,
        'code:Extends': 100,  # Generalization
        'code:ImplementationOf': 100,  # Implementation
        'code:Implements': 100  # Implementation
    }
    _root: ElementTree = None
    _graph: DataFrame = None
    _project_path: str
    _project_name: str

    def __init__(self, project_path: str, project_name: str):
        self._project_path = project_path
        self._project_name = project_name
        self._root = ElementTree.parse(f"{project_path}/{project_name}_kdm.xmi")
        logging.info("KDM parsed")

    def as_nx_graph(self):
        g = None
        if self._graph is not None: # Build a graph from the dataframe
            df = self._graph.copy().reset_index() # The original dataframe has the from and to columns as indices
            g = nx.from_pandas_edgelist(df, source='from', target='to', create_using=nx.Graph(), edge_attr='weight')
        return g

    def set_weights(self, weights: dict):
        self._weights = weights

    def generate(self):
        calls = []
        for element_type in ['actionRelation', 'codeRelation']:
            calls.extend(self._add_classes(element_type))
        graph = DataFrame(calls)
        self._graph = DataFrame(graph.groupby(['from', 'to'])['weight'].sum())
        logging.info("Call graph generated")

    def _add_classes(self, key):
        calls = []
        for element in self._root.iter(key):
            action_type = element.attrib['{http://www.w3.org/2001/XMLSchema-instance}type']
            if action_type in ['action:Calls', 'action:Creates', 'code:Extends', 'code:ImplementationOf',
                               'code:Implements']:
                from_class = self._parse_class(element.attrib['from'])
                to_class = self._parse_class(element.attrib['to'])
                weight = self._weights.get(action_type)
                if from_class and from_class.endswith('.java') and to_class and to_class.endswith('.java'):
                    calls.append({'from': from_class, 'to': to_class, 'weight': weight, 'type': action_type})
        return calls

    def _parse_source(self, location):
        source = location.find("source")
        if source:
            region = source.find("region")
            file = region.attrib.get("file")
            if file:
                name = self._get_node_from_location(file).attrib.get('name')
            else:
                name = location.attrib.get('name')
            return name

    def _parse_class(self, location):
        return self._parse_source(self._get_node_from_location(location))

    def _get_node_from_location(self, location):
        parts = location.split("/")[2:]
        node = None
        for part in parts:
            [name, index] = part.replace("@", "").split(".")
            index = int(index)
            if not node:
                node = list(self._root.iter("model"))[index]
            else:
                node = list(node.findall(name))[index]
        return node

    def export(self):
        filename = f"{self._project_path}/{self._project_name}_call_graph.csv"
        self._graph.to_csv(filename, header=None)
        logging.info(f"Call graph exported to {filename}")
