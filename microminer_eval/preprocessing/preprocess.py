from microminer_eval.preprocessing.call_graph import CallGraph
from microminer_eval.preprocessing.class_classification import ClassClassifier

def preprocess(project_path: str, project_name: str, model_path: str):
    class_classifier = ClassClassifier(model_path, project_path)
    class_classifier.classify()
    call_graph = CallGraph(project_path, project_name)
    call_graph.generate()
    call_graph.export()

    return call_graph
