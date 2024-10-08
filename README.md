# Monolith to Microservices algorithms evaluation

Available algorithms:
- *MicroMiner*: Trabelsi, Imen & Abdellatif, Manel & Abubaker, Abdalgader & Moha, Naouel & Mosser, Sébastien & Ebrahimi‐Kahou, Samira & Guéhéneuc, Yann-Gaël. (2022). From legacy to microservices: A type‐based approach for microservices identification using machine learning and semantic analysis. Journal of Software: Evolution and Process. 35. 10.1002/smr.2503.
## MicroMiner
### Prerequisites 
1. Create .xmi file from project to partition by using MoDisco plugin and place it along .java files (All .java files must be placed under the same directory).
2. Install dependencies running `poetry install`
### Usage
```bash
usage: main.py [-h] -p PROJECT_NAME -f PROJECT_PATH [-m MODEL_PATH]
options:
-h, --help            show this help message and exit
-p PROJECT_NAME, --project-name PROJECT_NAME
Project name
-f PROJECT_PATH, --project-path PROJECT_PATH
Project path
-m MODEL_PATH, --model_path MODEL_PATH
Classifier path
```
Example
```bash
python microminer_eval/main.py -p "Cargo Tracking" -f ../cargo
```

# EMAT
```bash
python microminer_eval/main_emat.py -p "jpetstore" -f ./jpetstore
```

```bash
python microminer_eval/partition_matrix.py -p "jpetstore" -f ./jpetstore
```

```bash
python microminer_eval/parameter_matrix.py -p "jpetstore" -f ./jpetstore
```