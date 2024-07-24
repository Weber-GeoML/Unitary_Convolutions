# TU and Heterophilous Node Classification Datasets

## Requirements
To configure and activate the conda environment for this subdirectory, run
```
conda env create -f environment.yml
conda activate borf 
pip install -r requirements.txt
```

## Experiments
### 1. For graph classification
To run experiments for the TUDataset benchmark, run the file ```run_graph_classification.py```. The following command will run the benchmark with a GCN with 4 UniConv layers:
```bash
python run_graph_classification.py --layer_type Unitary --num_layers 4
```

### 2. For node classification
To run node classification, simply change the script name to `run_node_classification.py`. For example:
```bash
python run_node_classification.py --dataset tolokers --layer_type Unitary
```