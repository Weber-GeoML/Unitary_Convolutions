from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid, HeterophilousGraphDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, dropout_edge
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment

import time
import os
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input

import torch_geometric.transforms as T
from torch_geometric.transforms import Compose


default_args = AttrDict({
    "dropout": 0.2,
    "num_layers": 8,
    "hidden_dim": 512,
    "learning_rate": 3 * 1e-5,
    "layer_type": "Unitary",
    "display": True,
    "num_trials": 10,
    "eval_every": 1,
    "rewiring": None,
    "num_iterations": 1,
    "num_relations": 2,
    "patience": 2000,
    "dataset": None,
    "borf_batch_add" : 4,
    "borf_batch_remove" : 2,
    "sdrf_remove_edges" : False,
    "encoding": None,
    "T" : 20
})


results = []
args = default_args
args += get_args_from_input()


largest_cc = LargestConnectedComponents()
cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
chameleon = WikipediaNetwork(root="data", name="chameleon")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
pubmed = Planetoid(root="data", name="pubmed")
roman_empire = HeterophilousGraphDataset(root="data", name="Roman-empire")
amazon_ratings = HeterophilousGraphDataset(root="data", name="Amazon-ratings")
minesweeper = HeterophilousGraphDataset(root="data", name="Minesweeper")
tolokers = HeterophilousGraphDataset(root="data", name="Tolokers", transform=largest_cc)
questions = HeterophilousGraphDataset(root="data", name="Questions", transform=largest_cc)


datasets = {"roman_empire": roman_empire, "amazon_ratings": amazon_ratings, "minesweeper": minesweeper, "tolokers": tolokers, "questions": questions}
for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()


if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}


for key in datasets:
    accuracies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]


    # encode the dataset using the given encoding, if args.encoding is not None
    if args.encoding in ["LAPE", "RWPE", "LDP", "SUB", "EGO"]:

        if os.path.exists(f"data/{key}_{args.encoding}.pt"):
            print('ENCODING ALREADY COMPLETED...')
            dataset = torch.load(f"data/{key}_{args.encoding}.pt")

        elif args.encoding == "RWPE":
            print('ENCODING STARTED...')
            transform = T.AddRandomWalkPE(walk_length=16)
            dataset.data.x = torch.cat((transform(dataset.data).random_walk_pe, dataset.data.x), dim=1)

        else:
            print('ENCODING STARTED...')

            if args.encoding == "LAPE":
                num_nodes = dataset[i].num_nodes
                eigvecs = np.min([num_nodes, 8]) - 2
                transform = T.AddLaplacianEigenvectorPE(k=eigvecs)

            elif args.encoding == "LDP":
                transform = T.LocalDegreeProfile()

            elif args.encoding == "SUB":
                transform = T.RootedRWSubgraph(walk_length=10)

            elif args.encoding == "EGO":
                transform = T.RootedEgoNets(num_hops=2)

            elif args.encoding == "VN":
                transform = T.VirtualNode()

            dataset.data = transform(dataset.data)

            # save the dataset to a file in the data folder
            torch.save(dataset, f"data/{key}_{args.encoding}.pt")


    start = time.time()
    for trial in range(args.num_trials):
        print(f"TRIAL #{trial+1}")
        train_accs = []
        test_accs = []
        for i in range(args.num_splits):
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        train_acc = max(train_accs)
        test_acc = max(test_accs)
        accuracies.append(test_acc)
        # accuracies.append(train_acc)
    end = time.time()
    run_duration = end - start

    log_to_file(f"RESULTS FOR {key} ({args.rewiring}):\n")
    log_to_file(f"average acc: {np.mean(accuracies)}\n")
    log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    results.append({
        "dataset": key,
        "layer_type": args.layer_type,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        # "train_mean": np.mean(accuracies),
        "test_mean": np.mean(accuracies),
        "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5),
        "run_duration" : run_duration,
    })
    results_df = pd.DataFrame(results)
    with open(f'results/node_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell()==0)