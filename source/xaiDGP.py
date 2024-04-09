import argparse

import torch
import torch.nn.functional as F
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import gzip
import random
import pickle
import os
import os.path as osp
import torch
import time
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer
from torch_geometric.data import Data

from source.DiseaseNet import DiseaseNet
from source.GeneNet import GeneNet
from source.gNetDGPModel import gNetDGPModel


class ExplainDGP():
    def __init__(self, model_path, node, mode):
        self.model_path = model_path
        self.node = node
        self.mode = mode
        self.model = None
        self.explanation = None
      
    def load_model():
        fc_hidden_dim = 3000
        gene_net_hidden_dim = 830
        disease_net_hidden_dim = 500
        gene_dataset_root = "./data/gene_net"
        disease_dataset_root = "/data/disease_net"
    
        print('Load the gene and disease graphs.')
        gene_dataset = GeneNet(
            root=gene_dataset_root,
            humannet_version='FN',
            features_to_use=['hpo'],
            skip_truncated_svd=True
        )

        disease_dataset = DiseaseNet(
            root=disease_dataset_root,
            hpo_count_freq_cutoff=40,
            edge_source='feature_similarity',
            feature_source=['disease_publications'],
            skip_truncated_svd=True,
            svd_components=2048,
            svd_n_iter=12
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gene_net_data = gene_dataset[0]
        disease_net_data = disease_dataset[0]
        gene_net_data = gene_net_data.to(device)
        disease_net_data = disease_net_data.to(device)

        disease_id_index_feature_mapping = disease_dataset.load_disease_index_feature_mapping()
        gene_id_index_feature_mapping = gene_dataset.load_node_index_mapping()
    
        self.model = gNetDGPModel(
            gene_feature_dim=gene_net_data.x.shape[1],
            disease_feature_dim=disease_net_data.x.shape[1],
            fc_hidden_dim=fc_hidden_dim,
            gene_net_hidden_dim=gene_net_hidden_dim,
            disease_net_hidden_dim=disease_net_hidden_dim,
            mode='DGP'
        ).to(device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        
    
    def generate_explanation(node_index=10):
        print("Running GNN explanation...")
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )
        #node_index = 596
        self.explanation = explainer(data.x, data.edge_index, index=node_index)


    def visualize_explanation():
        print(f'Generated explanations in {self.explanation.available_explanations}')
        #path_subgraph = './results/feature_importance.png'
        #explanation.visualize_feature_importance(path, top_k=10)
        #print(f"Feature importance plot has been saved to '{path}'")
        #plt.figure(figsize=(8, 6))
        path_subgraph = './results/subgraph.pdf'
        self.explanation.visualize_graph(path_subgraph)
        
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide arguments')
    parser.add_argument('--model_path', type=int, help='trained model')
    parser.add_argument('--node', type=str, help='node id for explanation')
    parser.add_argument('--mode', type=str, help='training mode - generic or specific')
    args = parser.parse_args()

    explain_node = ExplainDGP(args.arg1, args.arg2, args.arg3)
    
