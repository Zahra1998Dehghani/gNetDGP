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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split

from source.DiseaseNet import DiseaseNet
from source.GeneNet import GeneNet
from source.gNetDGPModel import gNetDGPModel


class HyperOptmisation():
    
    def run_optimisation(self, folds, max_epochs, max_trials, early_stopping_window, gene_dataset_root, disease_dataset_root, training_data_path, optimised_model_storage):
        print("Running optimisation...")
        print("Creating datasets folds...")
        negatives, positives, cov_disease, g_i_f_mapping, d_i_i_mapping = self.setup_data(gene_dataset_root, disease_dataset_root, training_data_path)
        self.train_optimise(folds, negatives, positives, cov_disease, g_i_f_mapping, d_i_i_mapping, max_epochs, max_trials, optimised_model_storage)
        print("Setting up optimisation values...")
        print("Best values of hyperparameters")
        print("Finished optimisation")
 
    def create_objective(self):
        #train_loader, test_loader = load_data()  # Load some data
        #model = ConvNet().to("cpu")  # Create a PyTorch conv net
        optimizer = torch.optim.SGD(  # Tune the optimizer
            model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )

        while True:
            train(model, optimizer, train_loader)  # Train the model
            acc = test(model, test_loader)  # Compute test accuracy
            train.report({"mean_accuracy": acc})  # Report to Tune

        search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
        algo = OptunaSearch()

        tuner = tune.Tuner(objective, 
                tune_config=tune.TuneConfig(metric="mean_accuracy", mode="max", search_alg=algo),
                    run_config=train.RunConfig(stop={"training_iteration": 5}),
                    param_space=search_space,
                )
        results = tuner.fit()
        print("Best config is:", results.get_best_result().config)


    def setup_data(self, gene_dataset_root, disease_dataset_root, training_data_path):
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

        self.gene_net_data = gene_dataset[0]
        self.disease_net_data = disease_dataset[0]

        self.gene_net_data = self.gene_net_data.to(device)
        self.disease_net_data = self.disease_net_data.to(device)

        print('Generate training data.')
        disease_genes = pd.read_table(
            training_data_path,
            names=['EntrezGene ID', 'OMIM ID'],
            sep='\t',
            low_memory=False,
            dtype={'EntrezGene ID': pd.Int64Dtype()}
        )

        disease_id_index_feature_mapping = disease_dataset.load_disease_index_feature_mapping()
        gene_id_index_feature_mapping = gene_dataset.load_node_index_mapping()

        all_genes = list(gene_id_index_feature_mapping.keys())
        all_diseases = list(disease_id_index_feature_mapping.keys())

        # 1. generate positive pairs.
        # Filter the pairs to only include the ones where the corresponding nodes are available.
        # i.e. gene_id should be in all_genes and disease_id should be in all_diseases.
        positives = disease_genes[
            disease_genes["OMIM ID"].isin(all_diseases) & disease_genes["EntrezGene ID"].isin(all_genes)
        ]
        covered_diseases = list(set(positives['OMIM ID']))
        covered_genes = list(set(positives['EntrezGene ID']))
        negatives_list = []
        while len(negatives_list) < len(positives):
            gene_id = all_genes[np.random.randint(0, len(all_genes))]
            disease_id = covered_diseases[np.random.randint(0, len(covered_diseases))]
            if not ((positives['OMIM ID'] == disease_id) & (positives['EntrezGene ID'] == gene_id)).any():
                negatives_list.append([disease_id, gene_id])
        negatives = pd.DataFrame(np.array(negatives_list), columns=['OMIM ID', 'EntrezGene ID'])
        return negatives, positives, covered_diseases, gene_id_index_feature_mapping, disease_id_index_feature_mapping


    def get_training_data_from_indexes(self, indexes, negatives, positives, covered_diseases, gene_id_index_feature_mapping, disease_id_index_feature_mapping, monogenetic_disease_only=False, multigenetic_diseases_only=False):
        train_tuples = set()
        for idx in indexes:
            pos = positives[positives['OMIM ID'] == covered_diseases[idx]]
            neg = negatives[negatives['OMIM ID'] == covered_diseases[idx]]
            if monogenetic_disease_only and len(pos) != 1:
                continue
            if multigenetic_diseases_only and len(pos) == 1:
                continue
            for index, row in pos.iterrows():
                train_tuples.add((row['OMIM ID'], row['EntrezGene ID'], 1))
            for index, row in neg.iterrows():
                train_tuples.add((row['OMIM ID'], row['EntrezGene ID'], 0))
            
        n = len(train_tuples)           
        x_out = np.ones((n, 2)) # will contain (gene_idx, disease_idx) tuples
        y_out = torch.ones((n,), dtype=torch.long)
        for i, (omim_id, gene_id, y) in enumerate(train_tuples):
            x_out[i] = (gene_id_index_feature_mapping[int(gene_id)], disease_id_index_feature_mapping[omim_id])
            y_out[i] = y
        return x_out, y_out
        
    
    def train_optimise(
        self,
        folds,
        negatives,
        positives,
        cov_diseases, 
        g_i_f_mapping,
        d_i_i_mapping,
        max_epochs,
        max_trials,
        optimised_model_storage,
        early_stopping_window=5,
        info_each_epoch=1, 
        lr=0.0005,
        weight_decay=5e-4,
        fc_hidden_dim=2048,
        gene_net_hidden_dim=512,
        disease_net_hidden_dim=512
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device))
        metrics = []
        dis_dict = {}
        fold = 0
        start_time = time.time()
        #gene_net_data = self.gene_net_data
        #disease_net_data = self.disease_net_data

        from hyperopt import hp
        from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK
        
        def optimise_params(config):
            fold = 0
            all_losses = {}
            all_losses['fold_train'] = list()
            all_losses['fold_val'] = list()
            
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(cov_diseases):
                fold += 1
                print(f'Generate training data for fold {fold}.')
                all_train_x, all_train_y = self.get_training_data_from_indexes(train_index, negatives, positives, cov_diseases, g_i_f_mapping, d_i_i_mapping)

                # Split into train and validation set.
                id_tr, id_val = train_test_split(range(len(all_train_x)), test_size=0.1, random_state=42)
                train_x = all_train_x[id_tr]
                train_y = all_train_y[id_tr].to(device)
                val_x = all_train_x[id_val]
                val_y = all_train_y[id_val].to(device)

                # Generate the test data for mono and multigenetic diseases.
                ## 1. Collect data.
                print(f'Generate test data for fold {fold}.')
                test_x = dict()
                test_y = dict()
                test_x['mono'], test_y['mono'] = self.get_training_data_from_indexes(test_index, negatives, positives, cov_diseases, g_i_f_mapping, d_i_i_mapping, monogenetic_disease_only=True)
                test_y['mono'] = test_y['mono'].to(device)
                test_x['multi'], test_y['multi'] = self.get_training_data_from_indexes(test_index, negatives, positives, cov_diseases, g_i_f_mapping, d_i_i_mapping, multigenetic_diseases_only=True)
                test_y['multi'] = test_y['multi'].to(device)

                # Create the model
                model = gNetDGPModel(
                    gene_feature_dim=self.gene_net_data.x.shape[1],
                    disease_feature_dim=self.disease_net_data.x.shape[1],
                    fc_hidden_dim=int(config["fc_hidden_dim"]),
                    gene_net_hidden_dim=int(config["gene_net_hidden_dim"]),
                    disease_net_hidden_dim=int(config["disease_net_hidden_dim"]),
                    mode='DGP'
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
                print(f'Stat training fold {fold}/{folds}:')

                losses = dict()
                losses['train'] = list()
                losses['val'] = list()
                losses['mono'] = {
                    'AUC': 0,
                    'TPR': None,
                    'FPR': None
                }
                losses['multi'] = {
                    'AUC': 0,
                    'TPR': None,
                    'FPR': None
                }

                best_val_loss = 1e80
                for epoch in range(max_epochs):
                    # Train model.
                    model.train()
                    optimizer.zero_grad()
                    out = model(self.gene_net_data, self.disease_net_data, train_x)
                    loss = criterion(out, train_y)
                    loss.backward()
                    optimizer.step()
                    losses['train'].append(loss.item())

                    # Validation.
                    with torch.no_grad():
                        model.eval()
                        out = model(self.gene_net_data, self.disease_net_data, val_x)
                        loss = criterion(out, val_y)
                        current_val_loss = loss.item()
                        losses['val'].append(current_val_loss)

                        if epoch % info_each_epoch == 0:
                            print(
                                'Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}'.format(
                                    epoch, losses['train'][epoch], losses['val'][epoch]
                                )
                            )
                print("Fold: {}, train_loss: {}, val_loss: {}".format(fold+1, np.mean(losses['train']), np.mean(losses['val'])))
                all_losses['fold_train'].append(np.mean(losses['train']))
                all_losses['fold_val'].append(np.mean(losses['val']))

                with torch.no_grad():
                    for modus in ['multi', 'mono']:
                        predicted_probs = F.log_softmax(
                            model(self.gene_net_data, self.disease_net_data, test_x[modus]).clone().detach(), dim=1
                        )
                        true_y = test_y[modus]
                        fpr, tpr, _ = roc_curve(true_y.cpu().detach().numpy(), predicted_probs[:, 1].cpu().detach().numpy(),
                                            pos_label=1)
                        roc_auc = auc(fpr, tpr)
                        losses[modus]['TEST_Y'] = true_y.cpu().detach().numpy()
                        losses[modus]['TEST_PREDICT'] = predicted_probs.cpu().numpy()
                        losses[modus]['AUC'] = roc_auc
                        losses[modus]['TPR'] = tpr
                        losses[modus]['FPR'] = fpr
                        print(f'"{modus}" auc for fold {fold}: {roc_auc}')
                metrics.append(losses)
                
            print("All folds, all epochs, validation loss: {}".format(all_losses['fold_val']))
            print("Trial ended, train_loss: {}, val_loss: {}".format(np.mean(all_losses['fold_train']), np.mean(all_losses['fold_val'])))
            
            return {'loss': np.mean(all_losses['fold_val']), 'status': STATUS_OK, 'model': model}

        #@click.option('--fc_hidden_dim', default=3000)
        #@click.option('--gene_net_hidden_dim', default=830)
        #@click.option('--disease_net_hidden_dim', default=500)
        #lr=0.0005,
        #weight_decay=5e-4,
        #fc_hidden_dim=2048,
        #gene_net_hidden_dim=512,
        #disease_net_hidden_dim=512
            
        space_params = {
            'learning_rate': hp.loguniform('learning_rate', -5, -4),
            'weight_decay': hp.uniform('weight_decay', 0, 0.5),
            'fc_hidden_dim': hp.uniform('fc_hidden_dim', 128, 4096),
            'gene_net_hidden_dim': hp.uniform('gene_net_hidden_dim', 128, 1024),
            'disease_net_hidden_dim': hp.uniform('disease_net_hidden_dim', 128, 1024)
        }
        trials = Trials()
        SEED = 12345
        best=fmin(
            fn=optimise_params, # function to optimize
            space=space_params, 
            algo=tpe.suggest, # optimization algorithm
            max_evals=max_trials, # maximum number of trials
            trials=trials, # logging
            rstate=np.random.default_rng(SEED) # fixing random state for the reproducibility
        )
        print(trials)
        print("Index for best model: {}".format(np.argmin([r['loss'] for r in trials.results])))
        for i, res in enumerate(trials.results):
            print(i, res["loss"])
        print()
        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        torch.save(best_model.state_dict(), optimised_model_storage + '/best_optmised_model.ptm')
        print("Optimisation finished")