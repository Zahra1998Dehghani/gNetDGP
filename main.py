import click

from source.DiseaseNet import DiseaseNet
from source.GeneNet import GeneNet
from source.gNetDGPModel import gNetDGPModel
from source.HyperOpt import HyperOptmisation

@click.group()
def cli():
    pass

@click.command()
#@click.option('--fc_hidden_dim', default=3000)
#@click.option('--gene_net_hidden_dim', default=830)
#@click.option('--disease_net_hidden_dim', default=500)
@click.option('--folds', default=5)
@click.option('--max_epochs', default=500)
@click.option('--early_stopping_window', default=20)
#@click.option('--lr', default=0.00004, help='Learning rate')
#@click.option('--weight_decay', default=0.15)
@click.option('--gene_dataset_root', default='./data/gene_net')
@click.option('--disease_dataset_root', default='./data/disease_net')
@click.option('--training_data_path', default='./data/training/genes_diseases.tsv')
@click.option('--model_tmp_storage', default='/tmp')
@click.option('--results_storage', default='./out')
@click.option('--experiment_slug', default='train_generic')
def optimise_parameters(
            folds,
            max_epochs,
            early_stopping_window,
            gene_dataset_root,
            disease_dataset_root,
            training_data_path,
            model_tmp_storage,
            results_storage
):
    opt =  HyperOptmisation()
    opt.run_optimisation(folds,
        max_epochs,
        early_stopping_window,
        gene_dataset_root,
        disease_dataset_root,
        training_data_path,
        model_tmp_storage,
        results_storage
    )

@click.command()
@click.option('--fc_hidden_dim', default=3000)
@click.option('--gene_net_hidden_dim', default=830)
@click.option('--disease_net_hidden_dim', default=500)
@click.option('--folds', default=5)
@click.option('--max_epochs', default=500)
@click.option('--early_stopping_window', default=20)
@click.option('--lr', default=0.00004, help='Learning rate')
@click.option('--weight_decay', default=0.15)
@click.option('--gene_dataset_root', default='./data/gene_net')
@click.option('--disease_dataset_root', default='./data/disease_net')
@click.option('--training_data_path', default='./data/training/genes_diseases.tsv')
@click.option('--model_tmp_storage', default='/tmp')
@click.option('--results_storage', default='./out')
@click.option('--experiment_slug', default='train_generic')
def generic_train(
        fc_hidden_dim,
        gene_net_hidden_dim,
        disease_net_hidden_dim,
        folds,
        max_epochs,
        early_stopping_window,
        lr,
        weight_decay,
        gene_dataset_root,
        disease_dataset_root,
        training_data_path,
        model_tmp_storage,
        results_storage,
        experiment_slug
):
    print('Import modules')
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

    # 2. Generate negatives.
    # Pick equal amount of pairs not in the positives.
    negatives_list = []
    while len(negatives_list) < len(positives):
        gene_id = all_genes[np.random.randint(0, len(all_genes))]
        disease_id = covered_diseases[np.random.randint(0, len(covered_diseases))]
        if not ((positives['OMIM ID'] == disease_id) & (positives['EntrezGene ID'] == gene_id)).any():
            negatives_list.append([disease_id, gene_id])
    negatives = pd.DataFrame(np.array(negatives_list), columns=['OMIM ID', 'EntrezGene ID'])

    def get_training_data_from_indexes(indexes, monogenetic_disease_only=False, multigenetic_diseases_only=False):
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
        ## 2. Concat data.
        n = len(train_tuples)
        x_out = np.ones((n, 2))  # will contain (gene_idx, disease_idx) tuples.
        y_out = torch.ones((n,), dtype=torch.long)
        for i, (omim_id, gene_id, y) in enumerate(train_tuples):
            x_out[i] = (gene_id_index_feature_mapping[int(gene_id)], disease_id_index_feature_mapping[omim_id])
            y_out[i] = y
        return x_out, y_out

    def train(
            max_epochs,
            early_stopping_window=5,
            info_each_epoch=1,
            folds=5,
            lr=0.0005,
            weight_decay=5e-4,
            fc_hidden_dim=2048,
            gene_net_hidden_dim=512,
            disease_net_hidden_dim=512
    ):
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device))
        metrics = []
        dis_dict = {}
        fold = 0
        start_time = time.time()
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(covered_diseases):
            fold += 1
            print(f'Generate training data for fold {fold}.')
            all_train_x, all_train_y = get_training_data_from_indexes(train_index)

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
            test_x['mono'], test_y['mono'] = get_training_data_from_indexes(test_index, monogenetic_disease_only=True)
            test_y['mono'] = test_y['mono'].to(device)
            test_x['multi'], test_y['multi'] = get_training_data_from_indexes(test_index,
                                                                              multigenetic_diseases_only=True)
            test_y['multi'] = test_y['multi'].to(device)

            # Create the model
            model = gNetDGPModel(
                gene_feature_dim=gene_net_data.x.shape[1],
                disease_feature_dim=disease_net_data.x.shape[1],
                fc_hidden_dim=fc_hidden_dim,
                gene_net_hidden_dim=gene_net_hidden_dim,
                disease_net_hidden_dim=disease_net_hidden_dim,
                mode='DGP'
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                out = model(gene_net_data, disease_net_data, train_x)
                loss = criterion(out, train_y)
                loss.backward()
                optimizer.step()
                losses['train'].append(loss.item())

                # Validation.
                with torch.no_grad():
                    model.eval()
                    out = model(gene_net_data, disease_net_data, val_x)
                    loss = criterion(out, val_y)
                    current_val_loss = loss.item()
                    losses['val'].append(current_val_loss)

                    if epoch % info_each_epoch == 0:
                        print(
                            'Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}'.format(
                                epoch, losses['train'][epoch], losses['val'][epoch]
                            )
                        )
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        torch.save(model.state_dict(), osp.join(model_tmp_storage, f'best_model_fold_{fold}.ptm'))

                # Early stopping
                if epoch > early_stopping_window:
                    # Stop if validation error did not decrease
                    # w.r.t. the past early_stopping_window consecutive epochs.
                    last_window_losses = losses['val'][epoch - early_stopping_window:epoch]
                    if losses['val'][-1] > max(last_window_losses):
                        print('Early Stopping!')
                        break

            # Test the model for the current fold.
            model.load_state_dict(
                torch.load(osp.join(model_tmp_storage, f'best_model_fold_{fold}.ptm'), map_location=device)
            )
            with torch.no_grad():
                for modus in ['multi', 'mono']:
                    predicted_probs = F.log_softmax(
                        model(gene_net_data, disease_net_data, test_x[modus]).clone().detach(), dim=1
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
                    print(f'"{modus}" auc for fold: {fold}: {roc_auc}')
            metrics.append(losses)

        print('Done!')
        return metrics, dis_dict, model

    # Define helpers for evaluation
    def negcum(rank_vec):
        rank_vec_cum = []
        prev = 0
        for x in rank_vec:
            if x == 0:
                prev += 1
                rank_vec_cum.append(prev)
            else:
                rank_vec_cum.append(prev)
        rank_vec_cum = np.array(rank_vec_cum)
        return rank_vec_cum

    disease_idx_to_omim_mapping = dict()
    for omim_id, disease_idx in disease_id_index_feature_mapping.items():
        disease_idx_to_omim_mapping[disease_idx] = omim_id

    gene_idx_entrez_id_mapping = dict()
    for entrez_id, gene_idx in gene_id_index_feature_mapping.items():
        gene_idx_entrez_id_mapping[gene_idx] = entrez_id

    def get_genes_assoc_to_omim_disease(omim_id):
        return positives[positives["OMIM ID"].isin([omim_id])]["EntrezGene ID"].values

    # Reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    metrics, dis_dict, model = train(
        max_epochs=max_epochs,
        early_stopping_window=early_stopping_window,
        folds=folds,
        lr=lr,
        weight_decay=weight_decay,
        fc_hidden_dim=fc_hidden_dim,
        gene_net_hidden_dim=gene_net_hidden_dim,
        disease_net_hidden_dim=disease_net_hidden_dim
    )

    # Store the results.
    if not os.path.exists(results_storage):
        os.makedirs(results_storage)
    with gzip.open(osp.join(results_storage, f'{experiment_slug}_metrics.pickle.gz'), mode='wb') as file:
        pickle.dump(metrics, file)
    with gzip.open(osp.join(results_storage, f'{experiment_slug}_dis_dict.pickle.gz'), mode='wb') as file:
        pickle.dump(dis_dict, file)


@click.command()
@click.option('--fc_hidden_dim', default=3000)
@click.option('--gene_net_hidden_dim', default=830)
@click.option('--disease_net_hidden_dim', default=500)
@click.option('--gene_dataset_root', default='./data/gene_net')
@click.option('--disease_dataset_root', default='./data/disease_net')
@click.option('--model_path', default='./model/generic_pre_trained_model_fold_1.ptm')
@click.option('--out_file', default='./generic_predict_results.tsv')
@click.option('--get_available_genes', is_flag=True, help='List available genes and exit')
@click.option('--get_available_diseases', is_flag=True, help='List available diseases and exit')
@click.option('--sort_result_by_score', default=True, help='Sort the result by predicted score. (Default is True)')
@click.argument('input_file')
def generic_predict(
        fc_hidden_dim,
        gene_net_hidden_dim,
        disease_net_hidden_dim,
        gene_dataset_root,
        disease_dataset_root,
        model_path,
        out_file,
        get_available_genes,
        get_available_diseases,
        sort_result_by_score,
        input_file
):
    """Predict pseudo probabilities for disease-gene prioritization.

    INPUT_FILE a file of tab separated gene, disease tuples. See example_predict_input.tsv
    """
    print('Import modules')
    import torch
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd

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

    if get_available_genes:
        all_genes = list(gene_id_index_feature_mapping.keys())
        print('Available genes:')
        for gene in all_genes:
            print(gene)
        return
    if get_available_diseases:
        all_diseases = list(disease_id_index_feature_mapping.keys())
        print('Available diseases:')
        for disease in all_diseases:
            print(disease)
        return

    model = gNetDGPModel(
        gene_feature_dim=gene_net_data.x.shape[1],
        disease_feature_dim=disease_net_data.x.shape[1],
        fc_hidden_dim=fc_hidden_dim,
        gene_net_hidden_dim=gene_net_hidden_dim,
        disease_net_hidden_dim=disease_net_hidden_dim,
        mode='DGP'
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Create input vector
    input_tuples = pd.read_csv(input_file, comment='#', sep='\t', header=None, names=['Gene ID', 'Omim Id'])
    n = len(input_tuples)
    x_in = np.ones((n, 2))  # will contain (gene_idx, disease_idx) tuples.
    for i, row in input_tuples.iterrows():
        x_in[i] = (gene_id_index_feature_mapping[int(row[0])], disease_id_index_feature_mapping[row[1]])

    print('Predict')
    model.eval()
    with torch.no_grad():
        predicted_probs = F.softmax(
            model(gene_net_data, disease_net_data, x_in).clone().detach(), dim=1
        )[:, -1:]
    input_tuples['Score'] = predicted_probs.cpu().numpy()

    if sort_result_by_score:
        input_tuples.sort_values(by=['Score'], inplace=True, ascending=False)

    input_tuples.to_csv(out_file, sep='\t', index=False)
    print(f'Results (stored to {out_file}):')
    print(input_tuples)


@click.command()
@click.option('--fc_hidden_dim', default=3000)
@click.option('--gene_net_hidden_dim', default=830)
@click.option('--disease_net_hidden_dim', default=500)
@click.option('--folds', default=5)
@click.option('--max_epochs', default=500)
@click.option('--early_stopping_window', default=20)
@click.option('--lr_classification', default=0.00000347821, help='Learning rate')
@click.option('--weight_decay_classification', default=0.5165618)
@click.option('--gene_dataset_root', default='./data/gene_net')
@click.option('--disease_dataset_root', default='./data/disease_net')
@click.option('--training_disease_genes_path', default='./data/training/genes_diseases.tsv')
@click.option('--training_disease_class_assignments_path', default='./data/training/extracted_disease_class_assignments.tsv')
@click.option('--model_tmp_storage', default='/tmp')
@click.option('--results_storage', default='./out')
@click.option('--pretrained_model_path', default='./model/generic_pre_trained_model_fold_1.ptm')
def specific_train(
    fc_hidden_dim,
    gene_net_hidden_dim,
    disease_net_hidden_dim,
    folds,
    max_epochs,
    early_stopping_window,
    lr_classification,
    weight_decay_classification,
    gene_dataset_root,
    disease_dataset_root,
    training_disease_genes_path,
    training_disease_class_assignments_path,
    model_tmp_storage,
    results_storage,
    pretrained_model_path
):
    print('Import modules')
    import gzip
    import pickle
    import os.path as osp
    import sklearn.metrics as skmetrics
    import torch
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import KFold, train_test_split

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

    print('load training data.')
    disease_genes = pd.read_table(
        training_disease_genes_path,
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

    # 2. Generate negatives.
    # Pick equal amount of pairs not in the positives.
    negatives_list = []
    while len(negatives_list) < len(positives):
        gene_id = all_genes[np.random.randint(0, len(all_genes))]
        disease_id = covered_diseases[np.random.randint(0, len(covered_diseases))]
        if not ((positives['OMIM ID'] == disease_id) & (positives['EntrezGene ID'] == gene_id)).any():
            negatives_list.append([disease_id, gene_id])
    negatives = pd.DataFrame(np.array(negatives_list), columns=['OMIM ID', 'EntrezGene ID'])

    # Disease classification data preparation.
    # Load the disease classes.
    disease_class_training_data = pd.read_csv(training_disease_class_assignments_path, sep='\t')
    # drop duplicates
    unique_labeled_disease_class_genes = disease_class_training_data.drop_duplicates()
    gene_id_node_index_df = pd.DataFrame(
        data=[(gene_id, node_index) for gene_id, node_index in gene_id_index_feature_mapping.items()],
        columns=['gene_id', 'node_index'])
    disease_id_node_index_df = disease_id_node_index_df = pd.DataFrame(
        data=[(disease_id, node_index) for disease_id, node_index in disease_id_index_feature_mapping.items()],
        columns=['disease_id', 'disease_node_index']
    )

    # Create the gene index
    # Join in the gene node indexes
    disease_class_training_data = pd.merge(
        unique_labeled_disease_class_genes,
        gene_id_node_index_df,
        left_on='gene_id',
        right_on='gene_id',
        validate='many_to_many'
    )

    disease_class_counts = disease_class_training_data['disease_class'].value_counts()
    disease_class_target_classes = [
        'Ophthamological',
        'Connective tissue',
        'Endocrine',
        'Skeletal',
        'Metabolic',
        'Cardiovascular',
        'Dermatological',
        'Renal',
        'Hematological',
        'Immunological',
        'Muscular',
        'Developmental'
    ]

    def get_negative_disease_class_data(pos_class, n):
        # n = n // 2
        return disease_class_training_data[disease_class_training_data['disease_class'] != pos_class].sample(
            n=n,
            random_state=42
        )

    def get_positive_disease_class_data(pos_class):
        return disease_class_training_data[disease_class_training_data['disease_class'] == pos_class].copy()

    def get_disease_class_training_data(pos_class):
        pos = get_positive_disease_class_data(pos_class)
        pos['label'] = 1
        neg = get_negative_disease_class_data(pos_class, len(pos))
        neg['label'] = 0
        data = pd.concat([pos, neg], ignore_index=True)
        x = data.iloc[:, 3:].values
        y = data.iloc[:, 4:5].values.ravel()

        return x, torch.tensor(y), data

    model = gNetDGPModel(
        gene_feature_dim=gene_net_data.x.shape[1],
        disease_feature_dim=disease_net_data.x.shape[1],
        fc_hidden_dim=fc_hidden_dim,
        gene_net_hidden_dim=gene_net_hidden_dim,
        disease_net_hidden_dim=disease_net_hidden_dim
    ).to(device)

    def train_disease_classification(model_parameter_file):
        # Load the pretrained model.
        model.load_state_dict(torch.load(model_parameter_file, map_location=device))

        # Set classification training hyperparameters.
        info_each_epoch = 1
        final_disease_class_metrics = dict()
        losses = {
            'train': [],
            'val': [],
            'AUC': 0,
            'TPR': None,
            'FPR': None
        }
        for disease_class in disease_class_target_classes:
            for fold in range(folds):
                losses[f'train_disease_class_{disease_class}_{fold}'] = []
                losses[f'val_disease_class_{disease_class}_{fold}'] = []
                final_disease_class_metrics[f'{disease_class}_{fold}'] = {
                    'roc_auc': 0,
                    'pr_auc': 0,
                    'fmax': 0
                }

        torch.save(model.state_dict(), osp.join(model_tmp_storage, 'tmp_model_state.ptm'))
        class_count = 0
        for disease_class in disease_class_target_classes:
            class_count += 1
            print(
                f'Evaluate pretrained model on disease class {disease_class} ({class_count}/{len(disease_class_target_classes)})')
            x_disease_class, y_disease_class, _ = get_disease_class_training_data(disease_class)
            optimizer_disease_class = torch.optim.Adam(model.parameters(), lr=lr_classification,
                                                       weight_decay=weight_decay_classification)
            criterion_disease_class = torch.nn.CrossEntropyLoss()

            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            fold = -1
            for train_fold_index, test_fold_index in kf.split(x_disease_class):
                fold += 1
                print(f'Starting Fold: {fold}')
                model.load_state_dict(
                    torch.load(osp.join(model_tmp_storage, 'tmp_model_state.ptm'), map_location=device)
                )
                model.mode = 'Classify'
                # Split into train and validation.
                x_test = x_disease_class[test_fold_index]
                y_test = y_disease_class[test_fold_index].to(device)
                id_tr, id_val = train_test_split(range(x_disease_class[train_fold_index].shape[0]), test_size=0.1,
                                                 random_state=42)
                x_train = x_disease_class[train_fold_index][id_tr]
                y_train = y_disease_class[train_fold_index][id_tr].to(device)
                x_val = x_disease_class[train_fold_index][id_val]
                y_val = y_disease_class[train_fold_index][id_val].to(device)

                best_val_loss = 1e80
                for epoch in range(max_epochs):
                    model.train()

                    batch_size = 16
                    permutation = torch.randperm(x_train.shape[0])
                    # train
                    loss_items = []
                    for i in range(0, x_train.shape[0], batch_size):
                        batch_indices = permutation[i:i + batch_size]
                        batch_x, batch_y = x_train[batch_indices].reshape(-1, 2), y_train[batch_indices]

                        optimizer_disease_class.zero_grad()
                        out = model(gene_net_data, disease_net_data, batch_x)
                        loss = criterion_disease_class(out, batch_y)
                        loss.backward()
                        optimizer_disease_class.step()
                        loss_items.append(loss.item())
                    losses[f'train_disease_class_{disease_class}_{fold}'].append(np.mean(loss_items))

                    # validation
                    with torch.no_grad():
                        model.eval()
                        out = model(gene_net_data, disease_net_data, x_val)
                        loss = criterion_disease_class(out, y_val)
                        losses[f'val_disease_class_{disease_class}_{fold}'].append(loss.item())

                        if epoch % info_each_epoch == 0:
                            print(
                                'Epoch {}, train_loss: {:.4f}, val_loss: {:.4f}'.format(
                                    epoch, losses[f'train_disease_class_{disease_class}_{fold}'][epoch],
                                    losses[f'val_disease_class_{disease_class}_{fold}'][epoch]
                                )
                            )
                        if loss < best_val_loss:
                            best_val_loss = loss
                            torch.save(
                                model.state_dict(),
                                osp.join(
                                    results_storage,
                                    f'best_specific_model_{disease_class}_fold_{fold}.ptm'
                                )
                            )

                    # Early stopping
                    if epoch > early_stopping_window:
                        # Stop if validation error did not decrease
                        # w.r.t. the past early_stopping_window consecutive epochs.
                        last_window_losses = losses[f'val_disease_class_{disease_class}_{fold}'][
                                             epoch - early_stopping_window:epoch]
                        if losses[f'val_disease_class_{disease_class}_{fold}'][-1] > max(last_window_losses):
                            print('Early Stopping!')
                            break

                # Test the disease classification model for current fold.
                print(f'Test the model on fold {fold}:')
                with torch.no_grad():
                    y_score = model(gene_net_data, disease_net_data, x_test)[:, 1].cpu().detach().numpy()
                    y = y_test.cpu().detach().numpy()
                    final_disease_class_metrics[f'{disease_class}_{fold}']['roc_auc'] = skmetrics.roc_auc_score(
                        y,
                        y_score
                    )
                    precision, recall, thresholds = skmetrics.precision_recall_curve(y, y_score)
                    final_disease_class_metrics[f'{disease_class}_{fold}']['pr_auc'] = skmetrics.auc(
                        recall,
                        precision
                    )
                    final_disease_class_metrics[f'{disease_class}_{fold}']['fmax'] = (
                                (2 * precision * recall) / (precision + recall + 0.00001)).max()
                    print(final_disease_class_metrics[f'{disease_class}_{fold}'])
        return final_disease_class_metrics

    all_final_disease_class_metrics = []
    results_file = osp.join(results_storage, f'specific_mode_results.gz')
    print('############################################################')
    print(f'# START TRAINING USING PRETRAINED MODEL: {pretrained_model_path} #')
    print('############################################################')
    results = train_disease_classification(
        pretrained_model_path
    )
    all_final_disease_class_metrics.append(results)
    with gzip.open(results_file, mode='wb') as f:
        pickle.dump(all_final_disease_class_metrics, f)


@click.command()
@click.option('--fc_hidden_dim', default=3000)
@click.option('--gene_net_hidden_dim', default=830)
@click.option('--disease_net_hidden_dim', default=500)
@click.option('--gene_dataset_root', default='./data/gene_net')
@click.option('--disease_dataset_root', default='./data/disease_net')
@click.option('--model_path', default='./model/generic_pre_trained_model_fold_1.ptm')
@click.option('--out_file', default='./specific_predict_results.tsv')
@click.option('--get_available_genes', is_flag=True, help='List available genes and exit.')
@click.option('--sort_result_by_score', default=True, help='Sort the result by predicted score. (Default is True)')
@click.argument('input_file')
def specific_predict(
    fc_hidden_dim,
    gene_net_hidden_dim,
    disease_net_hidden_dim,
    gene_dataset_root,
    disease_dataset_root,
    model_path,
    out_file,
    get_available_genes,
    sort_result_by_score,
    input_file
):
    """Predict pseudo probabilities for specific disease associations.

        \b
        MODEL      Path to the pretrained model to be used for prediction.
        INPUT_FILE A entrez gene IDs (one per line).
                   See e.g. test/example_input_specific_predict.tsv
        """
    print('Import modules')
    import torch
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd

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

    if get_available_genes:
        all_genes = list(gene_id_index_feature_mapping.keys())
        print('Available genes:')
        for gene in all_genes:
            print(gene)
        return

    model = gNetDGPModel(
        gene_feature_dim=gene_net_data.x.shape[1],
        disease_feature_dim=disease_net_data.x.shape[1],
        fc_hidden_dim=fc_hidden_dim,
        gene_net_hidden_dim=gene_net_hidden_dim,
        disease_net_hidden_dim=disease_net_hidden_dim,
        mode='DGP'
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.mode = 'Classify'

    # Create input vector
    input_genes = pd.read_csv(input_file, comment='#', sep='\t', header=None, names=['Gene ID'])
    n = len(input_genes)
    x_in = np.ones((n, 2))  # will contain gene_idx
    for i, row in input_genes.iterrows():
        x_in[i] = (gene_id_index_feature_mapping[int(row[0])], 0)

    print('Predict')
    model.eval()
    with torch.no_grad():
        predicted_probs = F.softmax(
            model(gene_net_data, disease_net_data, x_in).clone().detach(), dim=1
        )[:, -1:]
    input_genes['Score'] = predicted_probs.numpy()

    if sort_result_by_score:
        input_genes.sort_values(by=['Score'], inplace=True, ascending=False)

    input_genes.to_csv(out_file, sep='\t', index=False)
    print(f'Results (stored to {out_file}):')
    print(input_genes)


if __name__ == '__main__':
    cli.add_command(generic_train)
    cli.add_command(generic_predict)
    cli.add_command(specific_train)
    cli.add_command(specific_predict)
    cli()
