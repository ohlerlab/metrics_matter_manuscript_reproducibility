# Mimic distinct levels of batch correction
# Author: Pia Rautenstrauch
# Created: 1st of November, 2024

### For one random seed (0)
# liam_env_no_defaults environment

# Imports
import liam
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import anndata as ad
import os
import scvi

print(torch.cuda.is_available())

# Setup model to param mapping
model_param_mapping = {}

model_param_mapping['Liam_x1_minimal_example'] = {}
model_param_mapping['Liam_x1_minimal_example']['setup_anndata_params'] = {'batch_key': 'sample', 'rna_only': True}
model_param_mapping['Liam_x1_minimal_example']['Liam_params'] = {'adversarial_training': True, 'n_latent': 20,
                                                     'rna_only': True, 'factor_adversarial_loss': 1.0}

model_param_mapping['Liam_x5_minimal_example'] = {}
model_param_mapping['Liam_x5_minimal_example']['setup_anndata_params'] = {'batch_key': 'sample', 'rna_only': True}
model_param_mapping['Liam_x5_minimal_example']['Liam_params'] = {'adversarial_training': True, 'n_latent': 20,
                                                      'rna_only': True, 'factor_adversarial_loss': 5.0}

# Train models
for model in model_param_mapping.keys():
    print(model)
    seed = 0
    scvi._settings.ScviConfig(seed=seed)
    
    # Load data
    adata = ad.read_h5ad('data/original/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
    adata = adata[:,adata.var['feature_types'] == 'GEX'].copy()
    
    adata = adata[(adata.obs['Samplename'] == 'site1_donor1_multiome') | (adata.obs['Samplename'] == 'site1_donor3_multiome') | (adata.obs['Samplename'] == 'site4_donor8_multiome') | (adata.obs['Samplename'] == 'site4_donor9_multiome')].copy()

    # Setup input object
    adata.X = adata.layers["counts"].tocsr()
    adata.obs["sample"] = adata.obs["batch"]
    adata.obs["donor"] = adata.obs["batch"].apply(lambda x: x.split("d")[1])
    adata.obs["site"] = adata.obs["batch"].apply(lambda x: x.split("d")[0])
    
    # Setting up data.
    liam.Liam.setup_anndata(
            adata,
        **model_param_mapping[model]['setup_anndata_params'],
        )

    vae = liam.Liam(adata, **model_param_mapping[model]['Liam_params'])


    # Training Liam models
    vae.train(train_size=0.95, validation_size=0.05,
                  batch_size=128, early_stopping=True, save_best=True, early_stopping_patience=10)

    adata.obsm["embedding"] = vae.get_latent_representation()

    # Save embedding
    embedding = ad.AnnData(
        X=adata.obsm['embedding'],
        obs=adata.obs,
        uns={
            'dataset_id': adata.uns['dataset_id'],
            'method_id': model,
        }
    )
    embedding.write_h5ad("embeddings/{}.embedding.h5ad".format(model), compression="gzip")

    # For reproducibility
    print('Model: {}.'.format(model))
    print("Model's state_dict:")
    for param_tensor in vae.module.state_dict():
        print(param_tensor, '\t', vae.module.state_dict()[param_tensor].size())
    
    del vae
    del adata
    

# For reproducibility across all trained models
print('! nvidia-smi')
print(os.system('nvidia-smi')) 
print('torch.cuda.get_device_name()')
print(torch.cuda.get_device_name())
print('torch.version.cuda')
print(torch.version.cuda)
print('torch.cuda.get_device_capability()')
print(torch.cuda.get_device_capability())
print('torch.cuda.get_device_properties(torch.device)')
print(torch.cuda.get_device_properties(torch.device))
print('torch.cuda.get_arch_list()')
print(torch.cuda.get_arch_list())