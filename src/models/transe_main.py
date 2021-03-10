#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  29 11:26:45 2021

@author: Jaime Enriquez Ballesteros, @ebjaime
"""

"""
@inproceedings{han2018openke,
  title={OpenKE: An Open Toolkit for Knowledge Embedding},
  author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
  booktitle={Proceedings of EMNLP},
  year={2018}
}
"""

import os

## OpenKE ##
from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransE
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader

# =============================================================================
#  PARAMETERS
# =============================================================================

# Sampling parameters (Train)
nbatches = 100 # Number of batches when sampling during training
threads = 8
bern = 1 # ??? "method" in original script
filter = 1 # ???
negative_relations = 0 # Corrupt relations are not needed

# Sampling parameters (Test)
sampling_mode = "link"


# Training parameters
dataset = "Movielens1M"
d = 100 # Dimension
gamma = 1
lr = 0.001 # Learning rate
epochs = 1000 

# Normalization parameters
p_norm = 1 # order of the norm for score
norm_flag = True # Flag for triple nomralization



# =============================================================================
#  DATA LOADERS
# =============================================================================

# Dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../../data/%s/" % dataset, 
	nbatches = nbatches,
	threads = threads, 
	sampling_mode = "normal", 
	bern_flag = bern, 
	filter_flag = filter, 
# 	neg_ent = 25, # Will be declared after calculating batch_size
	neg_rel = negative_relations)

# Set number of corrupt entities equal to batch size. Each thread will create (batch size / threads)
train_dataloader.set_ent_neg_rate(train_dataloader.get_batch_size() / threads) 

# Dataloader for test
test_dataloader = TestDataLoader("../../data/%s/" % dataset, sampling_mode)

# =============================================================================
#  MODEL
# =============================================================================

transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = d, 
	p_norm = p_norm, 
	norm_flag = norm_flag)


# =============================================================================
#  LOSS FUNCTION
# =============================================================================

# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = gamma),
	batch_size = train_dataloader.get_batch_size()
)


# =============================================================================
#  TRAINING
# =============================================================================
# If the model does not yet exist
if os.path.isfile('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr)) is False:    
    
    trainer = Trainer(model = model, 
                      data_loader = train_dataloader, 
                      train_times = epochs, 
                      alpha = lr, # Learning rate
                      opt_method="sgd",
                      use_gpu = True)
    
    # Train the model
    trainer.run()
    try:
        transe.save_checkpoint('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr))
    except FileNotFoundError:
        os.system("mkdir ../checkpoint")
        transe.save_checkpoint('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr))
    
    
# =============================================================================
#  TESTING
# =============================================================================
# Test the model
transe.load_checkpoint('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr))
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
