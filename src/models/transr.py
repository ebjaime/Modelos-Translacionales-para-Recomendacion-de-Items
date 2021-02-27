#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  29 11:26:45 2021

@author: Jaime Enriquez Ballesteros, @ebjaime
"""

import os

## OpenKE ##
from OpenKE.openke.config import Trainer, Tester
from OpenKE.openke.module.model import TransR
from OpenKE.openke.module.loss import MarginLoss
from OpenKE.openke.module.strategy import NegativeSampling
from OpenKE.openke.data import TrainDataLoader, TestDataLoader

dataset = "Movielens1M"
gamma = 1
lr = 0.001 # Learning rate
d_e = 10 # Dimension entity
d_r = 10 # Dimension relation
epochs = 1000 

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../data/%s/" % dataset, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("../data/%s/" % dataset, "link")

# define the model
transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = d_e,
    dim_r = d_r,
	p_norm = 1, 
	norm_flag = True,
    rand_init=True)


# define the loss function
model = NegativeSampling(
	model = transr, 
	loss = MarginLoss(margin = gamma),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, 
                  data_loader = train_dataloader, 
                  train_times = epochs, 
                  alpha = lr, # Learning rate,
                  use_gpu = True)

if os.path.isfile('../checkpoint/transr_%s_de%d_dr%d_e%d_lr%f.ckpt' % (dataset, d_e, d_r, epochs, lr)) is False:    
    trainer.run()
    try:
        transr.save_checkpoint('../checkpoint/transr_%s_de%d_dr%d_e%d_lr%f.ckpt' % (dataset, d_e, d_r, epochs, lr))
    except FileNotFoundError:
        os.system("mkdir ../checkpoint")
        transr.save_checkpoint('../checkpoint/transr_%s_de%d_dr%d_e%d_lr%f.ckpt' % (dataset, d_e, d_r, epochs, lr))
    
    
# test the model
transr.load_checkpoint('../checkpoint/transr_%s_de%d_dr%d_e%d_lr%f.ckpt' % (dataset, d_e, d_r, epochs, lr))
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)