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

dataset = "Movielens1M"
gamma = 1
lr = 0.001 # Learning rate
d = 100 # Dimension
epochs = 1000 


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "../../data/%s/" % dataset, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("../../data/%s/" % dataset, "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = d, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = gamma),
	batch_size = train_dataloader.get_batch_size()
)

# If the model does not yet exist
if os.path.isfile('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr)) is False:    
    # train the model
    trainer = Trainer(model = model, 
                      data_loader = train_dataloader, 
                      train_times = epochs, 
                      alpha = lr, # Learning rate
                      opt_method="sgd",
                      use_gpu = True)
    trainer.run()
    try:
        transe.save_checkpoint('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr))
    except FileNotFoundError:
        os.system("mkdir ../checkpoint")
        transe.save_checkpoint('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr))
    
    
# test the model
transe.load_checkpoint('../checkpoint/transe_%s_d%d_e%d_lr%f.ckpt' % (dataset, d, epochs, lr))
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)