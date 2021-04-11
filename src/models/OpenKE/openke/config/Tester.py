# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

import pandas as pd

class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
    
    # =============================================================================
    #  @ebjaime - 10/03
    # =============================================================================
    def run_link_prediction(self, type_constrain = False):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
            
        training_range = tqdm(self.data_loader)
        
        num_relevant_documents_in_test = self.lib.getTestTotal()
        
        h_at_10 = 0
        h_at_5 = 0
        
        mean_avg_prec = 0
        
        # Obtener cinco y diez productos mas populares para calcular SER@n
        ten_most_popular, item_ids = self.k_most_popular(10)
        five_most_popular = ten_most_popular[:5]
        
        ser_at_5 = 0
        ser_at_10 = 0
        
        ndcg = 0
        
        for index, [data_head, data_tail] in enumerate(training_range):
            
            # Se seleccionan solo items que son peliculas
            data_tail["batch_t"] = item_ids
            
            # score = self.test_one_step(data_head)
            # self.lib.testHead(score.__array_interface__["data"][0], index, type_constrain)
            score = self.test_one_step(data_tail)
            # self.lib.testTail(score.__array_interface__["data"][0], index, type_constrain)
            
            # Menor score es mejor recomendacion
            score_sorted = item_ids[np.argsort(score)]
                        
            
            count_unrated = 0 # Contador de items unrated
            count_relevant = 0 # Contador de items relevantes
            
            map_aux = 0 # Calculo de MAP intermedio
            
            dcg = 0 # discounted cumulative gain
            
            for score in score_sorted:
                # Si es unrated (no pertenece al conjunto de entrenamiento ni validación) segun AllUnratedItems
                unrated = not self.lib._find_train(int(data_head['batch_h'][0]), int(score), int(data_head['batch_r'][0])) \
                            and not self.lib._find_val(int(data_head['batch_h'][0]), int(score), int(data_head['batch_r'][0]))
                
                if unrated:
                    count_unrated+=1
                    # Si es relevante pertenece al conjunto de test
                    relevant = self.lib._find_test(int(data_head['batch_h'][0]), int(score), int(data_head['batch_r'][0]))
                    if relevant:
                        count_relevant+=1 
                        dcg += 1/np.log2(count_unrated + 1)
                        
                        if count_unrated <= 5:
                            h_at_5 += 1
                            if score not in five_most_popular:
                                ser_at_5 += 1
                             
                        if count_unrated <= 10:
                            h_at_10 += 1
                            if score not in ten_most_popular:
                                ser_at_10 += 1
    
                        map_aux += count_relevant/(count_unrated+1) 

            if count_relevant != 0:
                mean_avg_prec += map_aux / count_relevant
                
            idcg = sum([1/np.log2(pos + 1) for pos in range(count_relevant)])
            ndcg += dcg/idcg
            
        
        p_at_5 = h_at_5 / (index * 5) 
        p_at_10 = h_at_10 / (index * 10)
        
        r_at_5 = h_at_5 / num_relevant_documents_in_test
        r_at_10 = h_at_10 /num_relevant_documents_in_test
        
        mean_avg_prec /= index
        
        ser_at_5 /= (index * 5)
        ser_at_10 /= (index * 10)
        
        ndcg /= len(training_range)
        
        print("\n\nP@5: ",p_at_5)
        print("P@10: ",p_at_10)  
        print("R@5: ",r_at_5)
        print("R@10: ",r_at_10)            
        print("MAP: ", mean_avg_prec)
        print("SER@5: ",ser_at_5)
        print("SER@10: ",ser_at_10) 
        print("NDCG: ", ndcg)
        
        return p_at_5, p_at_10, r_at_5, r_at_10, mean_avg_prec, ser_at_5, ser_at_10, ndcg
        
               

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod
    
    # K mas populares con relacion = 0 en datos de entrenamiento
    def k_most_popular(self, k):
        path_train = self.data_loader.get_path() + "train2id.txt"
        path_val = self.data_loader.get_path() + "valid2id.txt"
        data = pd.read_csv(path_train, 
                           sep="\t", 
                           header=None, 
                           names=["user_id","item_id","rel_id"],
                           skiprows=1)
        data = pd.concat([data, pd.read_csv(path_val, 
                           sep="\t", 
                           header=None, 
                           names=["user_id","item_id","rel_id"],
                           skiprows=1)])
        
        data_grouped = data.loc[data.rel_id == 0].groupby(["item_id"])
        k_sorted_indices = np.argsort(data_grouped.size())[::-1].values[:k]
        sorted_item_ids = np.sort(data.loc[data.rel_id==0]["item_id"].unique())
        
        return sorted_item_ids[k_sorted_indices], sorted_item_ids
    
    