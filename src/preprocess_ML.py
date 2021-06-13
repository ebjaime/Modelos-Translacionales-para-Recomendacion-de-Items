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
import pandas as pd
import numpy as np

dataset = "Movielens1M"
train_pct = 0.7
val_pct = 0.1
test_pct = 0.2


# Lectura de atributos
dbo_directors = pd.read_csv("../data/%s/graphs/dbo:director.edgelist" % dataset,
                            header=None,
                            sep=" ")
dbo_directors.columns = ["Movie", "Director"]
dbo_directors = dbo_directors[["Director", "Movie"]] 

dbo_starring = pd.read_csv("../data/%s/graphs/dbo:starring.edgelist"% dataset,
                            header=None,
                            sep=" ")
dbo_starring.columns = ["Movie", "Actor"]
dbo_starring = dbo_starring[["Actor", "Movie"]]

dbo_distributor = pd.read_csv("../data/%s/graphs/dbo:distributor.edgelist"% dataset,
                            header=None,
                            sep=" ")
dbo_distributor.columns = ["Movie", "Distributor"]
dbo_distributor = dbo_distributor[["Distributor","Movie"]]

dbo_writer = pd.read_csv("../data/%s/graphs/dbo:writer.edgelist"% dataset,
                            header=None,
                            sep=" ")
dbo_writer.columns = ["Movie", "Writer"]
dbo_writer = dbo_writer[["Writer", "Movie"]]

dbo_musicComposer = pd.read_csv("../data/%s/graphs/dbo:musicComposer.edgelist"% dataset,
                            header=None,
                            sep=" ")
dbo_musicComposer.columns = ["Movie", "MusicComposer"]
dbo_musicComposer= dbo_musicComposer[["MusicComposer", "Movie"]]

dbo_producer = pd.read_csv("../data/%s/graphs/dbo:producer.edgelist"% dataset,
                            header=None,
                            sep=" ")
dbo_producer.columns = ["Movie", "Producer"]
dbo_producer = dbo_producer[["Producer", "Movie"]]

dbo_cinematography = pd.read_csv("../data/%s/graphs/dbo:cinematography.edgelist"% dataset,
                            header=None,
                            sep=" ")
dbo_cinematography.columns = ["Movie", "Cinematography"]
dbo_cinematography = dbo_cinematography[["Cinematography", "Movie"]]


# Lectura de reseñas
feedback = pd.read_csv("../data/%s/original/ratings.dat" % dataset,
                        header=None,
                        sep="::")
feedback.columns = ["UserID", "MovieID", "Rating", "Timestamp"]

# Traduccion MovieID a Movie en reseñas
mappings = pd.read_csv("../data/%s/original/mappings.tsv" % dataset,
                       header=None,
                       sep="\t")
mappings.columns = ["MovieID", "MovieTitle", "Movie"]

feedback = feedback.merge(mappings[["MovieID", "Movie"]], on="MovieID")

# Filtracion de solo ratings>=4
feedback_relevant = feedback.drop(feedback.loc[feedback.Rating<4].index)
feedback_relevant.drop(["Rating", "Timestamp", "MovieID"], axis=1, inplace=True)

# Creacion entity2id.txt y relation2id.txt
directors = dbo_directors["Director"].unique()
actors = dbo_starring["Actor"].unique()
distributors = dbo_distributor["Distributor"].unique()
writers = dbo_writer["Writer"].unique()
musicComposers = dbo_musicComposer["MusicComposer"].unique()
producers = dbo_producer["Producer"].unique()
cinematographies = dbo_cinematography["Cinematography"].unique()


# all = pd.read_csv("../data/%s/train_test_entity2rec/all.dat" % dataset, header=None, sep=" ")

users = feedback["UserID"].unique() #all[0].unique()
movies = {*feedback["Movie"].unique(), *dbo_directors["Movie"].unique(), *dbo_starring["Movie"].unique(),
          *dbo_distributor["Movie"].unique(),*dbo_writer["Movie"].unique(), *dbo_musicComposer["Movie"].unique(),
          *dbo_producer["Movie"].unique(), *dbo_cinematography["Movie"].unique()}


relations_list = ["feedback", "dbo:director", "dbo:starring", "dbo:distributor", "dbo:writer", "dbo:musicComposer",
                  "dbo:producer", "dbo:cinematography"]
relations = {}
with open("../data/%s/relation2id.txt" % dataset, "w") as f:
    f.write(str(len(relations_list))+"\n")
    id=0
    for relation in relations_list:
        f.write(str(relation)+"\t"+str(id)+"\n")
        relations[str(relation)] = id
        id+=1

entities_list = {*users, *directors, *actors, *distributors, *writers, *musicComposers, 
                 *producers, *cinematographies, *movies}
entities = {}
with open("../data/%s/entity2id.txt" % dataset, "w") as f:
    f.write(str(len(entities_list))+"\n")
    id=0
    for entity in entities_list:
        f.write(str(entity)+"\t"+str(id)+"\n")
        entities[str(entity)] = id
        id+=1


# Creacion train2id.txt, test2id.txt y val2id.txt
feedback_np = feedback_relevant.values
np.random.shuffle(feedback_np)
train = {}
train["feedback"] = feedback_np[:int(train_pct*len(feedback_np))]
val = {}
val["feedback"] = feedback_np[int(train_pct*len(feedback_np)) : int(train_pct*len(feedback_np))+int(val_pct*len(feedback_np))]
test = {}
test["feedback"] = feedback_np[int(train_pct*len(feedback_np)) + int(val_pct*len(feedback_np)):]

# Solo el conjunto de entrenamiento contendra las triplas no feedback
for rel,vals in zip(relations_list[1:], [dbo_directors.values, dbo_starring.values, dbo_distributor.values,
                                     dbo_writer.values, dbo_musicComposer.values, dbo_producer.values, dbo_cinematography.values]):
    train[rel] = vals

# Creacion de cada conjunto de datos
sets = [] # Train, test, val

for set in [train, test, val]:
    set2id = []
    for rel in set:
        for tripla in set[rel]:
            head = str(tripla[0])
            tail = str(tripla[1])
            set2id.append((str(entities[head]), str(entities[tail]), str(relations[rel])))
    sets.append(set2id)

# Insertacion de cada conjunto en su fichero correspondiente
train2id =  open("../data/%s/train2id.txt" % dataset, "w")
test2id = open("../data/%s/test2id.txt" % dataset, "w")
val2id = open("../data/%s/valid2id.txt" % dataset, "w") 

for i, set in enumerate([train2id, test2id, val2id]):
    set.write(str(len(sets[i]))+"\n")
    for triple in sets[i]:
        set.write(triple[0]+"\t"+triple[1]+"\t"+triple[2]+"\n")

    set.close()

# Ejecutar n_n.py para crear 1-1.txt, 1-n.txt ....
os.system("cd ../data/%s; python n-n.py" % dataset)

