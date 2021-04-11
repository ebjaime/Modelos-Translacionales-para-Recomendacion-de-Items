#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:56:54 2021

@author: Jaime Enriquez Ballesteros, @ebjaime
"""

"""
@inproceedings{han2018openke,
  title={OpenKE: An Open Toolkit for Knowledge Embedding},
  author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
  booktitle={Proceedings of EMNLP},
  year={2018}
}

@book{Celma:Springer2010,
      	author = {Celma, O.},
      	title = {{Music Recommendation and Discovery in the Long Tail}},
       	publisher = {Springer},
       	year = {2010}
      }

http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = "LastFM"
train_pct = 0.7
val_pct = 0.1
test_pct = 0.2

# Lectura de atributos
dbo_associated = pd.read_csv("../data/%s/graphs/dbo:associatedBand.edgelist" % dataset,
                             header=None,
                             sep=" ")

dbo_associated = pd.concat([dbo_associated, pd.read_csv("../data/%s/graphs/dbo:associatedMusicalArtist.edgelist" % dataset,
                                                        header=None,
                                                        sep=" ")]).drop_duplicates()

dbo_associated.columns = ["Author1", "Author2"]
dbo_associated = dbo_associated[["Author1", "Author2"]]


dbo_bandMember = pd.read_csv("../data/%s/graphs/dbo:bandMember.edgelist" % dataset,
                             header=None,
                             sep=" ")
dbo_bandMember.columns = ["Author1", "Author2"]
dbo_bandMember = dbo_bandMember[["Author1", "Author2"]]

dbo_formerBandMember = pd.read_csv("../data/%s/graphs/dbo:formerBandMember.edgelist" % dataset,
                                   header=None,
                                   sep=" ")
dbo_formerBandMember.columns = ["Author1", "Author2"]
dbo_formerBandMember = dbo_formerBandMember[["Author1", "Author2"]]

dbo_genre = pd.read_csv("../data/%s/graphs/dbo:genre.edgelist" % dataset,
                        header=None,
                        sep=" ")
dbo_genre.columns = ["Author", "Genre"]
dbo_genre = dbo_genre[["Author", "Genre"]]

dbo_hometown = pd.read_csv("../data/%s/graphs/dbo:hometown.edgelist" % dataset,
                           header=None,
                           sep=" ")
dbo_hometown.columns = ["Author", "Hometown"]
dbo_hometown = dbo_hometown[["Author", "Hometown"]]

dbo_birthPlace = pd.read_csv("../data/%s/graphs/dbo:birthPlace.edgelist" % dataset,
                             header=None,
                             sep=" ")
dbo_birthPlace.columns = ["Author", "Place"]
dbo_birthPlace = dbo_birthPlace[["Author", "Place"]]

dbo_instrument = pd.read_csv("../data/%s/graphs/dbo:instrument.edgelist" % dataset,
                             header=None,
                             sep=" ")
dbo_instrument.columns = ["Author", "Instrument"]
dbo_instrument = dbo_instrument[["Author", "Instrument"]]

dbo_occupation = pd.read_csv("../data/%s/graphs/dbo:occupation.edgelist" % dataset,
                             header=None,
                             sep=" ")
dbo_occupation.columns = ["Author", "Occupation"]
dbo_occupation = dbo_occupation[["Author", "Occupation"]]

dbo_recordLabel = pd.read_csv("../data/%s/graphs/dbo:recordLabel.edgelist" % dataset,
                              header=None,
                              sep=" ")
dbo_recordLabel.columns = ["Author", "Label"]
dbo_recordLabel = dbo_recordLabel[["Author", "Label"]]

dbo_subject = pd.read_csv("../data/%s/graphs/dct:subject.edgelist" % dataset,
                          header=None,
                          sep=" ")
dbo_subject.columns = ["Author", "Subject"]
dbo_subject = dbo_subject[["Author", "Subject"]]

# Lectura de reseñas
feedback = pd.read_csv("../data/%s/original/feedback.txt" % dataset,
                       header=None,
                       sep="\t")
feedback.columns = ["UserID", "AuthorID", "Listens"]

# Traduccion AuthorID a Author en reseñas
mappings = pd.read_csv("../data/%s/original/mappings.tsv" % dataset,
                       header=None,
                       sep="\t")
mappings.columns = ["AuthorID", "AuthorTitle", "Author"]

feedback = feedback.merge(mappings[["AuthorID", "Author"]], on="AuthorID")

# Filtracion de solo autores relevantes (dentro del 60%-100%)
# Graficar escuchas de usuario segun autor


def plot_user_listens(user_id=2):
    plt.plot(range(len(feedback[feedback.UserID == user_id])),
             feedback[feedback.UserID == user_id].sort_values(by="Listens", ascending=False)["Listens"].values)
    plt.title("Top artists by ranking for user " + str(user_id))
    plt.ylabel("Listens")
    plt.xlabel("Author rank")
    plt.show()

# Calculo de Funcion de Distribucion Acumulada Complementaria (CCDF) para cada usuario
def ccdf_listens(feedback=feedback, user_id=2, plot=False, cutoff=0.7):
    if user_id is None:  # Caso en el que feedback ya viene como DF de un solo usuario
        listens_user = feedback.sort_values(by="Listens")["Listens"].values
    else:
        listens_user = feedback[feedback.UserID == user_id].sort_values(by="Listens")[
            "Listens"].values

    cdf = np.cumsum(listens_user / listens_user.sum())[::-1]
    if plot:
        plt.plot(range(len(cdf)), cdf)
        plt.title("CCDF of listens for user " + str(user_id))
        plt.ylabel("%")
        plt.xlabel("Author rank")
        plt.show()
    return listens_user[::-1][cdf > cutoff]


# Filtrar usuarios que están dentro del mejor 40% para cada usuario
feedback_relevant_idx = feedback.groupby(["UserID"]).apply(lambda x:
                                                           x.loc[x["Listens"].isin(ccdf_listens(x, user_id=None))])

feedback_relevant = feedback.loc[feedback_relevant_idx.reset_index( level=["UserID"], drop=True).index]
feedback_relevant.drop(["Listens", "AuthorID"], axis=1, inplace=True)

# Creacion entity2id.txt y relation2id.txt
associated = dbo_associated["Author2"].unique()
band_members = dbo_bandMember["Author2"].unique()
former_band_members = dbo_formerBandMember["Author2"].unique()
genres = dbo_genre["Genre"].unique()
hometowns = dbo_hometown["Hometown"].unique()
birth_places = dbo_birthPlace["Place"].unique()
instruments = dbo_instrument["Instrument"].unique()
occupations = dbo_occupation["Occupation"].unique()
record_labels = dbo_recordLabel["Label"].unique()
subjects = dbo_subject["Subject"].unique()


# all = pd.read_csv("../data/%s/train_test_entity2rec/all.dat" % dataset, header=None, sep=" ")

users = feedback_relevant["UserID"].unique() #all[0].unique()
authors = feedback_relevant["UserID"].unique() #all[1].unique()


relations_list = ["feedback", "dbo:associated", "dbo:bandMember", "dbo:formerBandMember",
                  "dbo:genre", "dbo:hometown", "dbo:birthPlace", "dbo:instrument",
                  "dbo:occupation", "dbo:recordLabel", "dbo:subject"]
relations = {}
with open("../data/%s/relation2id.txt" % dataset, "w") as f:
    f.write(str(len(relations_list))+"\n")
    id = 0
    for relation in relations_list:
        f.write(str(relation)+"\t"+str(id)+"\n")
        relations[str(relation)] = id
        id += 1


entities_list = {*users, *authors, *associated, *band_members, *former_band_members, *genres,
                 *hometowns, *birth_places, *instruments, *occupations, *record_labels, *subjects}

entities = {}
with open("../data/%s/entity2id.txt" % dataset, "w") as f:
    f.write(str(len(entities_list))+"\n")
    id = 0
    for entity in entities_list:
        f.write(str(entity)+"\t"+str(id)+"\n")
        entities[str(entity)] = id
        id += 1


# Creacion train2id.txt, test2id.txt y val2id.txt
feedback_np = feedback_relevant.values
np.random.shuffle(feedback_np)
train = {}
train["feedback"] = feedback_np[:int(train_pct*len(feedback_np))]
val = {}
val["feedback"] = feedback_np[int(train_pct*len(feedback_np)): int(
    train_pct*len(feedback_np))+int(val_pct*len(feedback_np))]
test = {}
test["feedback"] = feedback_np[int(
    train_pct*len(feedback_np)) + int(val_pct*len(feedback_np)):]

# Solo el conjunto de entrenamiento contendra las triplas no feedback
for rel, vals in zip(relations_list[1:], [dbo_associated.values, dbo_bandMember.values, dbo_formerBandMember.values,
                                          dbo_genre.values, dbo_hometown.values, dbo_birthPlace.values, dbo_instrument.values,
                                          dbo_occupation.values, dbo_recordLabel.values, dbo_subject.values]):
    train[rel] = vals

# Creacion de cada conjunto de datos
sets = [] # Train, test, val

for set in [train, test, val]:
    set2id = []
    for rel in set:
        for tripla in set[rel]:
            head = str(tripla[0])
            if head not in entities: # En el caso que en relaciones no feedback haya artista no registradas en feedback
                entities[head] = id
                id+=1
            tail = str(tripla[1])
            if tail not in entities: # En el caso que en relaciones no feedback haya artista no registradas en feedback
                entities[tail] = id
                id+=1
            set2id.append((str(entities[head]), str(entities[tail]), str(relations[rel])))
    sets.append(set2id)

# Insertacion de cada conjunto en su fichero correspondiente
train2id = open("../data/%s/train2id.txt" % dataset, "w")
test2id = open("../data/%s/test2id.txt" % dataset, "w")
val2id = open("../data/%s/valid2id.txt" % dataset, "w")

for i, set in enumerate([train2id, test2id, val2id]):
    set.write(str(len(sets[i]))+"\n")
    for triple in sets[i]:
        set.write(triple[0]+"\t"+triple[1]+"\t"+triple[2]+"\n")

    set.close()

# Ejecutar n_n.py para crear 1-1.txt, 1-n.txt ....
os.system("cd ../data/%s; python n-n.py" % dataset)
