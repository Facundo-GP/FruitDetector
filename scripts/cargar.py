import cv2
import os
from glob import glob
import numpy as np
import pandas as pd
import re



default = 'dataset/'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def read_img(path):
    img = cv2.imread(path)
    return img


class DataLoader():
    
    def __init__(self,DATA_DIR=default):
        
        dir_bananas = sorted(glob(os.path.join(DATA_DIR, "bananas/datos/*")), key = numericalSort)
        dir_ciruelas  = sorted(glob(os.path.join(DATA_DIR, "cerezas/datos/*")), key = numericalSort)
        dir_kiwis = sorted(glob(os.path.join(DATA_DIR, "kiwis/datos/*")), key = numericalSort)
        dir_mangos = sorted(glob(os.path.join(DATA_DIR, "mangos/datos/*")), key = numericalSort)
        dir_manzanas = sorted(glob(os.path.join(DATA_DIR, "manzanas/datos/*")), key = numericalSort)
        dir_naranjas = sorted(glob(os.path.join(DATA_DIR, "naranjas/datos/*")), key = numericalSort)

        labels_bananas = pd.read_csv(DATA_DIR + 'bananas/labels.csv').values
        labels_ciruelas = pd.read_csv(DATA_DIR + 'cerezas/labels.csv').values
        labels_kiwis = pd.read_csv(DATA_DIR + 'kiwis/labels.csv').values
        labels_mangos = pd.read_csv(DATA_DIR + 'mangos/labels.csv').values
        labels_manzanas = pd.read_csv(DATA_DIR + 'manzanas/labels.csv').values
        labels_naranjas = pd.read_csv(DATA_DIR + 'naranjas/labels.csv').values


        img_bananas = []
        img_ciruelas = []
        img_kiwis = []
        img_mangos = []
        img_manzanas = []
        img_naranjas = []

        for i in range(len(dir_bananas)):
            img_bananas.append(read_img(dir_bananas[i]))
            img_ciruelas.append(read_img(dir_ciruelas[i]))
            img_kiwis.append(read_img(dir_kiwis[i]))
            img_mangos.append(read_img(dir_mangos[i]))
            img_manzanas.append(read_img(dir_manzanas[i]))
            img_naranjas.append(read_img(dir_naranjas[i]))


        self.bananas_ds = [img_bananas,labels_bananas]
        self.ciruelas_ds = [img_ciruelas,labels_ciruelas]
        self.kiwis_ds = [img_kiwis,labels_kiwis]
        self.mangos_ds = [img_mangos,labels_mangos]
        self.manzanas_ds = [img_manzanas,labels_manzanas]
        self.naranjas_ds = [img_naranjas,labels_naranjas]
        
    def manzanas(self):
        return self.manzanas_ds
    
    def bananas(self):
        return self.bananas_ds
    
    def naranjas(self):
        return self.naranjas_ds
    
    def ciruelas(self):
        return self.ciruelas_ds
    
    def kiwis(self):
        return self.kiwis_ds
    
    def mangos(self):
        return self.mangos_ds
    
    def todos(self):
        return self.bananas_ds,self.ciruelas_ds,self.kiwis_ds,self.mangos_ds,self.manzanas_ds,self.naranjas_ds
    
    
