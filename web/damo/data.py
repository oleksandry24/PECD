from pathlib import Path
import signal
from scipy.io import loadmat
import pandas as pd
import numpy as np
from glob import glob
import scipy.signal as sig
from copy import deepcopy
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

import os

import pywt

## Load Data 


BASE_DIR = Path(__file__).resolve().parent.parent
directory = os.path.join(BASE_DIR, 'dataPVC')

path = pd.DataFrame(glob(directory+'/*'))
path.columns = ['data']

global_ecg = pd.DataFrame(columns=["ECG","IND","PVC"])

def globalDataFrame(global_ecg):

    for i in range(len(path)):
        load = loadmat(path['data'][i])
        dados = load['DAT'][0]
        df_dados = pd.DataFrame(dados)
        ecg_array = np.array(df_dados['ecg'][0])
        ind_array = np.array(df_dados['ind'][0])
        pvc_array = np.array(df_dados['pvc'][0])

        # Mudança de formatos de array de nested array para array com ints
        ecg_array = np.array([x[0] for x in ecg_array])
        ind_array = np.array([x[0] for x in ind_array])
        pvc_array = np.array([x[0] for x in pvc_array])
        fs = None
        ecg, ind, pvc = preprocess_ecg(ecg_array, ind_array, pvc_array, fs)

        for i, p in enumerate(pvc):
            if p != 0 | p != 1:
                pvc = np.delete(pvc,i)
                ind = np.delete(ind,i)
        
        global_ecg.loc[i] = [ecg,ind,pvc]

    return global_ecg

def preprocess_ecg(ecg_signal, ind_array, pvc, fs):
    # 1. Remoção de ruído de linha de base
    ecg_baseline = sig.medfilt(ecg_signal, kernel_size=31)
    ecg_signal = ecg_signal - ecg_baseline

    # 2. Filtro passa-baixa
    if fs is not None and fs != 0:
        nyquist_freq = 0.5 * fs
        low_cutoff = 5 / nyquist_freq
        b, a = sig.butter(1, low_cutoff, btype='low')
        ecg_signal = sig.filtfilt(b, a, ecg_signal)

    # 3. Filtro passa-alta
    if fs is not None and fs != 0:
        nyquist_freq = 0.5 * fs
        high_cutoff = 0.5 / nyquist_freq
        b, a = sig.butter(1, high_cutoff, btype='high')
        ecg_signal = sig.filtfilt(b, a, ecg_signal)

    # 4. Normalização
    max_value = np.max(ecg_signal)
    min_value = np.min(ecg_signal)    
    norm_ecg = (ecg_signal - min_value) / (max_value - min_value) * 10

    # 5. Remoção de picos R mal definidos
    window_size = 8
    smoothed_ecg = np.convolve(norm_ecg.flatten(), np.ones((window_size,))/window_size, mode='valid')
    ind_array_clean = []
    pvc_array_clean = []
    for i, peak in enumerate(ind_array):
        temp = ecg_signal[peak-20:peak+20]
        if len(temp) > 0:
            if np.max(temp) - np.min(temp) >= 2:  # Critério de amplitude mínima
                ind_array_clean.append(peak)
                pvc_array_clean.append(pvc[i])
    
    return smoothed_ecg, ind_array_clean, pvc_array_clean


global_ecg = globalDataFrame(global_ecg)

nb = GaussianNB()
clf = svm.LinearSVC()
model = RandomForestClassifier(n_estimators=100,random_state=42, max_depth=10, min_samples_split=5)

########################### 

for ind, row in global_ecg.iterrows():
    ecg = row['ECG']
    ind = row['IND']
    pvc = row['PVC']

    new_ecg = np.zeros(len(ind))

    for i in range(len(ind)):
        new_ecg[i] = ecg[ind[i]]

    X = new_ecg.reshape(-1,1)
    y = pvc

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)

    model.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    nb.fit(X_train, y_train)
