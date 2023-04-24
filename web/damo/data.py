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

import pywt

## Load Data 

path = pd.DataFrame(glob(r'dataPVC/*'))
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

        ecg, ind, pvc = preprocess_ecg(ecg_array, ind_array, pvc_array)
        
        sample_rate = 360
        time_ecg = (round(len(ecg)/sample_rate))
        min_ecg = time_ecg/60

        for i, p in enumerate(pvc):
            if p != 0 | p != 1:
                pvc = np.delete(pvc,i)
                ind = np.delete(ind,i)
        
        global_ecg.loc[i] = [ecg,ind,pvc]

    return global_ecg

## filtro de linha de base, filtro de alta frequência e filtro de baixa frequência
# Remoção de ruído de linha de base
# Filtro passa-baixa
# Filtro passa-alta
# Normalização
# Remoção de picos R mal definidos
# Redução de artefatos de linha de base
# Detecção de anomalias
 
def preprocess_ecg(ecg_signal, ind_array, pvc):
    # 1. Remoção de ruído de linha de base
    # ecg_baseline = signal.medfilt(ecg_signal, kernel_size=31)
    # ecg_signal = ecg_signal - ecg_baseline
    
    # 2. Filtro passa-baixa
    # nyquist_freq = 0.5 * fs
    # low_cutoff = 5 / nyquist_freq
    # b, a = signal.butter(1, low_cutoff, btype='low')
    # ecg_signal = signal.filtfilt(b, a, ecg_signal)
    
    # 3. Filtro passa-alta
    # high_cutoff = 0.5 / nyquist_freq
    # b, a = signal.butter(1, high_cutoff, btype='high')
    # ecg_signal = signal.filtfilt(b, a, ecg_signal)
    
    # 4. Normalização
    max = ecg_signal.max()
    min = ecg_signal.min()    
    norm_ecg = np.array([(x-min)/(max-min)*10 for x in ecg_signal])

    # 5. Remoção de picos R mal definidos
    window_size = 8
    smoothed_ecg = np.convolve(norm_ecg.flatten(), np.ones((window_size,))/window_size, mode='valid')
    ind_array_clean = deepcopy(ind_array)
    pvc_array_clean = deepcopy(pvc)
    mal_class = []
    for i, peak in enumerate(ind_array):
        temp = (ecg_signal[peak-20:peak+20])
        if len(temp) > 0:
            if (temp.max() - temp.min()) < 2:
                mal_class.append(i)
                ind_array_clean = np.delete(ind_array_clean, mal_class)
                pvc_array_clean = np.delete(pvc_array_clean, mal_class)
    
    # 6. Detecção de anomalias
    anomalies = []
    mean = np.mean(smoothed_ecg)
    std = np.std(smoothed_ecg)
    for i in range(len(smoothed_ecg)):
        if (smoothed_ecg[i] < mean - 3 * std) or (smoothed_ecg[i] > mean + 3 * std):
            anomalies.append(i)
    smoothed_ecg[anomalies] = np.mean(smoothed_ecg)

    return smoothed_ecg, ind_array_clean, pvc_array_clean


global_ecg = globalDataFrame(global_ecg)

clf = svm.SVC()
model = RandomForestClassifier(n_estimators=100, random_state=42)

########################### 
for indice, linha in global_ecg.iterrows():
    ecg = linha['ECG']
    ind = linha['IND']
    pvc = linha['PVC']

    coeffs = pywt.wavedec(ecg, 'db4', level=6)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = np.std(cD1)
    peaks, _ = sig.find_peaks(cD1, distance=200, height=threshold)

    # Etapa 3: Encontrar os picos R marcados na coluna do DataFrame correspondente
    r_peak_indices = [int(idx) for idx in ind]

    # Etapa 4: Separar as features das etiquetas e dividir o conjunto de dados em conjuntos de treinamento e teste
    X = np.array(peaks).reshape(-1, 1)
    y = np.array(pvc)

    y = y[:len(X)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)

    model.fit(X_train, y_train)
    clf.fit(X_train, y_train)
