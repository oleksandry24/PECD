import scipy.signal as sig
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template import loader
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pywt
from scipy.io import loadmat
import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from django.contrib.auth import login as logg
from django.contrib.auth import logout as l
from django.contrib.auth import authenticate
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from copy import deepcopy
from io import BytesIO
import base64

from copy import deepcopy

import matplotlib.pyplot as plt
from .data import model, clf, preprocess_ecg

# https://youtube.com/shorts/X7i9VNKrHl0?feature=share

ind = None
ecg = None
pvc = None

pacient = None
first_name = None
last_name = None
beats = 0
fs = 0
pvcs = 0
user = None
pvch = 0
fig_path = ''
ritmo_medio = 0
f1_score = 0
average_pvc_ecg = 0
ciclicos_pvcs = 0

def welcome(request):
    return render(request, "damo/welcome.html")

@csrf_exempt
def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        # Verificar se as senhas coincidem
        if password != confirm_password:
            return render(request, 'damo/r_1.html')

        # Verificar se o usuário já existe
        if User.objects.filter(username=username).exists():
            return render(request, 'damo/r2.html')
        # Criar um novo usuário e salvar no banco de dados
        try:
            user = User.objects.create_user(username=username, first_name=first_name, last_name=last_name, password=password)
            return render(request,'damo/login.html')
        except:
            return render(request,'damo/r_3.html')
    return render(request, 'damo/register.html')

@csrf_exempt
def login(request):
    global last_name
    global first_name
    global user
    if request.method == 'POST':
        # Receber os dados do formulário HTML
        username = request.POST['username']
        password = request.POST['password']

        # Autenticar o usuário
        user = authenticate(request, username=username, password=password)
        # Verificar se as credenciais estão corretas
        if user is not None:
            # Criar uma sessão de usuário ativa
            logg(request, user)
            return render(request,'damo/home.html')
        else:
            return render(request,'damo/login_error.html')
    return render(request, 'damo/login.html')

def logout(request):
    if request.method == 'POST':
        l(request)
        return render(request, 'damo/welcome.html')
    return render(request,'damo/logout.html')

def home(request):
    global pacient
    return render(request, 'damo/home.html')

def new_home(request):
    global pacient
    template = loader.get_template('damo/new_home.html')
    return HttpResponse(template.render(request))

def paciente(request):

    global pacient
    pacient = None
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        pacient = fs.url(filename)
        print(pacient)
        
        allowed_extensions = {'mat'}
        if allowed_file(pacient, allowed_extensions):
            load = loadmat(myfile)
            if load == {}:
                return render (request, 'damo/p2.html')
            else:
                predict(myfile)
                return render(request, 'damo/new_home.html')
        else:
            pacient = None
            return render(request, 'damo/p_1.html')
    return render(request, 'damo/pacientes.html')

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def ecg(request):
    global pacient
    global fs 

    if pacient is None:
        return render(request, 'damo/escolha_fs_error.html')
    else:
        if request.method == 'POST':
            fs = request.POST['fs']
            if fs != '' and fs != '0':
                fs = int(request.POST['fs'])
                print(fs)
                return redirect(ecg_medio_fs)
            elif fs == '0':
                return render(request, 'damo/ecg_error.html')
            fs = request.POST['fs']
        return render(request, 'damo/escolha_fs.html')

def beat(request):
    global beats
    global pacient
    if pacient is None:
        template = loader.get_template('damo/beat_error.html')
    else:
        template = loader.get_template('damo/beat.html')

    context = {'result':beats}

    return HttpResponse(template.render(context,request))

def pvc(request):
    global pvcs
    global pacient
    if pacient is None:
        template = loader.get_template('damo/pvc_error.html')
    else:
        template = loader.get_template('damo/pvc.html')

    context = {'result' : pvcs}
    return HttpResponse(template.render(context,request))


def pvcH(request):
    global pacient
    global pvch
    if pacient is None:
        template = loader.get_template('damo/pvcsH_error.html')
    else:
        template = loader.get_template('damo/pvcsH.html')

    context = {'result' : pvch}

    return HttpResponse(template.render(context,request))


def f1(request):
    global pacient
    global f1_score
    if pacient is None:
        template = loader.get_template('damo/f1_error.html')
    else:
        template = loader.get_template('damo/f1.html')

    context = {'result' : float(f1_score)*100}

    return HttpResponse(template.render(context,request))

def ecg_medio(request):
    global pacient
    global fig_path
    fig_path = plt_ecg_medio()
    return render(request, 'damo/ecg-medio.html', {'result': fig_path})

def ecg_medio_fs(request):
    global pacient
    global fs
    global fig_path_fs
    fig_path_fs = plot_ecg_with_fs(fs)
    print(fig_path_fs)
    return render(request, 'damo/ecg-medio-fs.html', {'result': fig_path_fs})

def ritmo(request):
    global pacient
    global ritmo_medio
    if pacient is None:
        template = loader.get_template('damo/ritmo_error.html')
    else:
        template = loader.get_template('damo/ritmo.html')

    context = {'result' : round(ritmo_medio,4)}

    return HttpResponse(template.render(context,request))


def meanpvcs(request):
    global pvcs
    global pacient
    global average_pvc_ecg
    if pacient is None:
        template = loader.get_template('damo/meanpvcs_error.html')
    else:
        template = loader.get_template('damo/meanpvcs.html')

    context = {'result' : round(average_pvc_ecg,4)}
    return HttpResponse(template.render(context,request))

def ciclicos(request):
    global pvcs
    global pacient
    global ciclicos_pvcs
    if pacient is None:
        template = loader.get_template('damo/ciclicos_error.html')
    else:
        template = loader.get_template('damo/ciclicos.html')

    context = {'result' : ciclicos_pvcs}
    return HttpResponse(template.render(context,request))


def relatorio(request):
    global ecg, pvc, ind
    global pacient
    global beats
    global pvcs
    global f1_score
    global pvch
    global ritmo_medio
    global average_pvc_ecg
    global ciclicos_pvcs
    if pacient is None:
        template = loader.get_template('damo/relatorio_error.html')
    else:
        template = loader.get_template('damo/relatorio.html')

    result1 = beats
    result2 = pvcs
    result3 = pvch
    result4 = float(f1_score) *100
    result5 = round(average_pvc_ecg,4)
    result6 = round(ritmo_medio,4)
    result7 = ciclicos_pvcs
    context = {'result1': result1, 'result2': result2, 'result3':result3, 
               'result4':result4,'result5':result5,'result6':result6,'result7':result7}

    return HttpResponse(template.render(context,request))


def predict(data):
    global ind, pvc, ecg
    global beats
    global pvcs
    global f1_score
    global pvch
    global ritmo_medio
    global ciclicos_pvcs    
    # PRE PROC
    df = pd.DataFrame(columns=["ECG","IND","PVC"])
    load = loadmat(data)
    dados = load['DAT'][0]
    df_dados = pd.DataFrame(dados)
    ecg_array = np.array(df_dados['ecg'][0])
    ind_array = np.array(df_dados['ind'][0])
    pvc_array = np.array(df_dados['pvc'][0])

    ecg_array = np.array([x[0] for x in ecg_array])
    ind_array = np.array([x[0] for x in ind_array])
    pvc_array = np.array([x[0] for x in pvc_array])

    ecg, ind, pvc = preprocess_ecg(ecg_array, ind_array, pvc_array)

    for i, p in enumerate(pvc):
        if p != 0 | p != 1:
            pvc = np.delete(pvc,i)
            ind = np.delete(ind,i)
    
    beats = len(ind)
    pvcs = np.sum(pvc == 1)
    duration_minutes = 30
    duration_hours = duration_minutes / 60
    pvch = pvcs / duration_hours

    #BPM
    duration_minutes = 30
    duration_seconds = duration_minutes * 60
    ritmo_medio = (beats / duration_seconds) * 60
    
    # Ciclicos

    for i in range(len(ind)-1):
    #Check if PVCs are detected in two consecutive beats
        if ind[i+1] - ind[i] == 1:
            ciclicos_pvcs = True
            ciclicos_pvcs = "Tem"
        else:
            ciclicos_pvcs = False
            ciclicos_pvcs = "Não tem"


    coeffs = pywt.wavedec(ecg, 'db4', level=6)
    cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = np.std(cD1)
    peaks, _ = sig.find_peaks(cD1, distance=200, height=threshold)

    # Etapa 4: Separar as features das etiquetas e dividir o conjunto de dados em conjuntos de treinamento e teste
    X = np.array(peaks).reshape(-1, 1)
    y = np.array(pvc)

    y = y[:len(X)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)

    y_pred = model.predict(X_test)
    # Etapa 7: Avaliar a precisão do modelo no conjunto de teste

    print('RFC:')
    report = classification_report(y_test, y_pred, zero_division=1)
    report_lines = report.split('\n')
    f1_score = report_lines[2].split()[3]
    print("F1-score: ", f1_score)
    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do modelo RFC: {accuracy:.2f}")

    print('\n')

    clf_pred = clf.predict(X_test)
    print('SVM:')
    report = classification_report(y_test, y_pred, zero_division=1)
    report_lines = report.split('\n')
    f1_score = report_lines[2].split()[3]
    print("F1-score: ", f1_score)
    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do modelo SVM: {accuracy:.2f}")

    detected()

def plt_ecg_medio():
    
    global ind,pvc,ecg

    ind_array_clean = deepcopy(ind)
    pvc_array_clean = deepcopy(pvc)

    # Set a fixed length for the ECG segments
    segment_length = 41

    # Extract ECG segments of normal beats
    normal_segments = []
    for i, peak in enumerate(ind_array_clean):
        if pvc_array_clean[i] == 0:  # Filter normal beats
            segment = ecg[peak-20:peak+21]  # Adjust the window size around the peak
            normal_segments.append(segment)

    # Pad the segments to a fixed length
    padded_segments = np.zeros((len(normal_segments), segment_length))
    for i, segment in enumerate(normal_segments):
        padded_segments[i, :len(segment)] = segment

    # Calculate the average ECG waveform
    average_ecg = np.mean(padded_segments, axis=0)

    # Plot the average ECG waveform
    plt.figure(figsize=(8, 6))
    plt.plot(average_ecg)
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.title('ECG Médio')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Codifica o buffer em base64
    fig_path = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return fig_path

def plot_ecg_with_fs(fs):

    global ind,pvc,ecg

    # Set a fixed length for the ECG segments
    segment_length = 41

    # Define the sample frequency (in Hz)
    sample_frequency = fs

    # Calculate the time axis values
    time_axis = np.arange(segment_length) / sample_frequency

    # Extract ECG segments of normal beats
    normal_segments = []
    for i, peak in enumerate(ind):
        if pvc[i] == 0:  # Filter normal beats
            segment = ecg[peak-20:peak+21]  # Adjust the window size around the peak
            normal_segments.append(segment)

    # Pad the segments to a fixed length
    padded_segments = np.zeros((len(normal_segments), segment_length))
    for i, segment in enumerate(normal_segments):
        padded_segments[i, :len(segment)] = segment

    # Calculate the average ECG waveform
    average_ecg = np.mean(padded_segments, axis=0)

    # Plot the average ECG waveform
    plt.figure(figsize=(8, 6))
    plt.plot(time_axis, average_ecg)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title('ECG Médio com Frequência')

    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)

    # Codifica o buffer em base64
    image1_png = buffer1.getvalue()
    buffer1.close()
    fig_path_fs = base64.b64encode(image1_png).decode('utf-8')

    return fig_path_fs

def detected():
    global ind,pvc,ecg
    global average_pvc_ecg
    window_size = 5  # Define the window size

    detected_pvcs = []  # List to store the indices of detected PVCs

    # Iterate through the beats using a sliding window
    for i in range(len(ind) - window_size + 1):
        current_window = ind[i:i+window_size]  # Get the beats in the current window
        current_beat = ecg[current_window[window_size // 2]]  # Get the current beat
        
        # Get the neighboring beats
        previous_beat = ecg[current_window[window_size // 2 - 1]]
        next_beat = ecg[current_window[window_size // 2 + 1]]
        
        # Calculate the mean squared error (MSE) between the current beat and its neighbors
        mse_previous = np.mean((current_beat - previous_beat) ** 2)
        mse_next = np.mean((current_beat - next_beat) ** 2)
        
        # Check if the MSE exceeds the threshold for PVC detection
        threshold = 0.1  # Adjust the threshold based on your specific application
        if mse_previous > threshold and mse_next > threshold:
            detected_pvcs.append(current_window[window_size // 2])

    average_pvc_ecg = np.mean(ecg[detected_pvcs], axis=0)