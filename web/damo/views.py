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
from .data import model, clf, preprocess_ecg, nb

# https://youtube.com/shorts/X7i9VNKrHl0?feature=share

ind = None
ecg = None
pvcc = None
myfile = None
pacient = None
first_name = None
last_name = None
beats = 0
fs = 0
pvcs = None
user = None
pvch = None
fig_path = ''
ritmo_medio = 0
f1_score = 0
average_pvc_ecg = 0
ciclicos_pvcs = 0
ciclos = 0
totalCiclos = 0

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

def escolha_fs(request):
    global fs
    if request.method == 'POST':
        fs = request.POST['fs']
        if fs != '' and fs != '0':
            fs = int(request.POST['fs'])
            return redirect(new_home)
        elif fs == '0':
            return render(request, 'damo/ecg_error.html')

    fs = None
    return render(request, 'damo/escolha_fs.html')

def new_home(request):
    global pacient
    global fs
    global myfile
    if myfile == None or myfile == 0:
        return redirect(paciente)
    predict(myfile, fs)
    template = loader.get_template('damo/new_home.html')
    context = {'fs': fs}
    return HttpResponse(template.render(context, request))

def paciente(request):
    global myfile
    global pacient
    pacient = None
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        load = loadmat(myfile)
        myfile = load
        pacient = fs.url(filename)

        allowed_extensions = {'mat'}
        if allowed_file(pacient, allowed_extensions):
            if load == {}:
                pacient = None
                return render(request, 'damo/p2.html')
            else:
                return redirect(escolha_fs)
        else:
            pacient = None
            return render(request, 'damo/p_1.html')
    return render(request, 'damo/pacientes.html')

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def ecg_medio(request):
    global pacient
    global fs
    if pacient is None:
        return render(request, 'damo/escolha_fs_error.html')
    else:
        if fs != None:
            return redirect(ecg_medio_fs)
        else:
            return redirect(ecg_medio_plot)

def beat(request):
    global beats
    global pacient
    if pacient is not None:
        template = loader.get_template('damo/beat.html')
    else:
        template = loader.get_template('damo/beat_error.html')

    context = {'result':beats}

    return HttpResponse(template.render(context,request))

def pvc(request):
    global pvcs
    global pacient
    if pacient is not None:
        template = loader.get_template('damo/pvc.html')
    else:
        template = loader.get_template('damo/pvc_error.html')


    context = {'result' : pvcs}
    return HttpResponse(template.render(context,request))


def pvcH(request):
    global pacient
    global pvch
    if pacient is not None:
        template = loader.get_template('damo/pvcsH.html')
    else:
        template = loader.get_template('damo/pvcsH_error.html')


    context = {'result' : pvch}

    return HttpResponse(template.render(context,request))


def f1(request):
    global pacient
    global f1_score
    if pacient is not None:
        template = loader.get_template('damo/f1.html')
    else:
        template = loader.get_template('damo/f1_error.html')

    context = {'result' : round(float(f1_score)*100,2)}

    return HttpResponse(template.render(context,request))

def error(request):
    return render(request, 'damo/error.html')
    

def ecg_medio_plot(request):
    global pacient
    global fig_path
    global ind, pvcc, ecg
    fig_path = plt_ecg_medio(ind,pvcc,ecg)
    if fig_path is None:
        return render(request,'damo/error.html')
    else: 
        return render(request, 'damo/ecg-medio.html', {'result': fig_path})

def ecg_medio_fs(request):
    global pacient
    global fig_path
    global fs
    global ind, pvcc, ecg
    fig_path = plot_ecg_with_fs(ind,pvcc,ecg,fs)
    if fig_path is None:
        return render(request,'damo/error.html')
    else: 
        return render(request, 'damo/ecg-medio.html', {'result': fig_path})

def ritmo(request):
    global pacient
    global ritmo_medio
    if pacient != 0:
        template = loader.get_template('damo/ritmo.html')
    else:
        template = loader.get_template('damo/ritmo_error.html')

    context = {'result' : round(ritmo_medio,2)}

    return HttpResponse(template.render(context,request))


def meanpvcs(request):
    global pvcs
    global pacient
    global average_pvc_ecg
    if pacient is not None:
        template = loader.get_template('damo/meanpvcs.html')
    else:
        template = loader.get_template('damo/meanpvcs_error.html')


    context = {'result' : round(average_pvc_ecg,2)}
    return HttpResponse(template.render(context,request))

def ciclicos(request):
    global pvcs
    global pacient
    global ciclicos_pvcs
    global ciclos
    global totalCiclos
    if pacient is not None:
        template = loader.get_template('damo/ciclicos.html')
    else:
        template = loader.get_template('damo/ciclicos_error.html')
    if ciclos != 0:
        context = {'result' : ciclicos_pvcs, 'ciclos':ciclos, 'total':totalCiclos}
    else:
        context = {'result': ciclicos_pvcs}

    return HttpResponse(template.render(context,request))


def relatorio(request):
    global ecg, pvcc, ind
    global pacient
    global beats
    global pvcs
    global f1_score
    global pvch
    global ritmo_medio
    global average_pvc_ecg
    global ciclicos_pvcs
    if pacient is not None:
        template = loader.get_template('damo/relatorio.html')
    else:
        template = loader.get_template('damo/relatorio_error.html')


    result1 = beats
    result2 = pvcs
    result3 = pvch
    result4 = round(float(f1_score) *100,2)
    result5 = round(average_pvc_ecg,2)
    result6 = round(ritmo_medio,2)
    result7 = ciclicos_pvcs
    context = {'result1': result1, 'result2': result2, 'result3':result3,
               'result4':result4,'result5':result5,'result6':result6,'result7':result7}

    return HttpResponse(template.render(context,request))


def predict(data, fs):
    global ind, pvcc, ecg
    global beats
    global pvcs
    global f1_score
    global pvch
    global ritmo_medio
    global ciclicos_pvcs
    global ciclos
    global totalCiclos
    # PRE PROC
    # df = pd.DataFrame(columns=["ECG","IND","PVC"])
    dados = data['DAT'][0]
    df_dados = pd.DataFrame(dados)
    ecg_array = np.array(df_dados['ecg'][0])
    ind_array = np.array(df_dados['ind'][0])
    pvc_array = np.array(df_dados['pvc'][0])

    ecg_array = np.array([x[0] for x in ecg_array])
    ind_array = np.array([x[0] for x in ind_array])
    pvc_array = np.array([x[0] for x in pvc_array])

    ecg, ind, pvcc = preprocess_ecg(ecg_array, ind_array, pvc_array, fs)
    pvcc = np.array(pvcc)
    ind = np.array(ind)

    for i, p in enumerate(pvcc):
        if p != 0 | p != 1:
            pvcc = np.delete(pvcc,i)
            ind = np.delete(ind,i)

    beats = len(ind)
    pvcs = np.count_nonzero(pvcc == 1)
    duration_minutes = 30
    duration_hours = duration_minutes / 60
    pvch = pvcs / duration_hours

    #BPM
    duration_minutes = 30
    duration_seconds = duration_minutes * 60
    ritmo_medio = (beats / duration_seconds) * 60

    # Ciclicos
    ciclos = 0
    threshold = 1
    ciclicos_pvcs = False
    ciclos = []
    totalCiclos = 0
    c = 0
    for i in range(len(ecg) - 2):
        # Verifica se o valor atual é menor que o valor anterior e também menor que o valor posterior
        if ecg[i] < ecg[i-1] and ecg[i] < ecg[i+1]:
            # Verifica se a diferença entre o valor anterior e o posterior é maior que o threshold
            if abs(ecg[i-1] - ecg[i+1]) > threshold:
                totalCiclos +=1
                c += 1
                ciclicos_pvcs = True
            else:
                ciclos.append(c)
                c = 0
                ciclicos_pvcs = False

    ciclos = max(ciclos)
    if ciclos > 0:
        ciclicos_pvcs = "Tem"
    else:
        ciclicos_pvcs = "Não tem"

    y = pvcc

    new_ecg = np.zeros(len(ind))

    for i in range(len(ind)):
        new_ecg[i] = ecg[ind[i]]

    X = new_ecg.reshape(-1,1)

    model_pred = model.predict(X)
    clf_pred = clf.predict(X)
    nb_pred = nb.predict(X)

    report = classification_report(y, model_pred, zero_division=1)
    report_lines = report.split('\n')
    f1_score_model = report_lines[3].split()[3]
    print("F1-score RFC:", f1_score_model)

    report = classification_report(y, clf_pred, zero_division=1)
    report_lines = report.split('\n')
    f1_score_clf = report_lines[3].split()[3]
    print("F1-score SVM:", f1_score_clf)

    report = classification_report(y, nb_pred, zero_division=1)
    report_lines = report.split('\n')
    f1_score_nb = report_lines[3].split()[3]
    print("F1-score Naive:", f1_score_nb)

    if f1_score_model > f1_score_clf:
        if f1_score_model > f1_score_nb:
            f1_score = f1_score_model
        else:
            f1_score = f1_score_nb
    else:
        f1_score = f1_score_clf

    detected()

def plt_ecg_medio(ind,pvcc,ecg):

    ind_array_clean = deepcopy(ind)
    pvc_array_clean = deepcopy(pvcc)
    ecg_array_clean = deepcopy(ecg)

    average_ecg = aligned_ecg_average(ecg_array_clean, ind_array_clean, pvc_array_clean)
    
    if average_ecg is None:
        return average_ecg
    
    # Plota o ECG médio alinhado no pico R
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

def plot_ecg_with_fs(ind,pvcc,ecg,fs):

    ind_array_clean = deepcopy(ind)
    pvc_array_clean = deepcopy(pvcc)
    ecg_array_clean = deepcopy(ecg)

    average_ecg = aligned_ecg_average(ecg_array_clean, ind_array_clean, pvc_array_clean)

    if average_ecg is None:
        return average_ecg

    segment_length = 41

    sample_frequency = fs
    # Calculate the time axis values
    time_axis = np.arange(segment_length) / sample_frequency

    # Plota o ECG médio alinhado no pico R
    plt.figure(figsize=(8, 6))
    plt.plot(time_axis, average_ecg)
    plt.xlabel('Amostras')
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

def aligned_ecg_average(ecg, ind_array, pvc_array):
        segment_length = 41  # Comprimento fixo do segmento de ECG
        aligned_segments = []

        if ind_array is not None:
            for i, peak in enumerate(ind_array):
                if pvc_array[i] == 0:  # Filtra batimentos cardíacos normais
                    start = peak - int(segment_length / 2)
                    end = peak + int(segment_length / 2) + 1
                    segment = ecg[start:end]
                    aligned_segments.append(segment)
        else:
            return
        # Alinha os segmentos com base no pico R
        aligned_segments = np.array(aligned_segments)
        aligned_segments = np.transpose(aligned_segments)

        # Calcula a média ao longo do eixo temporal
        average_ecg = np.mean(aligned_segments, axis=1)

        return average_ecg

def detected():
    global ind,pvcc,ecg
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