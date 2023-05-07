import scipy.signal as sig
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import pywt
from scipy.io import loadmat
import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.models import User
from django.contrib.auth import login as logg
from django.contrib.auth import logout
from django.contrib.auth import authenticate
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


from .data import model, clf, preprocess_ecg

# https://youtube.com/shorts/X7i9VNKrHl0?feature=share

pacient = None
first_name = None
last_name = None
beats = 0
user = None
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
            first_name = user.first_name
            last_name = user.last_name
            return render(request,'damo/home.html', {'first_name':first_name,'last_name':last_name})
        else:
            return render(request, 'damo/login_error.html')
    return render(request, 'damo/login.html')

def logout(request):
    template = loader.get_template('damo/logout.html')
    if request.method == 'POST':
        logout(request)
        return render(request, 'damo/welcome.html')
    return HttpResponse(template.render(request))

def home(request):
    global pacient
    template = loader.get_template('damo/home.html')
    return HttpResponse(template.render(request))

def new_home(request):
    global pacient
    template = loader.get_template('damo/new_home.html')
    return HttpResponse(template.render(request))

def paciente(request):
    global pacient
    pacient = None
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        print(myfile)
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        pacient = fs.url(filename)
        print(pacient)
        
        allowed_extensions = {'.mat'}
        #if allowed_file(pacient, allowed_extensions):
        #    return render(request, 'damo/new_home.html')
        #else:
        #    return render(request, 'damo/p_1.html')

        predict(myfile)
        return render(request, 'damo/new_home.html')

    return render(request, 'damo/pacientes.html')

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

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

    context = {'result' : 0}
    return HttpResponse(template.render(context,request))

def pvcH(request):
    global pacient
    global pvc
    if pacient is None:
        template = loader.get_template('damo/pvcsH_error.html')
    else:
        template = loader.get_template('damo/pvcsH.html')

    context = {'result' : 0}

    return HttpResponse(template.render(context,request))

def f1(request):
    global pacient
    if pacient is None:
        template = loader.get_template('damo/f1_error.html')
    else:
        template = loader.get_template('damo/f1.html')

    f1 = 0
    context = {'result' : f1}

    return HttpResponse(template.render(context,request))

def ecg(request):
    global pacient
    if pacient is None:
        template = loader.get_template('damo/ecg_error.html')
    else:
        template = loader.get_template('damo/ecg-medio.html')

    ecg = 0
    context = {'result' : ecg}

    return HttpResponse(template.render(context,request))

def relatorio(request):
    global ecg, pvc, ind
    global pacient
    if pacient is None:
        template = loader.get_template('damo/relatorio_error.html')
    else:
        template = loader.get_template('damo/relatorio.html')

    result1 = 0
    result2 = 0
    result3 = 0
    result4 = None
    result5 = None
    context = {'result1': result1, 'result2': result2, 'result3':result3, 
               'result4':result4,'result5':result5}

    return HttpResponse(template.render(context,request))


def predict(data):
    global beats
    beats = 0
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
        
    sample_rate = 360
    for i, p in enumerate(pvc):
        if p != 0 | p != 1:
            pvc = np.delete(pvc,i)
            ind = np.delete(ind,i)
    
    beats = len(ind)

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
    print(classification_report(y_test, y_pred, zero_division=1))
    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do modelo RFC: {accuracy:.2f}")

    print('\n')

    clf_pred = clf.predict(X_test)
    print('SVM:')
    print(classification_report(y_test, clf_pred, zero_division=1))
    accuracy = model.score(X_test, y_test)
    print(f"Acurácia do modelo SVM: {accuracy:.2f}")




