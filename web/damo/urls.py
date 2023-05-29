from django.urls import path

# Create your views here.

from . import views

urlpatterns = [
    path('', views.welcome, name = 'welcome'),
    path("login", views.login, name='login'),
    path("register", views.register, name = "register"),
    path('home', views.home, name = 'home'),
    path('paciente', views.paciente, name = 'pacientes'),
    path('new_home', views.new_home, name = 'new_home'),
    path('beat', views.beat, name = 'beat'),
    path('pvc', views.pvc, name = 'pvc'),
    path('pvcsH', views.pvcH, name = 'pvcH'),
    path('f1', views.f1, name = 'f1'),
    path('escolha_fs',views.escolha_fs, name= 'escolha_fs'),
    path('ecg-medio', views.ecg_medio, name='ecg_medio'),
    path('ecg-plot', views.ecg_medio_plot, name='ecg_medio_plot'),
    path('ecg-fs', views.ecg_medio_fs, name='ecg_medio_fs'),
    path('relatorio', views.relatorio, name = 'relatorio'),
    path('logout', views.logout, name = 'logout'),
    path('media-pvcs', views.meanpvcs, name = 'media'),
    path('ciclicos', views.ciclicos, name = 'ciclicos'),
    path('ritmo',views.ritmo, name='ritmo')
]