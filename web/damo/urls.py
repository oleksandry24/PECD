from django.urls import path

# Create your views here.

from . import views

urlpatterns = [
    path('', views.welcome, name = 'welcome'),
    path("login", views.login, name='login'),
    path("register", views.register, name = "register"),
    path('home', views.home, name = 'home'),
    path('paciente', views.paciente, name = 'pacientes'),
    path('new_home', views.home, name = 'new_home'),
    path('beat', views.beat, name = 'beat'),
    path('pvc', views.pvc, name = 'pvc'),
    path('pvcsH', views.pvcH, name = 'pvcH'),
    path('f1', views.f1, name = 'f1'),
    path('ecg-medio', views.ecg, name = 'ecg'),
    path('relatorio', views.relatorio, name = 'relatorio'),
    path('logout', views.logout, name = 'logout'),
    path('new_home',views.new_home, name = 'new_home')
]