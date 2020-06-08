# -*- coding: utf-8 -*-
"""
Created on Sat May 30 08:45:29 2020

@author: 
"""
#Se importan las librerias necesarias
import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
import pywt
import linearFIR
import os
import pandas as pd
#design

#Se define función para procesar los datos dados en el archivo de las señales
#como parámetros se ingresa el canal que se quiere visualizar y el nombre del archivo entre comillas
def filtro_frecuencias_file(x):  
    data, fs = sf.read(x+".wav") #Se lee el archivo de texto
    
    tiempo1=np.arange(0,len(data)/fs,1/fs)#Se define frecuencia de muestreo #se crea el vector de tiempo con la frecuencia de muestreo
    order, lowpass = linearFIR.filter_design(fs, locutoff = 0, hicutoff = 2000, revfilt = 0); # Se configura el pasa bajas
    order, highpass = linearFIR.filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1); # Se configura el pasa-altas
    data_pasa_bajas= signal.filtfilt(lowpass, 1, data); #Aplica filtro pasa bajas
    data_pasa_altas = signal.filtfilt(highpass, 1, data_pasa_bajas); #aplica filtro pasa-Altas
    
    
    Senal_Filtrada=data_pasa_altas 
    
    return Senal_Filtrada, tiempo1, fs #retorna señal filtrada y frecuencia de muestreo

def filtro_frecuencias(x, fs):  
   
    tiempo1=np.arange(0,len(x)/fs,1/fs)#Se define frecuencia de muestreo #se crea el vector de tiempo con la frecuencia de muestreo
    order, lowpass = linearFIR.filter_design(fs, locutoff = 0, hicutoff = 2000, revfilt = 0);
    order, highpass = linearFIR.filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);
    data_pasa_bajas= signal.filtfilt(lowpass, 1, x);
    data_pasa_altas = signal.filtfilt(highpass, 1, data_pasa_bajas);
    
    Senal_Filtrada=data_pasa_altas
    
    return Senal_Filtrada, tiempo1

def filtro_wavelet_file(x):
    
    data, fs = sf.read(x+".wav") #Se lee el archivo de audio y se extrae el vector de muestras junto con la frecuencia de muestreo
    tiempo=np.arange(0,len(data)/fs,1/fs) # Se calcula el tiempo de la señal
    LL = int(np.floor(np.log2(data.shape[0])))#
    
    coeff=pywt.wavedec(data, 'db6', level=LL)
    Num_samples = 0;
    for i in range(0,len(coeff)):
        Num_samples = Num_samples + coeff[i].shape[0];
    thr = np.sqrt(2*(np.log(Num_samples)))
    
    stdc = np.zeros((len(coeff),1));
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
    
    y   = list();
    s = stdc;
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
    
    x_rec = pywt.waverec(y, 'db6') # Se halla la señal filtrada con wavelet
    x_rec = x_rec[0:data.shape[0]] 
    x_filt =np.squeeze(data - x_rec);#Se adquiere la señal original umbralizada
    
    return x_filt, tiempo, fs

def filtro_wavelet(x,fs):
    
    data=x
    
    tiempo=np.arange(0,len(data)/fs,1/fs)
    LL = int(np.floor(np.log2(data.shape[0])))
    
    coeff=pywt.wavedec(data, 'db6', level=LL)
    Num_samples = 0;
    for i in range(0,len(coeff)):
        Num_samples = Num_samples + coeff[i].shape[0];
    thr = np.sqrt(2*(np.log(Num_samples)))
    
    stdc = np.zeros((len(coeff),1));
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
    
    y   = list();
    s = stdc;
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
    
    x_rec = pywt.waverec(y, 'db6')
    x_rec = x_rec[0:data.shape[0]]
    x_filt =np.squeeze(data - x_rec);
    
    return x_filt, tiempo

def Pre_procesamiento_file(x):
    data, fs = sf.read(x+".wav")
    x,tiempo=filtro_frecuencias(data, fs)
    x,tiempo=filtro_wavelet(x,fs)
    
    return x, tiempo, fs# Se retornal la señal filtrada, el tiempo de la señal y la frecuencia de muestreo

def Pre_procesamiento(x,fs):
    
     x,tiempo=filtro_frecuencias(x, fs)
     x,tiempo=filtro_wavelet(x,fs)
     
     return x, tiempo
  
    
def Ciclos_respiratorios(x):
   
    Inicio_ciclo=[] #Se crea un arreglo vacio para los tiempos de inicio de los ciclos
    Final_ciclo=[] #Se crea un arreglo vacio para los tiempos de finalización de los ciclos
    Estertores=[] # Se crea un arreglo vacio para guardar los ciclos con o sin precencia de estertores
    Sibilancias=[]# Se crea un arreglo vacio para guardar los ciclos con o sin precencia de Sibilancias
    if os.path.isdir('Ciclos_' + x): #Se condiciona la creación de la carpeta para guardar los ciclos
        print("el archivo existe") #Si ya esta creada imprime el mensaje
    else:    
        os.mkdir('Ciclos_' + x) #De lo contrario la crea
    signal, fs = sf.read(x+".wav") #Se extra el vector de la señal y la frecuencia de muestreo
    txt = open(x +".txt", 'r') #Se lee el archivo de texto
    tiempo=np.arange(0,len(signal)/fs,1/fs) #Se calcula el tiempo de la señal
    fila=txt.readlines() #Se leen las filas del archivo txt
    for i in np.arange(0,len(fila)): #Se crea una rutina para extraer las columnas del archivo de texto
        dato1=fila[i].find('\t')
        dato2=fila[i].find('\t', dato1+1)
        dato3=fila[i].find('\t', dato2+1)
        dato4=fila[i].find('\t', dato3+1)
        
        valor1=float(fila[i][0:dato1])
        valor2=float(fila[i][dato1+1:dato2])
        valor3=(fila[i][dato2+1:dato3])
        valor4=(fila[i][dato3+1:dato4])
        
        Inicio_ciclo.append(valor1)  #Se guarda el tiempo de inicio del ciclo
        Final_ciclo.append(valor2) #Se guarda el tiempo de finalización del ciclo
        Estertores.append(valor3) #Se guardan los valores para la precencia o ausencia de estertores
        Sibilancias.append(valor4) #Se guardan los valores para la presencia o ausencia de sibilancias
    
    Inicio_resp=np.zeros(len(Inicio_ciclo))# Se crea un vector con la longitud de la lista Inicio_ciclo
    Final_resp=np.zeros(len(Inicio_ciclo))# Se crea un vecrtor con la longitud de la lista Final_ciclo
    Ciclos=[] #Se crea una lista donde se guardaran los vectores correspondientes a los ciclos
    Vectores_tiempos=[] #Se crea una lista donde se guardaran los vectores correspondeintes a los tiempos de cada ciclo
    
    for i in np.arange(len(Inicio_ciclo)): #Se crea una rutina para extraer la señal de cada ciclo
        Inicio_resp[i]=tiempo.flat[np.abs(tiempo - Inicio_ciclo[i]).argmin()] #se compara los tiempos de inicio del archivo de texto con los del vector de tiempo para la señal original
        Final_resp[i]=tiempo.flat[np.abs(tiempo - Final_ciclo[i]).argmin()]#Se compara los tiempos del final del ciclo del archivo de texto con los del vector de tiempo de la señal orignal
        
        inicio=int(np.where(tiempo==Inicio_resp[i])[0])# Se halla la posicion donde se inicia el ciclo
        final=int(np.where(tiempo==Final_resp[i])[0])# Se halla la pocicion del vector de tiempo donde finaliza el ciclo
        
        Ciclo=signal[inicio:final] #Se construye el ciclo
        tiempo_C=tiempo[inicio:final] #Se construye el vector de tiempo correspondiente
        Ciclo,tiempo3= Pre_procesamiento(Ciclo,fs) #Se filtra el ciclo
        
        Ciclos.append(Ciclo) #Se agrega el ciclo filtrado a la lista creada
        Vectores_tiempos.append(tiempo_C) # Se agrega el tiempo a la lista creada
        
        plt.plot(tiempo_C,Ciclo) #Se grafica el ciclo
        
        if(Estertores[i]=="1"): #Se condiciona para mostrar si hay o no precencia de Crepitaciones
            legen_estertores='Presencia de Crepitacion'
        else:
            legen_estertores='Ausencia de Crepitacion'

        
        if(Sibilancias[i]=="1"):#Se condiciona para mostrar si hay o no precencia de Sibilancias
            legen_Sibilancia='Presencia de Sibilancia'
        else:
            legen_Sibilancia='Aucencia de Sibilancia'
        plt.suptitle(legen_Sibilancia+'   -   '+legen_estertores)# Se muestra la condicion del ciclo
           #shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=9   
        plt.title('Ciclo '+str(i+1))# Se crea el título de la gráfica dependiendo del ciclo
        plt.xlabel('Tiempo[s]')
        plt.ylabel('Amplitud')
        plt.grid()
        plt.savefig('Ciclos_'+x+'/''ciclo'+str(i)+'.png')#Se guarda la gráfica en la carpeta creada anteriormente
        plt.show()
        
       
    return  Ciclos, Vectores_tiempos, Estertores, Sibilancias, fs #retorna las variables de interés
    
def Rango(x): #Halla el rango de cualquier vector
    R=abs(np.max(x)-np.min(x))
    
    return R

def SMA(x): #Halla la media movil simple de cualquier vector
    
    SMA=np.zeros(len(x))
    
    for i in np.arange(len(x)):
        if(i+1<len(x)):
            MA=abs(x[i]-x[i+1])
            SMA[i]=MA
    SMA=np.sum(SMA)
    return SMA

def Spect_Mean(x,fs):#Halla el promedio del espectro para cualquier vector dada la frecuencia de muestreo
    
    f,pot=signal.welch(x,fs,'hanning', fs*2, fs)
    Mean=np.mean(pot)
    
    return Mean

def Ciclos_indices(x, fs): # Calcula los indices anteriores automáticamente
    
    R=Rango(x)
    V=np.var(x)
    SM=SMA(x)
    SMean=Spect_Mean(x,fs)
    
    return R, V, SM, SMean # Retorna los indices de interés
      

def Data_lung(Ciclos, Vectores_tiempos, Estertores, Sibilancias, fs):
    data_lung=pd.DataFrame({"Ciclos":Ciclos,"Tiempos":Vectores_tiempos,"Estado":np.zeros(len(Ciclos)),"Varianza":np.zeros(len(Ciclos)),"Rango":np.zeros(len(Ciclos)),"SMA":np.zeros(len(Ciclos)),"SMean":np.zeros(len(Ciclos))})#Se crea un dataframe para guardar los datos de cada ciclo
    for i in np.arange(len(Ciclos)): #Rutina para identificar el estado del ciclo y calcular los respectivos indices
        if((Estertores[i]=="0") and (Sibilancias[i]=="0" or Sibilancias[i]=="")):
            data_lung.Estado[i]="Ciclo Normal"
        if((Estertores[i]=="1") and (Sibilancias[i]=="0" or Sibilancias[i]=="")):
            data_lung.Estado[i]="Ciclo con Crepitacion"
        if((Estertores[i]=="0") and (Sibilancias[i]=="1")):
            data_lung.Estado[i]="Ciclo con Sibilancia"
        data_lung.Rango[i]=Rango(Ciclos[i]) #Se agrega el rango calculado a la columna rango
        data_lung.Varianza[i]=np.var(Ciclos[i])#Se agrega la varianza calculada a la columna varianza
        data_lung.SMA[i]=SMA(Ciclos[i])#Se agrega la media movil simple a la columna sma
        data_lung.SMean[i]=Spect_Mean(Ciclos[i],fs/len(Ciclos))#Se agrega el promedio del espectro a la columna SMean
    return data_lung
