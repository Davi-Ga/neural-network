import pandas as pd
import numpy as np
from neural import test_and_train_separation,neural_train,prediction
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('data/Cancer_Data.csv')
data['diagnosis'].replace(['M', 'B'], [1, 0], inplace=True)
#Removendo colunas desnecessárias
data.drop(columns=['Unnamed: 32','id'],inplace=True)

#Descrevendo os dados
characteristics=data.drop(columns=['diagnosis'])
proactive=data['diagnosis']

#Separação e Treino
x_train,x_test,y_train,y_test=test_and_train_separation(characteristics,proactive)


print('Acurácia: ',neural_train(characteristics,proactive))

#
