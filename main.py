import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')

import warnings
warnings.filterwarnings('ignore')



data=pd.read_csv('data/Cancer_Data.csv')
data['diagnosis'].replace(['M', 'B'], [1, 0], inplace=True)

eixo=sns.countplot(data=data,x='diagnosis')
plt.title('Verificando Diagnóstico')
eixo.bar_label(eixo.containers[0],label_type='edge')

figura, eixo=plt.subplots(figsize=(15,8))

corr=data.corr()
