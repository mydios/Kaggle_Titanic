import sys
sys.path.insert(1, '../Kaggle_Titanic')
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

from preprocessing import preprocess_basic

df_train = pd.read_csv('./titanic/train.csv')
df_train = df_train[['Survived', 'Fare']]

x = [0.5*i for i in range(2*np.round(np.max(df_train['Fare'])).astype(np.int))]
y1 = [np.sum(df_train[df_train['Fare']<x[i]]['Survived']) for i in range(len(x))]
y2 = [len(df_train[df_train['Fare']<x[i]]['Survived']) for i in range(len(x))]

sb.lineplot(x, y1)
sb.lineplot(x, y2)
plt.legend(['#survived with this fare or less', '#passengers with this fare or less'])
#plt.xscale('log')
plt.xlim((200, 500))
plt.show()