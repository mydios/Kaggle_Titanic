import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('titanic/train.csv')
profile = ProfileReport(df, title="Pandas Profiling Report Titanic dataset train set")
profile.config.interactions.targets = ['Survived']
profile.to_file("exploratory_analysis_train.html")

df = pd.read_csv('titanic/test.csv')
profile = ProfileReport(df, title="Pandas Profiling Report Titanic dataset test set")
profile.config.interactions.targets = ['Survived']
profile.to_file("exploratory_analysis_test.html")

