import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('titanic/train.csv')
profile = ProfileReport(df, title="Pandas Profiling Report Titanic dataset")
profile.config.interactions.targets = ['Survived']
profile.to_file("exploratory_analysis.html")

