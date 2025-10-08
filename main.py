import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

from models import LassoModel
from preprocessor import Preprocessor


# warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv('files//train.csv')
macro = pd.read_csv('files//macro.csv')

df = df.merge(macro, on='timestamp', how='left')


preprocessor = Preprocessor(df)
df_new = preprocessor.run()

Lasso = LassoModel(df_new)
rmse = Lasso.run()
print(rmse)




