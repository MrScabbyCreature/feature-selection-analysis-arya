import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_df(df):
    df2 = df.copy()
    df2[df==0] = 1
    df2 = np.log(df2)
    return df2

# Load and process data
data_df = process_df(pd.read_csv("test_set.csv", index_col=0))


# Get valid split
X = StandardScaler().fit_transform(data_df)

# Get valid scores
model = joblib.load("model.joblib")
print("Test predictions:", model.predict(X))