import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def process_df(df):
    df2 = df.copy().drop(columns=["Y"])
    df2[df==0] = 1
    df2 = np.log(df2)
    df2["Y"] = df["Y"]
    return df2

# Load and process data
data_df = process_df(pd.read_csv("training_set.csv", index_col=0))


# Get valid split
X = StandardScaler().fit_transform(data_df.drop(columns=["Y"]))
y = data_df["Y"].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42) # Same random state as training gives the same train-valid split

# Get valid scores
model = joblib.load("model.joblib")
print("Validation accuracy:", model.score(X_valid, y_valid))
print("Features used:", model.n_features_)