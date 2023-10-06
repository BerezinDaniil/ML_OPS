import joblib
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    f = open("dataset.csv")
    f.close()
except FileNotFoundError:
    file_id = "1HJTDfP6njwgsToIrSSPBNNRXvh9QmX_R"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    while r.status_code != 200:
        r = requests.get(url)
    with open("dataset.csv", "wb") as f:
        f.write(r.content)

df = pd.read_csv("dataset.csv")
target = "Dx:Cancer"
df = df.drop_duplicates()

for name in df.columns:
    mean = np.array(df[f"{name}"] != "?").mean()
    df[name] = df[name].replace({"?": mean})

X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y, shuffle=True
)
sm = SMOTE()
x_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

scaler = MinMaxScaler()
scaler.fit(X_train)
scaler_filename = "scaler"
joblib.dump(scaler, scaler_filename)

X_train_scal = scaler.transform(X_train)

TRAIN_WITH_SMOTE = True


def fit_model(train_pool, **kwargs):
    model = CatBoostClassifier(iterations=1000)
    return model.fit(train_pool, silent=True)


if TRAIN_WITH_SMOTE:
    train_pool = Pool(x_train_sm, y_train_sm)
else:
    train_pool = Pool(X_train, y_train)

model = fit_model(train_pool)
model.save_model("model")