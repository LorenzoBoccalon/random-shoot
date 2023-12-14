# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from common import DATASET_FOLDER, DATASET_CSV

print(os.listdir(DATASET_FOLDER))

if not os.path.exists(DATASET_CSV):
    df = pd.read_csv(DATASET_FOLDER + "train.csv")
    df = df.dropna(axis=0).reset_index(drop=True)
    uid = df["id"].unique()
    train_ids, test_ids = train_test_split(uid, test_size=0.2, random_state=42)
    df["split"] = ""
    df.loc[df["id"].isin(train_ids), "split"] = "train"
    df.loc[df["id"].isin(test_ids), "split"] = "test"
    df.to_csv(DATASET_CSV)
else:
    df = pd.read_csv(DATASET_CSV)
print(df.head())
