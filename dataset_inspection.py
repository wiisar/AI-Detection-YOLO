import pandas as pd

url = (
    "hf://datasets/jrahn/yolochess_deepblue/"
    "data/train-00000-of-00001-a3d480609ce6a124.parquet"
)

df = pd.read_parquet(url)

print("Dimensions :", df.shape)
print("Colonnes :", df.columns.tolist())
print("\nTypes :")
print(df.dtypes)

print("\nPremières lignes :")
print(df.head())

print("\nInformations détaillées :")
print(df.info())

print("\nValeurs manquantes :")
print(df.isna().sum())

print("\nExemple complet :")
print(df.iloc[0].to_dict())