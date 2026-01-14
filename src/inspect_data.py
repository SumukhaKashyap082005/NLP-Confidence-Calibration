import pandas as pd


df = pd.read_csv("../data/train.csv")

print("Total rows:", len(df))
print("\nColumns:")
print(df.columns)

print("\nFirst 3 rows:")
print(df.head(3))

print("\nLabel value counts (toxic):")
print(df["toxic"].value_counts())
