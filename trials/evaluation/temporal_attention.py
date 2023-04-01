import json

import arrow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

b = pd.read_csv("rolling.csv")
b = b[b.columns[0]].values[2:]
b = [arrow.get(val).datetime for val in b]
print(b)
a = open("temp_atte.json")
a = json.load(a)
a = np.array(a["data"][0]["z"])
mask = np.zeros_like(a)
mask[np.triu_indices_from(mask)] = True
names = [
    "AAPL",
    "ABT",
    "ADSK",
    "ALB",
    "AMGN",
    "APD",
    "BLK",
    "CAT",
    "CDNS",
    "CLX",
    "COF",
    "DE",
    "DHI",
    "EMR",
    "EQR",
    "FE",
    "FMC",
    "GIS",
    "IP",
    "IPG",
    "JPM",
    "MS",
    "NEE",
    "NEM",
    "NTAP",
    "NWL",
    "ROP",
    "ROST",
    "TJX",
    "VLO",
]
sns.heatmap(a, xticklabels=False, yticklabels=names)
plt.savefig("temp_atte.png")
# plt.show()
