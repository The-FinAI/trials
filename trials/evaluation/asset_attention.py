import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

a = open("asset_atte.json")
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
sns.heatmap(a, xticklabels=names, yticklabels=names)
plt.savefig("asset_atte.png")
plt.show()
