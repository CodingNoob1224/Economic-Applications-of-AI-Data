import pandas as pd
bwght_data = pd.read_csv("C:/Users/AUSER/Documents/data/BWGHT.raw",skiprows=0, header=None, engine="python", sep="\s{2,}", names=["faminc", "cigtax", "cigprice", "bwght", "fatheduc", "motheduc", "parity", "male", "white", "cigs", "lbwght", "bwghtlbs", "packs", "lfaminc"])
# bwght_data = pd.read_csv("C:/Users/AUSER/Documents/data/BWGHT.raw", names=["faminc", "cigtax", "cigprice", "bwght", "fatheduc", "motheduc", "parity", "male", "white", "cigs", "lbwght", "bwghtlbs", "packs", "lfaminc"])
print (bwght_data)
# bwgtlbs_cigs = bwght_data[[11,9]].copy()
# bwgtlbs_cigs.columns = ["bwgtlbs", "cigs"]
# print (bwgtlbs_cigs)