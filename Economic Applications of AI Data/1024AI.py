import pandas as pd
# datal = pd.read_csv("sales.csv", delimiter=",", header=None, names =["year","productl","product2","product3"])
# sales = pd.read_table("C:/Users/AUSER/Documents/data/sales.txt", delimiter=" ")
# print(sales)
import matplotlib.pyplot as plt
# plt.plot(sales["year"], sales["product2"],color="blue")
# plt.plot(sales["year"], sales["product3"],color="red")
# plt.show()# show graph  onscreen70 / 117

import numpy as np
# dsales = sales["product3"]-sales["product2"]
# print(dsales)

# mean_dsals = np.mean(dsales)
# print(mean_dsals)

# n=len(dsales)
# print(n)

# sd_dsales = np.std(dsales, ddof=1)
# print(sd_dsales)

# se_dsales = sd_dsales / np.sqrt(n)
# print(se_dsales)

import scipy.stats as stats
# c = stats.t.ppf(0.975, n-1)
# print(c)

# CI_lower = mean_dsals - c * se_dsales
# CI_upper = mean_dsals + c * se_dsales

# print(f"({CI_lower}, {CI_upper})")

import statsmodels.api as sm
import statsmodels.formula.api as smf
wagedata = pd.read_csv("C:/Users/AUSER/Documents/data/WAGE1.raw",skiprows=0, header=None, engine="python", nrows=526, sep="\s{2,}")
print(wagedata)

wage_edu = wagedata[[0,1]].copy()
wage_edu.columns = ["wage","edu"]
print(wage_edu)
model = smf.ols(formula="wage~edu",data=wage_edu)
results = model.fit()
print(results.summary())
wage_hat = results.fittedvalues
print(wage_hat)
plt.plot(wage_edu["edu"], wage_edu["wage"],marker="o",linestyle=" ")
plt.plot(wage_edu["edu"],wage_hat ,linestyle="-")
plt.ylabel("wage")
plt.xlabel("edu")
plt.show()

from sklearn.model_selection import train_test_split
wage = wage_edu["wage"]
edu = wage_edu["edu"]
edu_train , edu_test , wage_train ,wage_test = train_test_split (edu,wage,test_size =0.4)
x_train = sm.add_constant(edu_train)
model = sm.OLS(wage_train, x_train)
results = model.fit()
print(results.summary())
# lwage = np.log(wage_edu["wage"])
# wage_edu = wage_edu.assign(lwage=lwage)
# model = smf.ols(formula="lwage~edu",data=wage_edu)
# results = model.fit()
# beta = results.params
# lwage_hat = results .fittedvalues
# r2 = results.rsquared 