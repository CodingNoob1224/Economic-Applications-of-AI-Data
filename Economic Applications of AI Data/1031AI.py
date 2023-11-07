import pandas as pd
wagedata = pd.read_csv("C:/Users/AUSER/Documents/data/WAGE1.raw",skiprows=0, header=None, engine="python", nrows=526, sep="\s{2,}")
print (wagedata)

import statsmodels.api as sm
import statsmodels.formula.api as smf
wage_edu = wagedata[[0,1]]. copy()
wage_edu.columns = ["wage", "edu"]


model = smf.ols(formula="wage ~ edu", data=wage_edu)
results =model.fit()
# print(results.summary())

import numpy as np
lwage = np.log(wage_edu["wage"])
wage_edu = wage_edu.assign(lwage=lwage)
model = smf.ols(formula="lwage~edu",data=wage_edu)
results = model.fit()
# print (results.summary())
beta = results.params
lwage_hat = results.fittedvalues
r2 = results.rsquared

import matplotlib.pyplot as plt
plt.plot(wage_edu["edu"],wage_edu["lwage"],marker="o",linestyle=" ")
plt.plot(wage_edu["edu"],lwage_hat ,linestyle="-")
plt.ylabel("lwage")
plt.xlabel("edu")
# plt.show()

from sklearn.model_selection import train_test_split
wage =wage_edu["wage"]
edu = wage_edu["edu"]
edu_train , edu_test , lwage_train ,lwage_test = train_test_split (edu,wage,test_size =0.4, random_state=9)
x_train = sm.add_constant(edu_train)
print (x_train)
model = smf.ols(lwage_train, x_train)
results = model.fit()
beta_train =results.params
lwage_hat_test = beta_train[0] + beta_train[1] * edu_test
lwage_hat_test_exp=np.exp(lwage_hat_test)
wage_test=np.exp(lwage_test)
u_exp_hat_test = wage_test - lwage_hat_test_exp
MSE_exp = np.mean(u_exp_hat_test**2)
print(MSE_exp)