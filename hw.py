import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.colors as mcolors
bwght_data = pd.read_csv("C:/Users/AUSER/Documents/data/BWGHT.raw",skiprows=0, header=None, engine="python", sep="\s{2,}", names=["faminc", "cigtax", "cigprice", "bwght", "fatheduc", "motheduc", "parity", "male", "white", "cigs", "lbwght", "bwghtlbs","packs","lfaminc"])
cigs = bwght_data.iloc[:, 9]
lbs = bwght_data.iloc[:, 11]

#task 1
cigs_lbs = bwght_data[["cigs","bwghtlbs"]].copy()
cigs_lbs.columns = ["cigs", "lbs"]
model = smf.ols(formula="cigs ~ lbs", data=cigs_lbs)
results =model.fit()
print(results.summary())
beta=results.params
r2 = results.rsquared 
lbs_hat=results.fittedvalues
#task 3-1, MSE1
lbs = cigs_lbs["lbs"]
cigs = cigs_lbs["cigs"]
cigs_train ,cigs_test ,lbs_train ,lbs_test=train_test_split (cigs,lbs,test_size=.4,random_state=9)
X_train = sm.add_constant(cigs_train)
model = sm.OLS(lbs_train ,X_train)
results =model.fit()
beta_train =results.params
u_hat_test = lbs_test - beta_train[0] - beta_train[1]*cigs_test
MSE =np.mean (u_hat_test**2)
print("MSE1: ",MSE) #MSE1:  1.555206690257327

#task two
cigssq =np.square(cigs_lbs["cigs"])
cigs_lbs =cigs_lbs.assign(cigssq=cigssq)
model = smf.ols(formula="lbs~cigs+cigssq", data=cigs_lbs)
results =model.fit()
beta =results.params
lbs_hat=results.fittedvalues
#task 3-2, MSE2
r2 = results.rsquared
cigs_train ,cigs_test ,cigssq_train ,cigssq_test ,lbs_train ,lbs_test=train_test_split (cigs,cigssq,lbs,test_size =0.4,random_state=9)
X_train = pd.concat([cigs_train ,cigssq_train],axis=1)
X_train = sm.add_constant(X_train)
model = sm.OLS(lbs_train ,X_train)
results =model.fit()
print(results.summary())
beta_train = results.params
u_hat_test = lbs_test - beta_train[0] - beta_train[1]*cigs_test -beta_train[2]*cigssq_test
MSE =np.mean (u_hat_test**2)
print("MSE2: ",MSE) #MSE2:  1.5497617795426448

#task four
#bwghtlbs=β0+cigsβ1+cigs2β2+u 這個model is better，因為他的MSE比較小
print("conclusion: bwghtlbs=β0+cigsβ1+cigs2β2+u this regression model is better.\n")