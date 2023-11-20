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

'''
check data:
print (bwght_data)
print (cigs)
print(lbs)
'''

'''
by hand: 
cov_cigs_lbs = np.cov(cigs, lbs)[1,0]
var_cigs = np.var(cigs, ddof=1)
mean_cigs=np.mean(cigs)
mean_lbs=np.mean(lbs)
b1=cov_cigs_lbs/var_cigs
b0=mean_lbs - b1*mean_cigs
lbs_hat=b0+b1*lbs
u_hat =lbs-lbs_hat
'''
#task one
cigs_lbs = bwght_data[["cigs","bwghtlbs"]].copy()
cigs_lbs.columns = ["cigs", "lbs"]
model = smf.ols(formula="cigs ~ lbs", data=cigs_lbs)
results =model.fit()
print(results.summary())
beta=results.params
r2 = results.rsquared 
lbs_hat=results.fittedvalues

'''
plt show:
import matplotlib.pyplot as plt
plt.plot(cigs_lbs["cigs"], cigs_lbs["lbs"],marker="o",linestyle=" ")
plt.plot(cigs_lbs["cigs"], lbs_hat ,linestyle="-")
plt.ylabel("lbs")
plt.xlabel("cigs")
plt.show()
'''

u_hat = results.resid
u_hat_mean = np.mean(u_hat)
cigs_u_cov = np.cov((cigs_lbs["cigs"], u_hat) [1:0])
cigs_mean = np.mean(cigs_lbs["cigs"])
lbs_hat =beta[0] + beta[1]*cigs_mean
lbs_mean = np.mean(cigs_lbs["lbs"])

lbs = cigs_lbs["lbs"]
cigs = cigs_lbs["cigs"]
cigs_train ,cigs_test ,lbs_train ,lbs_test=train_test_split (cigs,lbs,test_size=.4,random_state=9)
X_train = sm.add_constant(cigs_train)
model = sm.OLS(lbs_train ,X_train)
results =model.fit()
print(results.summary())
beta_train =results.params
u_hat_test = lbs_test - beta_train[0] - beta_train[1]*cigs_test
MSE =np.mean (u_hat_test**2)
print("MSE: ",MSE)

# llbs = np.log(cigs_lbs["lbs"])
# cigs_lbs = cigs_lbs.assign(llbs=llbs)
# model = smf.ols(formula="llbs~cigs",data=cigs_lbs)
# results =model.fit()
# beta =results.params
# llbs_hat=results .fittedvalues
# r2 = results.rsquared

# llbs=np.log(cigs_lbs["lbs"])
# cigs_lbs =cigs_lbs.assign(llbs=llbs)
# cigs_train,cigs_test,llbs_train,llbs_test=train_test_split (cigs,llbs,test_size  =.4,random_state=9 )
# X_train = sm.add_constant (cigs_train )
# model = sm.OLS (lbs_train , X_train)
# results2 =model.fit ()
# print(results2.summary ())
# llbs_hat_test=  beta_train [0] + beta_train[1]*cigs_test
# lbs_hat_test = np.exp(llbs_hat_test )
# lbs_test = np.exp(llbs_test)
# u2_hat_test = lbs_test - lbs_hat_test
# MSE2 =np.mean (u2_hat_test**2)
# print("MSE: ",MSE) #MSE:  1.555206690257327

cigssq =np.square(cigs_lbs["cigs"])
cigs_lbs =cigs_lbs.assign(cigssq=cigssq)
model = smf.ols(formula="lbs~cigs+cigssq", data=cigs_lbs)
results =model.fit()
beta =results.params
lbs_hat=results.fittedvalues
r2 = results.rsquared
# cigssq =np.square(cigs_lbs["cigs"])
# cigs_lbs =cigs_lbs.assign(cigssq=cigssq)
cigs_train ,cigs_test ,cigssq_train ,cigssq_test ,lbs_train ,lbs_test=train_test_split (cigs,cigssq,lbs,test_size =0.4)
X_train = pd.concat([cigs_train ,cigssq_train],axis=1)
X_train = sm.add_constant(X_train)
model = sm.OLS(lbs_train ,X_train)
results =model.fit()
print(results.summary())
beta_train = results.params
u_hat_test = lbs_test - beta_train[0] - beta_train[1]*cigs_test -beta_train[2]*cigssq_test
MSE =np.mean (u_hat_test**2)
print("MSE: ",MSE) #MSE:  1.4522356025380312

#conclusion
#bwghtlbs=β0+cigsβ1+cigs2β2+u 這個model is better