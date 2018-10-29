#!/usr/bin/env python

import py_ddspls
import numpy as np
import sklearn.metrics as sklm
n = 100
mean = (0,0,0,0,0,0,0,0,0)
cov = [[1, 0.8,0.8,0.8,0.1,0.1,0.1,0.1,0.1], 
[0.8,1, 0.8,0.8,0.1,0.1,0.1,0.1,0.1],
[0.8,0.8,1, 0.8,0.1,0.1,0.1,0.1,0.1],
[0.8,0.8,0.8,1, 0.1,0.1,0.1,0.1,0.1],
[0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1],
[0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1],
[0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1],
[0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1],
[0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1]]
df = np.random.multivariate_normal(mean, cov, n)
Y = df[:,[0]]
k_groups = 2
lolo = np.linspace(min(Y),max(Y),k_groups+1)
Y_bin = np.zeros(n)
for ii in range(n):
	for k_i in range(k_groups):
		if (Y[ii]>=lolo[k_i])&(Y[ii]<lolo[k_i+1]):
			Y_bin[ii] = k_i
		if Y[ii]==lolo[k_groups]:
			Y_bin[ii] = k_groups-1

Y = df[:,[0,2]]
X0 = df[:,[1,4,5]]
X0[0,:] = None
X1 = df[:,[6,8]]
X1[:,1] = 1
X2 = df[:,[3,7]]
Xs = {0:X0,1:X1,2:X2}
pos_0 = np.where(Y_bin==0)[0]
pos_1 = np.where(Y_bin==1)[0]
Y_classif = np.repeat("Class 2",n)
Y_classif[pos_1] = "Class 1"

# dd-sPLS regularization parmater is fixed to 0.6
lambd=0.6

# A train/test dataset is defined
id_train = range(30,100)
id_test = range(30)
Xtrain = {0:X0[id_train,:],1:X1[id_train,:],2:X2[id_train,:]}
Ytrain = Y[id_train,:]
Xtest = {0:X0[id_test,:],1:X1[id_test,:],2:X2[id_test,:]}

# Here is performed **regression analysis**
R=2
mod_0=py_ddspls.model.ddspls(Xtrain,Ytrain,lambd=lambd,R=R,mode="reg",verbose=True)
Y_est_reg = mod_0.predict(Xtest)		
print(sklm.mean_squared_error(Y[id_test,:],Y_est_reg))
perf_model_reg = py_ddspls.model.perf_ddspls(Xs,Y,R=R,kfolds=3,n_lambd=3,NCORES=3,mode="reg")

# Here is performed **classification analysis**
R=1
mod_0_classif=py_ddspls.model.ddspls(Xs,Y_bin,lambd=lambd,R=R,mode="clas",verbose=True)
Y_est = mod_0_classif.predict(Xtest)
print(sklm.classification_report(Y_est, Y_classif[id_test]=='Class 1'))

perf_model_class = py_ddspls.model.perf_ddspls(Xs,Y_classif,R=1,kfolds=3,n_lambd=3,NCORES=3,mode="classif")