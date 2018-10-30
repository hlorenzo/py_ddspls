=====================================
Multi (& Mono) Data-Driven Sparse PLS
=====================================

	*mddspls is the python light package of the data-driven sparse PLS algorithm*

In the high dimensional settings (large number of variables), one objective is to select the relevant variables and thus to reduce the dimension. That subspace selection is often managed with supervised tools. However, some data can be missing, compromising the validity of the sub-space selection. We propose a PLS, Partial Least Square, based method, called **dd-sPLS** for data-driven-sparse PLS, allowing jointly variable selection and subspace estimation while training and testing missing data imputation through a new algorithm called Koh-Lanta.

It contains one main class **mddspls** and one associated important method denote **predict** permitting to predict from a new dataset. The function called **perf_mddsPLS** permits to compute cross-validation.

Data simulation
===============
One might be interested to simulate data and test the package through **regression** and **classification**::

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

The dd-sPLS regularization parameter is fixed to 0.6::

	lambd=0.6

A train/test dataset is defined::

	id_train = range(30,100)
	id_test = range(30)
	Xtrain = {0:X0[id_train,:],1:X1[id_train,:],2:X2[id_train,:]}
	Ytrain = Y[id_train,:]
	Xtest = {0:X0[id_test,:],1:X1[id_test,:],2:X2[id_test,:]}

Regression analysis
-------------------

Let us produce *2* axes::

	R=2

Start model building and tcheck results with sklearn tools::

	mod_0=py_ddspls.model.ddspls(Xtrain,Ytrain,lambd=lambd,R=R,mode="reg",verbose=True)
	Y_est_reg = mod_0.predict(Xtest)		
	print(sklm.mean_squared_error(Y[id_test,:],Y_est_reg))

Leave-one-out cross validation can be performed with built tools, the parameter **NCORES** permits to fix the number of cores to be used in the process ::

	perf_model_reg = py_ddspls.model.perf_ddspls(Xs,Y,R=R,kfolds="loo",n_lambd=10,NCORES=4,mode="reg")
	print(perf_model_reg)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(perf_model_reg[:,1], perf_model_reg[:,2], 'r',perf_model_reg[:,1], perf_model_reg[:,3],'b')
	plt.legend(('Y_1 RMSE', 'Y_2 RMSE'),loc='upper')
	plt.title('Leave-One-Out Cross-validation error against $\lambda$')
	plt.xlabel('$\lambda$')
	plt.ylabel('$RMSE$')
	plt.show()

Which returns this kind of graphics .. image::

	https://raw.githubusercontent.com/hlorenzo/py_ddspls/master/images/reg.png
	:width: 600

One that figure one can see that $\lambda\approx 0.79$ permits to estimate both of the $Y$ responses.

Classification analysis
-----------------------

Let us produce *1* axis::

	R=1

Start model building and tcheck results with sklearn tools::

	mod_0_classif=py_ddspls.model.ddspls(Xs,Y_bin,lambd=lambd,R=R,mode="clas",verbose=True)
	Y_est = mod_0_classif.predict(Xtest)
	print(sklm.classification_report(Y_est, Y_classif[id_test]=='Class 1'))

Cross validation can be performed with built tools, the parameter **NCORES** permits to use parallellization::

	perf_model_class = py_ddspls.model.perf_ddspls(Xs,Y_classif,R=1,kfolds="loo,n_lambd=10,NCORES=5,mode="classif")
	print(perf_model_class)


**Enjoy :)**