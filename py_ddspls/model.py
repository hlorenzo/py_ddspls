# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from pandas import get_dummies
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	import imp
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from multiprocessing import Pool
import random as rd
from itertools import product as itertools_product

def getResult(dic):
	Xs = dic["Xs"]
	Y = dic["Y"]
	q = dic["q"]
	mode = dic["mode"]
	maxIter_imput = dic["maxIter_imput"]
	errMin_imput = dic["errMin_imput"]
	paras = dic["paras"]
	decoupe = dic["decoupe"]
	pos_decoupe = dic["pos_decoupe"]
	fold = dic["fold"]
	K = len(Xs)
	paras_here_pos = np.where(np.array(decoupe)==pos_decoupe)[0]
	l_p_h = len(paras_here_pos)
	if mode=="reg":
		errors = np.zeros((l_p_h,q))
		select_y = np.zeros((l_p_h,q))
	else:
		errors = []
		select_y = None
	for i in range(l_p_h):
		R = paras[0][paras_here_pos[i]]
		lambd = paras[1][paras_here_pos[i]]
		i_fold = paras[2][paras_here_pos[i]]
		pos_test = np.where(np.array(fold)==i_fold)[0]
		pos_train = np.where(np.array(fold)!=i_fold)[0]
		X_train = {}
		X_test = {}
		for k in range(K):
			X_train[k] = Xs[k][pos_train,:]
			X_test[k] = Xs[k][pos_test,:]
		if mode=="reg":
			Y_train = Y[pos_train,:]
			Y_test = Y[pos_test,:]
		else:
			Y_train = Y[pos_train]
			Y_test = Y[pos_test]
		mod_0 = ddspls(Xs=X_train,Y=Y_train,lambd=lambd,R=R,mode=mode,
			errMin_imput=errMin_imput,maxIter_imput=maxIter_imput)
		Y_est = mod_0.predict(X_test)
		if mode=="reg":
			error_here = Y_est - Y_test
			errors[i,:] = np.sqrt(np.sum(error_here*error_here,
	  axis=0)/len(pos_test))
			v_no_null = np.where(np.sum(abs(mod_0.model.v),axis=0)>1e-10)
			select_y[i,v_no_null] <- 1
		else:
			numer = float(len([ii for ii,v in enumerate(Y_test) if Y_est[ii] == v]))
			denom = float(len(Y_test))
			errors.append(1.-numer/denom)
	if mode=="reg":
		out = {"RMSE":errors,"v_no_null":v_no_null,"select_y":select_y}
	else:
		out = errors
	return out;

def reshape_dict(Xs_top):
	test_matrix = type(Xs_top)==np.ndarray
	if test_matrix:
		if(len(Xs_top.shape)!=0):
			Xs_w_resh = {}
			Xs_w_resh[0] = Xs_top
		else:
			Xs_w_resh = Xs_top	
	else:
		Xs_w_resh = Xs_top
	K = len(Xs_w_resh)
	how_much_dim = []
	min_dim = []
	for k in range(K):
		dim_k = Xs_w_resh[k].shape
		if len(dim_k)==0:
			dim_k_0 = (1)
			dim_k = dim_k_0
			min_dim.append(dim_k)
			how_much_dim.append(1)
		else:
			how_much_dim.append(2)
			min_dim.append(min(dim_k))
	if max(how_much_dim) == 1:
		for k in range(K):
			Xs_here = np.zeros((1,min_dim[k]))
			Xs_here[0,:] = Xs_w_resh[k]
			Xs_w_resh[k] = Xs_here
	else:
		for k in range(K):
			if how_much_dim[k]==1:
				Xs_here = np.zeros((Xs_w_resh[k].shape[0],1))
				Xs_here[:,0] = Xs_w_resh[k]
				Xs_w_resh[k] = Xs_here
	return Xs_w_resh;
				
def MddsPLS_core(Xs,Y,lambd=0,R=1,mode="reg",verbose=False):
	Xs_w = reshape_dict(Xs)
	K = len(Xs_w)
	n = Xs_w[0].shape[0]
	# Standardize X
	mu_x_s = {}
	sd_x_s = {}
	pos_nas = {}
	p_s = np.repeat(0,K)
	for i in range(K):
		if len(Xs_w[i].shape)!=2:
			Xs_w[i] = Xs_w[i].to_frame()
		p_i = Xs_w[i].shape[1]
		p_s[i] = p_i
		mu_x_s[i] = Xs_w[i].mean(0)
		sd_x_s[i] = Xs_w[i].std(0)
		pos_nas[i] = np.where(np.isnan(Xs_w[i][:,0]))[0]
		pos_no_na = np.where(np.isnan(Xs_w[i][:,0])==False)[0]
		Xs_w[i][pos_no_na,:] = preprocessing.scale(Xs_w[i][pos_no_na,:])
		if len(pos_nas[i])!=0:
			Xs_w[i][pos_nas[i],:] = 0
	# Standardize Y
	if mode != "reg":
		Y_w = get_dummies(Y)
	else:
		Y_w = Y
	q = Y_w.shape[1]
	mu_y = Y_w.mean(0)
	sd_y = Y_w.std(0)
	Y_w = preprocessing.scale(Y_w*1.0)
	# Create soft-thresholded matrices
	Ms = {}
	for i in range(K):
		M0 = Y_w.T.dot(Xs_w[i])/(n-1)
		M = abs(M0) - lambd
		pos_soft = np.where(np.sign(M)==-1)
		for j in range(len(pos_soft[0])):
			M[pos_soft[0][j],pos_soft[1][j]] = 0
		Ms[i] = np.multiply(np.sign(M0),M)
	u_t_r = {}
	u_t_r_0 = {}
	t_r = {}
	z_r = {}
	R_w = min(R,min(p_s))
	if R_w > R:
		R_w = R
	for k in range(K):
		if sum(sum(abs(Ms[k])))==0:
			svd_k = {"v":np.zeros((Ms[k].shape[1], R_w))}
		else:
			svd_k_res = np.linalg.svd(Ms[k],full_matrices=False)
			v_k_res = svd_k_res[2].T
			svd_k = {"v":v_k_res[:,range(R_w)]}
		u_t_r[k]=svd_k["v"]
		u_t_r_0[k]=svd_k["v"]
		if k==0:
			for r in range(R_w):
				t_r[r] = np.zeros((n, K))
				z_r[r] = np.zeros((q, K))
		for r in range(R_w):
			t_r[r][:,k] = np.dot(Xs_w[k],u_t_r[k][:,r])
			z_r[r][:,k] = np.dot(Ms[k],u_t_r[k][:,r])
	t = np.zeros((n, R_w))
	v = np.zeros((q, R_w))
	t_all = np.zeros((n, R_w*K))
	z_all = np.zeros((q, R_w*K))
	for r in range(R_w):
		z_all[:,np.repeat(r*K,K)+range(K)] = np.array(z_r[r])
		t_all[:,np.repeat(r*K,K)+range(K)] = np.array(t_r[r])
	svd_all_python = np.linalg.svd(z_all,full_matrices=False)
	u = svd_all_python[2].T
	v0 = svd_all_python[0]
	v = v0
	s = np.dot(Y_w,v0)
	t = np.dot(t_all,u)
	svd_all_frak_python = np.linalg.svd(np.dot(t.T,s),full_matrices=False)
	u_frak = svd_all_frak_python[0]
	v_frak = svd_all_frak_python[2].T
	t_frak = np.dot(t,u_frak)
	s_frak = np.dot(s,v_frak)
	# Get regression matrix
	alphas = []
	for r in range(R_w):
		n_t_2 = np.dot(t_frak[:,r].T,t_frak[:,r])
		if n_t_2 != 0:
			val = np.dot(s_frak[:,r].T,t_frak[:,r])/n_t_2
			alphas.append(val)
		else:
			alphas.append(0)
	if mode == "reg":
		B = {}
		for k in range(K):
			beta_k = u[np.repeat(k*R_w,R_w)+range(R_w),:]
			B[k] = np.dot(u_t_r[k],np.dot(beta_k,u_frak))
			for r in range(R_w):
				B[k][:,r] = B[k][:,r]*alphas[r]
			B[k] = np.dot(B[k],np.dot(v_frak.T,v.T))
	else:
		if np.sum(t_frak*t_frak)!=0:
			n_components = min(R_w,len(set(Y))-1)
			B = LinearDiscriminantAnalysis(n_components=n_components)
			B.fit(t,Y)
		else:
			B = None
	out = {"u":u_t_r,"v":v,"ts":t_r,"beta_comb":u,"t":t,"s":s,
	"t_frak":t_frak,"s_frak":s_frak,"B":B,"mu_x_s":mu_x_s,
	"sd_x_s":sd_x_s,"mu_y":mu_y,"sd_y":sd_y,"R":R_w,"q":q,
	"Ms":Ms,"lambd":lambd}
	return out;

class model_class:
	def __init__(self,u,v,ts,beta_comb,t,s,t_frak,s_frak,B,mu_x_s,sd_x_s,mu_y,sd_y,R,q,Ms,lambd):
		self.u = u
		self.v = v
		self.ts = ts
		self.beta_comb = beta_comb
		self.t = t
		self.s = s
		self.t_frak = t_frak
		self.s_frak = s_frak
		self.B = B
		self.mu_x_s = mu_x_s
		self.sd_x_s = sd_x_s
		self.mu_y = mu_y
		self.sd_y = sd_y
		self.R = R
		self.q = q
		self.Ms = Ms
		self.lambd = lambd
	
class ddspls:
	def __init__(self,Xs,Y,lambd=0,R=1,mode="reg",errMin_imput=1e-9,maxIter_imput=50,verbose=False,model=None):				
		self.Xs = Xs
		self.Y = Y
		self.lambd = lambd
		self.R = R
		self.mode = mode
		self.errMin_imput = errMin_imput
		self.maxIter_imput = maxIter_imput
		self.verbose = verbose
		n = Xs[0].shape[0] 
		if n!=1:
			self.getModel(model)
		else:
			self.model = {}

	def getModel(self,model):
		if model==None:
			Xs = self.Xs
			Y = self.Y
			lambd = self.lambd
			R = self.R
			mode = self.mode
			errMin_imput = self.errMin_imput
			maxIter_imput = self.maxIter_imput
			verbose = self.verbose
			Xs_w = reshape_dict(Xs)
			has_converged = maxIter_imput
			id_na = {}
			K = len(Xs_w)
			na_lengths = 0
			for k in range(K):
				id_na[k] = np.where(np.isnan(Xs_w[k][:,0]))[0]
				na_lengths = na_lengths + len(id_na[k])
			if na_lengths != 0:
				for k in range(K):
					if len(id_na[k]) != 0:
						mu_k = Xs_w[k].mean(0)
						for k_ik in id_na[k]:
							Xs_w[k][k_ik,:] = mu_k
			mod = MddsPLS_core(Xs_w,Y,lambd=lambd,R=R,mode=mode,verbose=verbose)
			if K>1:
				mod_0 = MddsPLS_core(Xs_w,Y,lambd=lambd,R=R,mode=mode,verbose=verbose)
				if sum(sum(abs(mod_0["t"])))!=0:
					err = 2
					iterat = 0
					while (iterat < maxIter_imput)&(err>errMin_imput):
						iterat = iterat+1
						for k in range(K):
							if len(id_na[k])>0:
								no_k = np.arange(K)
								np.delete(no_k ,k)
								i_k = id_na[k]
								Xs_i = mod_0["s"]
								Xs_i = np.delete(Xs_i, i_k, axis=0)
								newX_i = mod_0["s"][i_k,:]
								u_k = mod_0["u"][k].T[0]
								Var_selected_k = np.where(abs(u_k)!=0)[0]
								if(len(Var_selected_k)>0):
									Y_i_k = Xs_w[k][:,Var_selected_k]
									Y_i_k = np.delete(Y_i_k,i_k,axis=0)
									model_here_0 = MddsPLS_core(Xs_i,Y_i_k,lambd=lambd)
									model_here = model_class(u=model_here_0["u"],v=model_here_0["v"],ts=model_here_0["ts"],
								  beta_comb=model_here_0["beta_comb"],t=model_here_0["t"],s=model_here_0["s"],
								  t_frak=model_here_0["t_frak"],s_frak=model_here_0["s_frak"],B=model_here_0["B"],
								  mu_x_s=model_here_0["mu_x_s"],sd_x_s=model_here_0["sd_x_s"],mu_y=model_here_0["mu_y"],
								  sd_y=model_here_0["sd_y"],R=model_here_0["R"],q=model_here_0["q"],Ms=model_here_0["Ms"],
								  lambd=model_here_0["lambd"])
									mod_i_k = ddspls(Xs=Xs_i,Y=Y_i_k,lambd=lambd,R=R,
							  model=model_here,maxIter_imput=maxIter_imput,mode="reg")
									Xs_w[k][i_k,Var_selected_k] = mod_i_k.predict(newX_i)
						mod = MddsPLS_core(Xs_w,Y,lambd=lambd,R=R,mode=mode)
						if sum(sum(abs(mod["t"])))!=0:
							err = 0
							for r in range(R):
								n_new = np.sqrt(sum(np.square(mod["t_frak"][:,r])))
								n_0 = np.sqrt(sum(np.square(mod_0["t_frak"][:,r])))
								if n_new*n_0!=0:
									err_r = 1 - abs(np.dot(mod["t_frak"][:,r].T,mod_0["t_frak"][:,r]))/(n_new*n_0)
									err = err + err_r
						else:
							err = 0
						if iterat >= maxIter_imput:
							has_converged = 0
						if err < errMin_imput:
							has_converged = iterat
						mod_0 = mod
			self.model = model_class(u=mod["u"],v=mod["v"],ts=mod["ts"],
					  beta_comb=mod["beta_comb"],t=mod["t"],s=mod["s"],
					  t_frak=mod["t_frak"],s_frak=mod["s_frak"],B=mod["B"],
					  mu_x_s=mod["mu_x_s"],sd_x_s=mod["sd_x_s"],mu_y=mod["mu_y"],
					  sd_y=mod["sd_y"],R=mod["R"],q=mod["q"],Ms=mod["Ms"],
					  lambd=mod["lambd"])
			self.has_converged = has_converged
		else:
			self.model = model

	def fill_X_test(self,X_test_0):
		X_test = reshape_dict(X_test_0)
		lambd,R,mod = self.lambd,self.R,self.model
		K = len(X_test)
		n = mod.t_frak.shape[0]
		id_na_test = []
		# id_na_test : blocks with missing valueA
		na_test_lengths = 0
		X_test_w = X_test
		pos_vars_Y_here,t_X_here,number_coeff_no_ok = {},{},0
		for k in range(K):
			id_na_test.append((np.isnan(X_test[k][:,0]))[0]*1)
			na_test_lengths = na_test_lengths + id_na_test[k]
		if na_test_lengths != 0:
			pos_ok = np.where(np.array(id_na_test)==0)[0]
			len_pos_ok = len(pos_ok)
			t_X_here = np.zeros((n,len_pos_ok*R))
			for r in range(R):
				for id_ok_r in range(len_pos_ok):
					t_X_here[:,r*len_pos_ok+id_ok_r] = mod.ts[r][:,pos_ok[id_ok_r]]
			u_X_here,mu_x_here,sd_x_0 = {},{},{}
			for id_ok_r in range(len_pos_ok):
				u_X_here[id_ok_r]=mod.u[pos_ok[id_ok_r]]
				mu_x_here[id_ok_r]=mod.mu_x_s[pos_ok[id_ok_r]]
				sd_x_0[id_ok_r] = mod.sd_x_s[pos_ok[id_ok_r]]
		 
			## Create to be predicted matrix train
			pos_no_ok = range(K)
			pos_no_ok = [x for x in pos_no_ok if x not in pos_ok]
			len_pos_no_ok = len(pos_no_ok)
			for pp in range(len_pos_no_ok):
				u_pos_no_ok_pp = mod.u[pos_no_ok[pp]]
				pos_vars_Y_here[pp] = np.where(np.sum(abs(u_pos_no_ok_pp),
	                  axis=1)!=0)[0]
				number_coeff_no_ok = number_coeff_no_ok + len(pos_vars_Y_here[pp])
			if number_coeff_no_ok!=0:
				vars_Y_here = np.zeros((n,number_coeff_no_ok))
				C_pos = 0
				for k_id in range(len_pos_no_ok):
					vars_k_id = pos_vars_Y_here[k_id]
					len_vars_k_id = len(vars_k_id)
					if len_vars_k_id!=0:
						for j in range(len_vars_k_id):
							vars_Y_here[:,C_pos+j] = self.Xs[pos_no_ok[k_id]][:,j]
						C_pos = C_pos + len_vars_k_id
			else:
				vars_Y_here = np.zeros((n,1))
			## General model
			model_impute_test = ddspls(Xs=t_X_here,Y=vars_Y_here,
				lambd=lambd,R=R,
				maxIter_imput=self.maxIter_imput)
			## Create test dataset
			n_test = 1
			t_X_test=np.zeros((n_test,t_X_here.shape[1]))
			K_h = len(np.where(id_na_test==False)[0])
			for r_j in range(R):
				for k_j in range(K_h):
					kk = pos_ok[k_j]
					xx = X_test[kk]
					for id_xx in range(n_test):
						variab_sd_no_0 = np.where(sd_x_0[k_j]!=0)
						for v_sd_no_0 in variab_sd_no_0:
							xx[id_xx,v_sd_no_0] = (xx[id_xx,v_sd_no_0] - 
	        mu_x_here[k_j][v_sd_no_0])/sd_x_0[k_j][v_sd_no_0]
					t_X_test[:,r_j*K_h+k_j] = np.dot(xx,u_X_here[k_j][:,r_j])
			## Estimate missing values
			res = model_impute_test.predict(t_X_test)
			## Put results inside Xs
			C_pos = 0
			for k_id in range(len_pos_no_ok):
				vars_k_id = pos_vars_Y_here[k_id]
				len_v_k_id = len(vars_k_id)
				pos_n_k_h = pos_no_ok[k_id]
				X_test_w[pos_n_k_h][0,:] = mod.mu_x_s[pos_n_k_h]
				if len_v_k_id!=0:
					for vv in range(len_v_k_id):
						X_test_w[pos_n_k_h][0,vars_k_id[vv]] = res[0,C_pos + vv]
					C_pos = C_pos + len_v_k_id
		return X_test_w;

	def predict(self,newX):
		newX_w = reshape_dict(newX)
		K = len(newX_w)
		n_new = newX_w[0].shape[0]
		mod = self.model
		R = mod.R
		q = mod.q
		if n_new==1:
			id_na_test,na_test_lengths = [],0
			# id_na_test : blocks with missing value
			na_test_lengths = 0
			for k in range(K):
				id_na_test.append(np.isnan(newX_w[k][0,0])*1)
				na_test_lengths = na_test_lengths + id_na_test[k]
			if na_test_lengths!=0:
				if (K>1)&(self.maxIter_imput>0):
					newX_w = self.fill_X_test(newX)
				else:
					for k in range(K):
						if id_na_test[k] != 0:
							newX_w[k][0,:] = mod.mu_x_s[k]
			for k in range(K):
				variab_sd_no_0 = np.where(mod.sd_x_s[k]!=0)[0]
				for v_sd_no_0 in variab_sd_no_0:
					newX_w[k][0,v_sd_no_0] = (newX_w[k][0,v_sd_no_0] - 
						mod.mu_x_s[k][v_sd_no_0])/mod.sd_x_s[k][v_sd_no_0]
			if self.mode=="reg":
				newY = np.zeros((1,q))
				for k in range(K):
					newY = newY + np.dot(newX_w[k],mod.B[k])
				for q_i in range(q):
					newY[0,q_i] = newY[0,q_i]*mod.sd_y[q_i]+mod.mu_y[q_i]
			else:
				if mod.B != None:
					t_r_all = np.zeros((1,K*R))
					for k in range(K):
						for r in range(R):
							t_r_all[0,np.repeat(K*r,K)+range(K)] = np.dot(newX_w[k],mod.u[k][:,r])
					df_new = np.dot(t_r_all,mod.beta_comb)
					newY = mod.B.predict(df_new)
				else:
					newY = rd.sample(set(self.Y),1)
		else:
			if self.mode=="reg":		
				newY = np.zeros((n_new,q))
			else:
				newY = []
			for i_new in range(n_new):
				t_i_new = {}				
				for k in range(K):
					t_i_new[k] = newX[k][(i_new):(i_new+1),:]
				if self.mode=="reg":
					newY[i_new,:] = self.predict(t_i_new)
				else:
					newY.append(self.predict(t_i_new)[0])
		return newY;
	
def perf_ddspls(Xs,Y,lambd_min=0,lambd_max=None,n_lambd=1,lambds=None,R=1,
	kfolds="loo",mode="reg",fold_fixed=None,maxIter_imput=5,errMin_imput=1e-9,
	NCORES=1):

	def expandgrid(*itrs):
		product = list(itertools_product(*itrs))
		return {i:[x[i] for x in product] for i in range(len(itrs))};

	Xs_w = reshape_dict(Xs)
	K = len(Xs_w)
	n = Xs_w[0].shape[0]
	p_s = np.repeat(0,K)
	for i in range(K):
		p_s[i] = Xs_w[i].shape[1]
	if mode=="reg":
		Y_w = reshape_dict(Y)[0]
		q = Y_w.shape[1]
	else:
		q = 1
		Y_w = Y
	## cv design
	if kfolds=="loo":
		kfolds_w = n
		fold = range(n)
	elif kfolds=="fixed":
		fold = fold_fixed
	else:
		fold = []
		rapport = int(np.ceil(float(n)/float(kfolds)))
		val_to_sample = range(kfolds)
		for iterat in range(rapport):
			oo = rd.sample(val_to_sample,kfolds)
			for popo in val_to_sample:
				pos = iterat*kfolds + popo
				if pos<n:
					fold.append(oo[popo])
	## Get highest Lambda
	if lambds==None:
		if lambd_max == None:
			MMss0 = ddspls(Xs,Y,lambd = 0,R = 1,mode = mode,maxIter_imput = 0).model.Ms
			K = len(MMss0)
			lambd_max_w = 0
			for k in range(K):
				lambd_max_w = max([lambd_max_w,np.max(abs(MMss0[k]))])
		else:
			lambd_max_w = lambd_max
		lambds_w = np.linspace(lambd_min,lambd_max_w,n_lambd)
	else:
		lambds_w = lambds
	try:
		iter(R)
	except TypeError:
		R = [R]
	try:
		iter(lambds_w)
	except TypeError:
		lambds_w = [lambds_w]
	paras = expandgrid(R,lambds_w,range(max(fold)+1))
	if (NCORES>len(paras[0])):
		   decoupe = range(len(paras[0]))
	else:
		decoupe = []
		rapport = int(np.ceil(len(paras[0])/float(NCORES) ))
		val_to_sample = range(NCORES)
		for iterat in range(rapport):
			oo = rd.sample(val_to_sample,NCORES)
			for popo in val_to_sample:
				pos = iterat*NCORES + popo
				if pos<len(paras[0]):
					decoupe.append(oo[popo])
	paral_list = []
	for pos_decoupe in range(max(decoupe)+1):
		dicoco = {"Xs":Xs_w,"Y":Y_w,"q":q,"mode":mode,
			"maxIter_imput":maxIter_imput,"errMin_imput":errMin_imput,
			"paras":paras,"decoupe":decoupe,"pos_decoupe":pos_decoupe,
			"fold":fold}
		paral_list.append(dicoco)
	NCORES_w = int(min(NCORES,len(paras[0])))
	p = Pool(processes=NCORES_w)
	ERRORS = p.map(getResult, paral_list)
	paras_out = expandgrid(R,lambds_w)
	if mode=="reg":
		ERRORS_OUT = np.zeros((len(paras_out[0]),q))
		DF_OUT = np.zeros((len(paras_out[0]),2+q))
	else:
		ERRORS_OUT = []
		DF_OUT = np.zeros((len(paras_out[0]),2+1))
	for i in range(len(paras_out[0])):
		R_yo = paras_out[0][i]
		lambd_yo = paras_out[1][i]
		DF_OUT[i,range(2)] = R_yo,lambd_yo
		errs = []
		for koko in range(len(paral_list)):
			pos_decoupe = paral_list[koko]["pos_decoupe"]
			pos_pos_decoupe = np.where(np.array(decoupe)==pos_decoupe)[0]
			R_koko = [paras[0][i_loc] for i_loc in pos_pos_decoupe]
			lambd_koko = [paras[1][i_loc] for i_loc in pos_pos_decoupe]
			for ll in range(len(R_koko)):
				if (R_koko[ll]==R_yo)&(lambd_koko[ll]==lambd_yo):
					if mode =="reg":
						errs.append(ERRORS[koko]["RMSE"][ll,])
					else:
						errs.append(ERRORS[koko][ll])
		DIM = len(errs)
		DIM_2 = 1
		if type(errs[0])!=float:
			DIM_2 = len(errs[0])
		if mode=="reg":
			ERRS_KOKO = np.zeros((DIM,DIM_2))
		else:
			ERRS_KOKO = []
		for koko in range(DIM):
			if mode=="reg":
				ERRS_KOKO[koko,:] = errs[koko]
			else:
				ERRS_KOKO.append(errs[koko])
		if mode=="reg":
			aaa = np.sqrt(np.sum(ERRS_KOKO*ERRS_KOKO,axis=0)/DIM)
			for ppp in range(len(aaa)):
				DF_OUT[i,2+ppp] = aaa[ppp]
		else:
			ERRORS_OUT.append(np.sum(ERRS_KOKO)/DIM)
	if mode!="reg":
		for iii in range(len(ERRORS_OUT)):
			DF_OUT[iii,2:DF_OUT.shape[1]] = ERRORS_OUT[iii]
	p.terminate()
	return DF_OUT;