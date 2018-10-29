=====================================
Multi (& Mono) Data-Driven Sparse PLS
=====================================

	*mddspls is the python light package of the data-driven sparse PLS algorithm*

In the high dimensional settings (large number of variables), one objective is to select the relevant variables and thus to reduce the dimension. That subspace selection is often managed with supervised tools. However, some data can be missing, compromising the validity of the sub-space selection. We propose a PLS, Partial Least Square, based method, called **dd-sPLS** for data-driven-sparse PLS, allowing jointly variable selection and subspace estimation while training and testing missing data imputation through a new algorithm called Koh-Lanta.

It contains one main class **mddspls** and one associated important method denote **predict** permitting to predict from a new dataset. The function called **perf_mddsPLS** permits to compute cross-validation.