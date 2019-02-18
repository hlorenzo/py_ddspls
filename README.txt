=====================================
Multi (& Mono) Data-Driven Sparse PLS
=====================================

In the high dimensional settings (large number of variables), one objective is to select the relevant variables and thus to reduce the dimension. That subspace selection is often managed with supervised tools. However, some data can be missing, compromising the validity of the sub-space selection. We propose a PLS, Partial Least Square, based method, called **dd-sPLS** for data-driven-sparse PLS, allowing jointly variable selection and subspace estimation while training and testing missing data imputation through a new algorithm called Koh-Lanta.