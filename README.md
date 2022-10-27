# Adaptive Sparse Gaussian Processes

This repository includes the implemention of the Adaptive Sparse Gaussian Process proposed in the paper:

Vanessa Gómez-Verdejo and Manel Martínez-Ramón, "Adaptive Sparse Gaussian Process Regression". Submitted to IEEE Trans. Neural Networks and Learning Systems, 2022.

In the repository you can find:
* lib/: the library files with the class and utilities to run the Adaptvie Sparse GP models proposed in the above paper.
* notebooks/: two notebooks which let you replicate the paper experiments. Besides, there is an additional notebook ('Demo_ToyProblem') that you can use to get familiar with the different models and functions.

To run this code you have to take into account:
* To run some of the refence models included in the experimental results, you need to install some specific libreries. So, go to the authors github repository (https://github.com/wjmaddox/online_gp) for details in this instalation.
* To run the experiments of load forecasting, you need to download the data set. This is freely avaliable at: https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info. 

