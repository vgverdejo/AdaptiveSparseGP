# Adaptive Sparse Gaussian Processes

This repository includes the implemention of the Adaptive Sparse Gaussian Process proposed in the paper:

Vanessa Gómez-Verdejo, Emilio Parrado-Hernández and Manel Martínez-Ramón, "Adaptive Sparse Gaussian Process Regression". IEEE Trans. Neural Networks and Learning Systems, 2023.

In the repository you can find:
* lib/: the library files with the class and utilities to run the Adaptvie Sparse GP models proposed in the above paper.
* notebooks/: three notebooks which let you replicate the paper experiments. Besides, there is an additional notebook ('Demo_ToyProblem') that you can use to get familiar with the different models and functions.

To run this code you have to take into account:
* To run the proposed model, you need the choldate library from https://github.com/modusdatascience/choldate
* To run some of the refence models included in the experimental results, you need to install some specific libreries. So, go to the authors github repository (https://github.com/wjmaddox/online_gp) for details in this instalation.
* To run the experiments of load forecasting, you need to download the data set. This is freely avaliable at: https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info. 
* To run the experiments of Shop Prediction you need these datasets from kaggle: https://archive.ics.uci.edu/ml/datasets/Online+Retail and https://www.kaggle.com/code/ryanholbrook/linear-regression-with-time-series/data?select=train.csv

**Some tips**:

We advise to create a python enviroment, for instance:

    conda create --name adaptive_gp 

    conda activate adaptive_gp


Later, to install choldate, run:

    pip install cython

    unzip master.zip

    cd choldate-master/

    python -m pip install .


Be carefull your sytem path is correctly defined. If the choldate path is not found you can force it as:

    PYTHON_LIBRARY=YOUR_ENV_PATH/lib/pythonX.X/site-package


    

