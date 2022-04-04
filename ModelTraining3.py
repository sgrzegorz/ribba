import math
import pyabc

import numpy as np
from scipy.integrate import odeint
import scipy.stats as st
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# pyabc.settings.set_figure_params('pyabc')  # for beautified plots
from CancerModelClass import CancerModel

import numpy as np
import pandas as pd


patient = pd.read_csv('sztucznyDemo.csv')

observation = np.array([ patient["P"].tolist(),  patient["Q"].tolist(), patient["Q_p"].tolist(), patient["C"].tolist()],dtype=float)
count = len(patient["P"].tolist())

def model(parameters):
    X0 = [parameters["P"], parameters["Q"], parameters["Q_p"], parameters["C"]]

    lambda_p = parameters["lambda_p"]
    delta_qp = parameters["delta_qp"]
    gamma_q = parameters["gamma_q"]
    gamma_p = parameters["gamma_p"]
    KDE = parameters["KDE"]
    k_qpp = parameters["k_qpp"]
    k_pq = parameters["k_pq"]
    K = parameters["K"]


    m1 = CancerModel(lambda_p, delta_qp, gamma_q, gamma_p, KDE, k_qpp, k_pq, K)
    t = m1.time_interval(0,len(patient["P"]))

    x = odeint(m1.model, X0, t)

    P = x[:, 0]
    Q = x[:, 1]
    Q_p = x[:, 2]
    C = x[:, 3]

    return {"data" :np.array([ P,  Q,  Q_p,C],dtype=float)}


def distance(x, y):
    dif = 0
    p=0.0
    q=0.0
    qp=0.0
    c=0.0
    for i in range(count):
        dif += abs(x["data"][0][i] - y["data"][0][i])
    return dif


def distance1(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][1][i] - y["data"][1][i])
    return dif

def distance2(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][2][i] - y["data"][2][i])
    return dif

def distance3(x, y):
    dif = 0
    for i in range(count):
        dif += abs(x["data"][3][i] - y["data"][3][i])
    return dif


prior = pyabc.Distribution(lambda_p=pyabc.RV("uniform", 0, 1), delta_qp=pyabc.RV("uniform", 0.1, 1), gamma_q=pyabc.RV("uniform", 0.1, 2), gamma_p=pyabc.RV("uniform", 0.1, 10), KDE=pyabc.RV("uniform", 0.01, 0.5), k_qpp=pyabc.RV("uniform", 0.0, 0.5),
                           k_pq=pyabc.RV("uniform", 0.1, 0.7), K=pyabc.RV("uniform", 0.01, 200),P=pyabc.RV("uniform", 0.01, 6),Q=pyabc.RV("uniform", 0.01, 100),Q_p=pyabc.RV("uniform", 0, 1),C=pyabc.RV("uniform", 0.0, 1))

dist = pyabc.distance.AggregatedDistance([distance,distance1,distance2,distance3])
abc = pyabc.ABCSMC(model, prior, dist, population_size=1000)

db_path = ("sqlite:///" +  os.path.join(tempfile.gettempdir(), "test.db"))

abc.new(db_path, {"data": observation})

history = abc.run(minimum_epsilon=1.0, max_nr_populations=5000)

history is abc.history

posterior2 = pyabc.MultivariateNormalTransition()
posterior2.fit(*history.get_distribution(m=0))

t_params = posterior2.rvs()

print(t_params)