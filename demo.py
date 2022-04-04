from CancerModelClass import CancerModel
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

lambda_p = 0.6326
delta_qp = 0.6455
gamma_q = 1.3495
gamma_p = 4.6922
KDE = 0.10045
k_qpp = 0.0
k_pq = 0.43562
K = 192.418



def cancer_plot(t,P,Q,Q_p):
    plt.plot(t,P,label="P",color='g')
    plt.plot(t,Q,label="Q",color='r')
    plt.plot(t,Q_p,label="Q_p",color='black')
    plt.plot(t,P+Q+Q_p, label="P+Q+Q_p",color='w')

x1 = [2.1730773841597526, 82.79733369328996, 0.0, 1.0]

print('Parameters from starting treatment')
print(x1)
m1 = CancerModel(lambda_p, delta_qp, gamma_q, gamma_p, KDE, k_qpp, k_pq, K)
t = m1.time_interval(0, 200)

x = odeint(m1.model,x1, t)

P = x[:, 0]
Q = x[:, 1]
Q_p = x[:, 2]
C = x[:, 3]

plt.title('Cancer')
plt.xlabel("months")
plt.ylabel('volume [mm]')
cancer_plot(t,P,Q,Q_p)
plt.show()


df = pd.DataFrame()
df['P'] = P
df['Q'] = Q
df['Q_p'] = Q_p
df['C'] = C
df.to_csv("sztucznyDemo.csv")