import matplotlib.pyplot as plt
import numpy as np


f1_score_q0 = [0.16872603 ,0.65700895 ,0.69407755 ,0.69706005 ,0.7068598 ,0.73072 ,0.7409459 ,0.7315722 ,0.730294 , 0.73455477 ,
               0.74563265 ,0.75756294 ,0.7562846 ,0.76991904 ,0.7690669 ,0.7665104 ,0.76267576 ,0.60374945 ,0.76821476 ,0.7618236]
f1_score_fedavg = [0.20025565, 0.66893905, 0.69620794, 0.7013208, 0.69961655, 0.73412865, 0.7371112, 0.73327655,0.730294 , 0.75074565,
                   0.75415426, 0.75713676, 0.75756294, 0.747337, 0.7498935, 0.74520665,0.74520665, 0.7601193, 0.74478066, 0.752876]

auroc_score_q0 = [0.4723231 ,0.812041 ,0.82942885 ,0.83061427 ,0.8438611 ,0.85232407 ,0.86369103 ,0.8599138 ,0.8565796 ,0.8587117 ,
                  0.8641929 ,0.867605 ,0.85744876 ,0.8708392 ,0.8739114 ,0.8646572 ,0.86660385 ,0.78350633 ,0.8784652 ,0.87254316 ]
auroc_score_fedavg = [0.47124478,0.8129606,0.83414173,0.8296842,0.83243775,0.85245794,0.8573732,0.85416794,0.8398215,0.85239047,
                      0.85398036,0.8541854,0.8502745,0.84329385,0.8467997,0.8563009,0.8461588,0.8435621,0.823694,0.8581743]

f1_score_q1 = [0.53977275, 0.47017047, 0.66903406, 0.6448864, 0.64914775, 0.65767044, 0.6761364, 0.65625, 0.6761364, 0.6619318, 
               0.68039775, 0.6832386, 0.6875, 0.6875, 0.6832386, 0.67897725, 0.66335225, 0.6832386, 0.6676136, 0.671875 ]
auroc_score_q1 = [0.65136796, 0.6102907, 0.78619856, 0.8045974, 0.81604964, 0.8115975, 0.80790615, 0.81207204, 0.8141725, 0.81585383, 
                  0.81997496, 0.8240168, 0.82403636, 0.8260825, 0.7900481, 0.79417306, 0.77882504, 0.79897195, 0.79036665, 0.7923074, ]
x = range(1, 21)






# first figure is a plot of both F1 scores from q=0 and normal fedavg
fig_F1_q_0 = plt.figure(1)

F1_q = plt.plot(x, f1_score_q0, 'r', label = 'q-fedavg_0')
F1fedavg = plt.plot(x, f1_score_fedavg, 'g', label = 'fedavg')
plt.xlabel('comunication')
plt.ylabel('F1 score')
plt.legend()


# Second plot is of both AUROC scores from q=0 and normal fedavg
fig_AUROC_q_0 = plt.figure(2)
AUROC_q = plt.plot(x, auroc_score_q0, 'r', label = 'auroc_q_0')
AUROC_fedavg = plt.plot(x, auroc_score_fedavg, 'g', label = 'fedavg')
plt.xlabel('comunication')
plt.ylabel('F1 score')
plt.legend()


# third plot is of F1 and AUROC from q=1 L = 1
fig_q_1 = plt.figure(3)
F1_q_1 = plt.plot(x, f1_score_q1, 'b', label = 'F1_q1')
AUROC_q_1 = plt.plot(x, auroc_score_q1, 'g', label = 'AUROC_q_1')
plt.xlabel('comunication')
plt.ylabel('F1/AUROC')
plt.legend()
plt.show()
