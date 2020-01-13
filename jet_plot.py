import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


M = 200
N = 100
W = 100

def data_matrix(i):
    data_txt = open("data/X_Y_"+str(i)+".dat", "r")

    data_read = data_txt.readlines()
    data = []

    for line in data_read:
        row = []
        row_txt = line.split()
        for num in row_txt:
            row.append(float(num))
        if row == []:
            continue
        else:
            data.append(row)

    data = np.asarray(data)
    m = np.zeros((N,M))

    for i in range(M):
        for j in range(N):
            m[j][i] = data[i*N+j][4]

    return m 

for i in range(300,350):
    plt.imshow(data_matrix(2*i), cmap='jet')
    plt.savefig("render/jet_velocity_"+str(i)+".png")