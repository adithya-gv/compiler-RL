import matplotlib.pyplot as plt
import numpy as np

f = open('data.txt', 'r')
data = f.read()
f.close()

g = open('data2.txt', 'r')
data2 = g.read()
g.close()


data = data.split('\n')
data = np.array([float(i) for i in data])

data2 = data2.split('\n')
data2 = np.array([float(i) for i in data2])

data3 = data - data2
plt.plot(data3)
plt.title("DQN Advantage over Random on 150 Games")
plt.show()