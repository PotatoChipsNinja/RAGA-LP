import matplotlib.pyplot as plt
import numpy as np

losses = np.load('loss.npy')
print(losses)

plt.plot(losses[1:])
plt.show()