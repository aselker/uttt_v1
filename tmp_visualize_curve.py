import numpy as np
import matplotlib.pyplot as plt

values = np.arange(-0.999, 0.999, 0.001)
# values = np.array([0.13566712, 0.51654632, 0.01534753, 0.07794687, 0.09706823, 0.15742393])

assert np.all(values > -1)  # Avoid 1/0
weights = 1 / (1 + (values - 1) / 2) - 1
assert np.all(0 <= weights)  # Negative weight?  Could just clip value I think, /shrug
weights /= np.sum(weights)
plt.plot(values, np.log(weights), '.')
plt.show()
