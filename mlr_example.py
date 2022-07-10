import numpy as np
import dn_multiple_linear_regression
import dn_scale_norm

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

model = dn_multiple_linear_regression.MLRegression()

model.fit(X_train, y_train, 10, 0.001)
