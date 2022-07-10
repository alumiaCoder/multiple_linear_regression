import numpy as np
import dn_multiple_linear_regression
import dn_scale_norm

if __name__ == "__main__":
    
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    model = dn_multiple_linear_regression.MLRegression()

    scaler_X = dn_scale_norm.Normalizer()
    scaler_y = dn_scale_norm.Normalizer()

    scaler_X.fit(X_train, [-1,1])
    scaler_y.fit(X_train, [-1,1])

    new_X = scaler_X.transform(X_train)
    new_y = scaler_y.transform(y_train)

    model.fit(new_X, new_y, 10000, 0.001)
