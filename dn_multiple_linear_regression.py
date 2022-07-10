import numpy as np

class MLRegression():

    def __init__(self):
        self.n_examples = 0
        self.n_features = 0

    def cost_function(self, X: np.array, y: np.array, w: np.array, b: int):
        m = X.shape[0]
        total_cost = 0.0
        error = 0.0
        for i in range(m):
            error += (np.dot(w, X[i]) + b - y[i])**2

        total_cost = error/(2*m)
        return total_cost
    
    def gradient_descent(self, X: np.array, y: np.array, w_in: np.array, b_in: int, alpha: float):
        
        w = w_in.copy()
        b = b_in
        
        d_dw = np.zeros((self.n_features))
        d_db = 0
        
        for i in range(self.n_examples):
            error = np.dot(w, X[i]) + b - y[i]
            for j in range(self.n_features):
                d_dw[j] += error*X[i, j]

            d_db += error
        
        d_db = d_db/self.n_examples
        d_dw = d_dw/self.n_examples

        return d_dw, d_db

    def fit(self, X: np.array, y: np.array, epochs: int, alpha: float):

        self.w_history = list()
        self.b_history = list()
        self.cost_history = list()
        
        if len(X.shape) > 1:
            n_examples = X.shape[0]
            n_features = X.shape[-1]
        else:
            n_examples = 1
            n_features = X.shape[-1]

        self.n_examples = n_examples
        self.n_features = n_features

        w = np.random.rand(n_features)
        b = np.random.rand(1)

        for epoch in range(epochs): 
            d_dw, d_db = self.gradient_descent(X, y, w, b, alpha)

            w = w - alpha*d_dw
            b = b - alpha*d_db

            self.w_history.append(w)
            self.b_history.append(b)

            if epoch%1000 == 0:
                current_cost = self.cost_function(X, y, w, b)
                print(f"{w, b}Epoch {epoch}, current cost is {current_cost}")

            self.cost_history.append(current_cost)


    def predict(self, X: np.array):
        n_examples = X.shape[0]        
        yhat = np.array(n_examples)

        for i in range(n_examples):
            yhat[i] = np.dot(w, X[i]) + b

        return yhat