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
    
    def gradient_descent(self, X: np.array, y: np.array, w: np.array, b: int, alpha: float):
        
        d_dw = np.zeros((self.n_features))
        m  = self.n_examples
        dj_db = 0
        
        for i in range(m):
            error = np.dot(w, X[i]) + b - y[i]
            for j in range(self.n_features):
                d_dw[j] += error*X[i, j]

            dj_db += error
        
        dj_db = dj_db/m
        d_dw = d_dw/m

        w = w - alpha*d_dw
        b = b - alpha*dj_db

        return w, b

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

        #w = np.random.rand(n_features)
        #b = np.random.rand(1)

        b = 785.1811367994083
        w = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

        for epoch in range(epochs): 
            
            self.w_history.append(w)
            self.b_history.append(b)

            current_cost = self.cost_function(X, y, w, b)
            #if epoch%1000 == 0:
            print(f"{w, b}Epoch {epoch}, current cost is {current_cost}")
            self.cost_history.append(current_cost)
            w, b = self.gradient_descent(X, y, w, b, alpha)

            #if epoch%1000 == 0:
            #print(f"{w, b}Epoch {epoch}, current cost is {current_cost}")

    def predict(self, X: np.array):
        n_examples = X.shape[0]        
        yhat = np.array(n_examples)

        for i in range(n_examples):
            yhat[i] = np.dot(w, X[i]) + b

        return yhat