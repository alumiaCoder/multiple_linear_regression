import numpy as np

class Normalizer():
    """A scaler that performs normalization to given range.
    Deals with multiple feature/multiple parameter arrays
    Expects a 3D array
    """    
    def __init__(self):
        self.parameters = list() #stores max and min for each feature prior to transformation
        self.n_examples = 0
        self.n_features = 0
        self.norm_range = list()

    def fit(self, X: np.array, norm_range: list()):

        parameters = list()        
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        if len(X_1.shape) > 1:
            n_examples = X_1.shape[0] 
            n_features = X_1.shape[1]
        else:
            n_examples = 1
            n_features = X_1.shape[-1]

        #find maximum and minimum for each column, store it and perform transformation
        for i in range(n_features):
            temp_max = np.max(X_1[:, i:i+1], axis=None)
            temp_min = np.min(X_1[:, i:i+1],axis=None)
            parameters.append([temp_min, temp_max])

        self.n_examples = n_examples
        self.n_features = n_features
        self.parameters = parameters
        self.norm_range = norm_range

        return parameters
    
    def transform(self, X: np.array):
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        for i in range(self.n_features):
            X_1[:,i:i+1] = ((X_1[:, i:i+1] - self.parameters[i][0])/(self.parameters[i][1]-self.parameters[i][0]))*(self.norm_range[1]-self.norm_range[0])+self.norm_range[0] 
        
        return X_1

    def inverse_transform(self, X: np.array):
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        for i in range(self.n_features):
            X_1[:,i:i+1] = ((X_1[:, i:i+1] - self.norm_range[0])/(self.norm_range[1]-self.norm_range[0]))*(self.parameters[i][1]-self.parameters[i][0])+self.parameters[i][0] 
        
        return X_1

class MeanScaler():
    """A scaler that performs mean normalization.
    Deals with multiple feature/multiple parameter arrays
    Expects a 3D array
    """    
    def __init__(self):
        self.parameters = list() #stores max and min for each feature prior to transformation
        self.n_examples = 0
        self.n_features = 0

    def fit(self, X: np.array):

        parameters = list()        
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        if len(X_1.shape) > 1:
            n_examples = X_1.shape[0] 
            n_features = X_1.shape[1]
        else:
            n_examples = 1
            n_features = X_1.shape[-1]

        #find maximum and minimum for each column, store it and perform transformation
        for i in range(n_features):
            temp_max = np.max(X_1[:, i:i+1], axis=None)
            temp_min = np.min(X_1[:, i:i+1],axis=None)
            temp_mean = np.mean(X_1[:, i:i+1],axis=None)
            parameters.append([temp_min, temp_max, temp_mean])

        self.n_examples = n_examples
        self.n_features = n_features
        self.parameters = parameters

        return parameters
    
    def transform(self, X: np.array):
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        for i in range(self.n_features):
            X_1[:,i:i+1] = (X_1[:, i:i+1] - self.parameters[i][2])/(self.parameters[i][1]-self.parameters[i][0])
        
        return X_1

    def inverse_transform(self, X: np.array):
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        for i in range(self.n_features):
            X_1[:,i:i+1] = X_1[:, i:i+1]*(self.parameters[i][1]-self.parameters[i][0])+self.parameters[i][2] 
        
        return X_1

class Zscore():

    """A scaler that performs mean normalization.
    Deals with multiple feature/multiple parameter arrays
    Expects a 3D array
    """    
    def __init__(self):
        self.parameters = list() #stores max and min for each feature prior to transformation
        self.n_examples = 0
        self.n_features = 0

    def fit(self, X: np.array):

        parameters = list()        
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        if len(X_1.shape) > 1:
            n_examples = X_1.shape[0] 
            n_features = X_1.shape[1]
        else:
            n_examples = 1
            n_features = X_1.shape[-1]

        #find maximum and minimum for each column, store it and perform transformation
        for i in range(n_features):
            temp_max = np.max(X_1[:, i:i+1], axis=None)
            temp_min = np.min(X_1[:, i:i+1],axis=None)
            temp_mean = np.mean(X_1[:, i:i+1],axis=None)
            temp_std = np.std(X_1[:, i:i+1],axis=None)
            parameters.append([temp_min, temp_max, temp_mean, temp_std])

        self.n_examples = n_examples
        self.n_features = n_features
        self.parameters = parameters

        return parameters
    
    def transform(self, X: np.array):
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        for i in range(self.n_features):
            X_1[:,i:i+1] = (X_1[:, i:i+1] - self.parameters[i][2])/(self.parameters[i][3])
        
        return X_1

    def inverse_transform(self, X: np.array):
        X_1 = X.copy()
        X_1 = X_1.astype(np.float32)

        for i in range(self.n_features):
            X_1[:,i:i+1] = X_1[:, i:i+1]*self.parameters[i][3]+self.parameters[i][2] 
        
        return X_1

    