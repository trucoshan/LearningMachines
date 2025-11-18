import numpy as np

class LogisticRegression:

    def __init__(self):
        self.X = None
        self.y = None
        self.n_iters = None
        self.lr = None

    def _initialize_weights(self,X_train:np.ndarray):
        self.X = X_train
        n = self.X.shape[0]
        W = np.zeros((n,1),dtype=float)
        b = 0.0
        return W, b
    
    def _sigmoid(self,z:np.ndarray):
        z = np.clip(z, -500, 500)
        a = 1.0/(1.0+np.exp(-z))
        return a
    
    def _cost(self,y_pred:np.ndarray,y_train:np.ndarray):
        eps = 1e-12
        y = y_train.reshape((1,-1)) if y_train.ndim == 1 else y_train
        m = y.shape[1]
        y_predicted = np.clip(y_pred,eps,1-eps)
        cost_arr = -(y*np.log(y_predicted) + (1-y)*np.log(1-y_predicted))
        cost = np.sum(cost_arr)/m
        return cost

    def fit(self,X_train:np.ndarray,y_train:np.ndarray,lr=0.0001,n_iters=100000):
        self.X = X_train
        self.y = y_train.reshape((1,-1))
        self.lr = lr
        self.n_iters = n_iters
        w, b = self._initialize_weights(X_train)
        tol = 1e-4
        m = self.X.shape[1]
        patience = 20
        self.X = self.X.astype(float)
        self.y = self.y.astype(float)
        best_cost = np.inf
        wait = 0
        for epoch in range(self.n_iters):
            y_pred = np.dot(w.T, self.X)
            y_pred += b
            y_pred = self._sigmoid(y_pred)
            cost = self._cost(y_pred,self.y)
            dz = (y_pred - self.y)
            dw = (1.0/m)*np.dot(self.X,dz.T)
            w = w - self.lr * dw
            db = (1.0/m)*np.sum(dz)
            b = b - self.lr * db
            y_pred2 = np.dot(w.T, self.X)
            y_pred2 += b
            y_pred2 = self._sigmoid(y_pred2)
            new_cost = self._cost(y_pred2,self.y)

            if new_cost<best_cost - tol:
                best_cost = new_cost
                wait = 0
            else:
                wait += 1

            if wait>=patience:
                print(f"Converged after {epoch} epochs\nWeight norm : {np.linalg.norm(w)} | Cost : {new_cost}")
                break

            if epoch % 1000 == 0:
                print(f"Epoch : {epoch}\nWeight norm : {np.linalg.norm(w)} | Cost : {new_cost}")

        self.w = w
        self.b = b

        return self
    
    def predict(self,X_val:np.ndarray):

        z_init = np.dot(self.w.T,X_val)
        z = z_init + self.b
        y_pred = self._sigmoid(z)

        return y_pred
    
    def evaluate(self,y_pred:np.ndarray,y_val:np.ndarray):

        y = y_val
        err = (y_pred.reshape(-1) - y.reshape(-1))
        mse = np.mean(err**2)
        rmse = np.sqrt(mse)

        my_dict = {'mse':mse,
                   'rmse':rmse}
        
        return my_dict
