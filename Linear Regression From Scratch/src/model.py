import numpy as np

class LinearRegression:

    def __init__(self,X_train:np.ndarray,y_train:np.ndarray,lr=0.01,n_iters=100000):

        self.X = X_train
        self.y = y_train
        self.lr = lr
        self.n_iters = n_iters

    def _initialize_weights(self):

        w = np.zeros(self.X.shape[1],dtype=float)
        return w
    
    def _cost(self,y_hat):

        cost = (1/(2*self.X.shape[0])) * np.sum(np.square(y_hat - self.y))
        return cost
    
    def _gd(self):

        w = self._initialize_weights()

        for i in range(self.n_iters):
            y_hat = self.X @ w
            grad = (1/self.X.shape[0]) * self.X.T @ (y_hat - self.y)
            w = w - self.lr * grad
            y_hat = self.X @ w
            cost = self._cost(y_hat)
            if i%10000 == 0:
                print(f"Iteration: {i}\nWeights: {np.linalg.norm(w)} | Cost: {cost}")
            if cost < 1e-5:
                print(f"Converged after {i} iteration")
                break
        
        self.w = w
        return w
    
    def _sgd(self):

        w = self._initialize_weights()
        patience = 10
        tol = 1e-4
        y_hat = np.zeros(self.y.shape,dtype=float)
        for epoch in range(self.n_iters):
            old_cost = self._cost(y_hat)
            p = np.random.permutation(self.X.shape[0])
            x_shuffled = self.X[p]
            y_shuffled = self.y[p]
            y_hat_shuffled = y_hat[p]
            for i in range(self.X.shape[0]):
                y_hat_shuffled[i] = x_shuffled[i] @ w
                error = y_hat_shuffled[i] - y_shuffled[i]
                cost = 0.5 * (error)**2
                grad = error * x_shuffled[i]
                w = w - self.lr * grad
            new_cost = self._cost(y_hat_shuffled)
            
            if epoch%1000==0:
                print(f"Iteration: {epoch}\nWeight Norm {np.linalg.norm(w)} | Cost: {new_cost}")

            if old_cost - new_cost <= tol:
                patience -= 1
            else:
                patience = 10

            if patience == 0:
                print(f"Converged after {epoch} iterations.")
                break

        self.w = w

        return w
    
    def fit(self,optimization='sgd'):

        if optimization.lower()=='sgd':
            self.model = self._sgd()
            return self
        elif optimization.lower()=='gd':
            self.model = self._gd()
            return self
        else:
            raise ValueError("Selection for optimization not recognized\nPlease enter one of 'gd' or 'sgd'.")
        
    def predict(self,X_val:np.ndarray):

        y_pred = X_val @ self.w

        return y_pred
    
    def evaluate(self,y_val:np.ndarray,y_pred:np.ndarray,metric='all'):

        err = y_pred - y_val
        err_mse = np.mean(np.square(err))
        err_rmse = np.sqrt(err_mse)
        if metric=='mse':
            return err_mse
        elif metric=='rmse':
            return err_rmse
        elif metric=='all':
            return {'mse':err_mse,'rmse':err_rmse}
        else:
            raise ValueError("Specified type of evaluation metric does not match current ability.\nPlease try either 'mse' or 'rmse'.")