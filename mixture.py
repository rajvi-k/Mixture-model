import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import os

#np.random.seed(1)
#torch.manual_seed(1)


class mixture_model(torch.nn.Module):
    """A Laplace mixture model trained using marginal likelihood maximization

    Arguments:
        K: number of mixture components

    """

    def __init__(self, K=5):
         
        super(mixture_model, self).__init__()
        self.K = K

    def get_params(self):
        """Get the model parameters.

        Returns:
            a list containing the following parameter values:

            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))

        """

        # pass
        mu = self.mu
        b = np.exp(self.b)
        pi = np.exp(self.pi) / np.sum(np.exp(self.pi))
        mu = self.tmu.detach().numpy()
        b = np.exp(self.tb.detach().numpy())
        pi = np.exp(self.tpi.detach().numpy()) / np.sum(np.exp(self.tpi.detach().numpy()))
        return [mu, b, pi]

    def set_params(self, mu, b, pi):
        """Set the model parameters.

        Arguments:
            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))
        """

    
        self.mu = mu
        self.b = np.log(b)
        self.pi = np.log(pi)
        self.s = np.sum(np.exp(pi))

        self.tmu = torch.nn.Parameter(torch.from_numpy(self.mu), requires_grad=True)
        self.tpi = torch.nn.Parameter(torch.from_numpy(self.pi), requires_grad=True)
        self.tb = torch.nn.Parameter(torch.from_numpy(self.b), requires_grad=True)

    def forward(self, X):
        # pass
        K = self.tmu.shape[1]
        N = (X.shape[0])
        D = (X.shape[1])
        mask = torch.isnan(X)
    

        tpi = torch.exp(self.tpi)/torch.sum(torch.exp(self.tpi))
        tb = torch.exp(self.tb)
        likelihood = torch.zeros([N])
      
        nans = X != X
        X[nans] = 0
      
        for z in range(K):
            PZ = torch.ones([N])

            for d in range(D):
                MU = (self.tmu[d, z])
                B = (tb[d, z])

                p = torch.exp(-1 * torch.abs(X[:, d] - MU) / B) / (2 * B)
                
                p[nans[:, d]] = 1
                PZ = PZ.clone() * p

            likelihood += PZ * (tpi[z])
      
        return torch.sum(torch.log(likelihood))

    def marginal_likelihood(self, X):
        """log marginal likelihood function.
           Computed using the current values of the parameters
           as set via fit or set_params.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's

        Returns:
            Marginal likelihood of observed data
        """
        
        [mu,b1,pi1]=self.get_params()
        K = self.mu.shape[1]
        N = X.shape[0]
        D = X.shape[1]
        likelihood = np.zeros(N)
        for z in range(K):
            p = np.ones(N)
            PI = pi1[z]
            for d in range(D):
                MU = self.mu[d, z]
                B = b1[d, z]
                inter = np.exp(-1 * np.abs(X[:, d] - MU) / B) / (2 * B)
                
                nans = np.isnan(inter)
                inter[nans] = 1

                p = p * inter
            likelihood += p * PI
 
        log_likelihood = np.sum(np.log(likelihood))
        return log_likelihood

    def predict_proba(self, X):
        """Predict the probability over clusters P(Z=z|X=x) for each example in X.
           Use the currently stored parameter values.

        Arguments:
            X (numpy ndarray, shape = (N,D)):
                Input matrix where each row is a feature vector.

        Returns:
            PZ (numpy ndarray, shape = (N,K)):
                Probability distribution over classes for each
                data case in the data set.
        """

        [mu, b, pi] = self.get_params()

        K = mu.shape[1]
        N = X.shape[0]
        D = X.shape[1]
        PZ = np.ones((N, K))

        likelihood = np.zeros(N)
        for z in range(K):
            # PZ[:,z] = np.ones(N)
            PI = pi[z]
            for d in range(D):
                MU = mu[d, z]
                B = b[d, z]
                inter = np.exp(-1 * np.abs(X[:, d] - MU) / B) / (2 * B)
                nans = np.isnan(inter)
                inter[nans] = 1
                PZ[:, z] = PZ[:, z] * inter

            PZ[:, z] = PZ[:, z] * PI
        likelihood = np.sum(PZ, axis=1)
        PZ = PZ / likelihood[:, np.newaxis]
       
        return PZ

    def impute(self, X):
        """Mean imputation of missing values in the input data matrix X.
           Ipmute based on the currently stored parameter values.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's

        Returns:
            XI (numpy ndarray, shape = (N, D)):
                The input data matrix where the missing values on
                each row (indicated by np.nans) have been imputed
                using their conditional means given the observed
                values on each row.
        """

        # pass
        [mu, b, pi] = self.get_params()

        K = mu.shape[1]
        N = X.shape[0]
        D = X.shape[1]
        # print(X.shape)
        PZ = self.predict_proba(X)
        s = np.sum(PZ, axis=1)
        for i in range(N):
            for d in range(D):
                if np.isnan(X[i, d]):
                    X[i, d] = 0
                    for k in range(K):
                        X[i, d] += PZ[i, k] * mu[d, k]
        # print(X)
        return X

    def fit(self, X, mu_init=None, b_init=None, pi_init=None, step=0.1, epochs=20):
        """Train the model according to the given training data
           by directly maximizing the marginal likelihood of
           the observed data. If initial parameters are specified, use those
           to initialize the model. Otherwise, use a random initialization.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's
            mu_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density mean paramaeters for each mixture component
                to use for initialization
            b_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density scale parameters for each mixture component
                to use for initialization
            pi_init (None or numpy ndarray, shape = (K,)):
                Mixture proportions to use for initialization
            step (float):
                Initial step size to use during training
            epochs (int): number of epochs for training
        """

        # pass
        if mu_init is None:
            mu_init = np.random.rand(X.shape[1], self.K)
        if pi_init is None:
            pi_init = np.random.rand(self.K)
            pi_init = pi_init / np.sum(pi_init)
        if b_init is None:
            b_init = np.random.rand(X.shape[1], self.K)
            b_init = np.exp(b_init)
        self.set_params(mu_init, b_init, pi_init)
        
        X_tensor = torch.from_numpy(X).float()
        y_class_tensor = torch.from_numpy(np.zeros((X.shape[0],)))
        my_dataset = TensorDataset(X_tensor, y_class_tensor)
        my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=64)

        learning_rate = step
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for e in range(20):
            # self.train()
            for x, y in my_dataloader:
                # print(x.type)
                f1 = self(x)  # N*K
                print(f1)
                s = (f1)

                loss = -1 * s
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
def main():
    data = np.load("data/data.npz")
    xtr1 = data["xtr1"]
    xtr2 = data["xtr2"]
    xte1 = data["xte1"]
    xte2 = data["xte2"]

    xte2[np.isnan(xte2)] = xtr2[np.isnan(xte2)]
    # xtr2[np.isnan(xtr2)]=xte2[np.isnan(xtr2)]
    sum_of_nans = (np.sum(np.isnan(xtr2)))
    
    mm = mixture_model(K=5)

    mm.fit(xtr2)
    mm.predict_proba(xtr2)


if __name__ == '__main__':
    main()