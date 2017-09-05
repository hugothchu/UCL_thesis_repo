import pymc3 as pm
import numpy as np
import theano.tensor as tt
import os, pickle

approx_store = './bayesian_hierarchical_regression_store'

def hierarchicalRegressionBaseline(strength_x_train, strength_y_train, strength_x_test, strength_y_test):
    x_train_indices = strength_x_train.pop('idx').as_matrix()
    x_train_array = strength_x_train.as_matrix()
    y_train_array = strength_y_train.as_matrix()

    x_test_indices = strength_x_test.pop('idx').as_matrix()
    x_test_array = strength_x_test.as_matrix()
    y_test_array = strength_y_test.as_matrix()

    N, D = x_train_array.shape
    n_users = len(np.unique(x_train_indices))

    with pm.Model():
        # Precision Priors
        tau = pm.Gamma('tau_coeff', 1.0, 1.0)
        lambda_ = pm.Uniform('lambda_coeff', 0, 5)

        coefficients_mu_prior = pm.Normal('mu_p', 0, tau=tau * lambda_)
        coefficients = pm.Normal('coefficients', coefficients_mu_prior, tau=tau * lambda_, shape=(n_users, D))
        intercept = pm.Normal('intercept', coefficients_mu_prior, tau=tau * lambda_, shape=n_users)

        mu = tt.sum(coefficients[x_train_indices] * x_train_array, axis=1) + intercept[x_train_indices]
        pm.Normal('y', mu=mu, tau=tau * lambda_, observed=y_train_array, total_size=N)

        if os.path.isfile(approx_store):
            with open(approx_store, 'rb') as handle:
                approx = pickle.load(handle)
        else:
            approx = pm.sample(progressbar=True)
            with open(approx_store, "wb") as handle:
                pickle.dump(approx, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Assess posterior fit
        pos_coefficients = approx['coefficients'].mean(axis=0)
        pos_intercept = approx['intercept'].mean(axis=0)
        y_pred = np.sum(pos_coefficients[x_test_indices] * x_test_array, axis=1) + pos_intercept[x_test_indices]
        
        rmse = np.sqrt(np.sum(np.square(y_pred - y_test_array)))

    return rmse
