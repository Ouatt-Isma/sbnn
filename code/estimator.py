import numpy as np
from scipy.special import psi, polygamma, gammaln
from scipy.optimize import root, minimize

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


class BetaEstimator:
    @staticmethod
    def estimate_moments(data):
    # def estimate_moments(data: list[float]):
        flat = [p for p in data]
        mean = np.mean(flat)
        var = np.var(flat, ddof=1)
        if var == 0:
            return mean * 100, (1 - mean) * 100
        common = mean * (1 - mean) / var - 1
        alpha = mean * common
        beta = (1 - mean) * common
        return alpha, beta

    @staticmethod
    def estimate_mle(data):
    # def estimate_mle(data: list[float]):
        flat = [p for p in data]
        alpha0, beta0 = BetaEstimator.estimate_moments(data)

        def equations(params):
            a, b = params
            dig_ab = psi(a + b)
            return [psi(a) - dig_ab - np.mean(np.log(flat)),
                    psi(b) - dig_ab - np.mean(np.log(1 - flat))]

        sol = root(equations, [alpha0, beta0], method='hybr')
        if sol.success and all(x > 0 for x in sol.x):
            return tuple(sol.x)
        return alpha0, beta0

    @staticmethod
    def estimate_kl(data):
    # def estimate_kl(data: list[float]):
        flat = [p for p in data]

        def neg_log_likelihood(params):
            a, b = params
            if a <= 0 or b <= 0:
                return np.inf
            log_pdf = (a - 1) * np.log(flat) + (b - 1) * np.log(1 - flat) - (
                gammaln(a) + gammaln(b) - gammaln(a + b))
            return -np.mean(log_pdf)

        alpha0, beta0 = BetaEstimator.estimate_moments(data)
        res = minimize(neg_log_likelihood, x0=[alpha0, beta0], method='L-BFGS-B', bounds=[(1e-6, None), (1e-6, None)])
        if res.success:
            return tuple(res.x)
        return alpha0, beta0

    def plot(data_beta, alpha_val, beta_val):
        flat_data = np.array([p for sub in data_beta for p in sub])
        # Plot histogram of data
        x = np.linspace(0, 1, 200)
        plt.hist(flat_data, bins=10, density=True, alpha=0.5, label="Empirical Data")

        # Plot estimated Beta PDF
        plt.plot(x, beta.pdf(x, alpha_val, beta_val), 'r-', lw=2, label=f"Beta MLE (α={alpha_val:.2f}, β={beta_val:.2f})")

        plt.title("Data vs Estimated Beta Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

class DirichletEstimator:
    @staticmethod
    def estimate_moments(data):
    # def estimate_moments(data: list[list[float]]):
        X = np.array(data)
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0, ddof=1)
        m = mean
        v = var
        # Crude moment estimate: use average variance
        alpha0 = m * ((m * (1 - m)).sum() / v.mean() - 1)
        return alpha0

    @staticmethod
    # def estimate_mle(data: list[list[float]]):
    def estimate_mle(data):
        X = np.array(data)
        n, k = X.shape
        m = np.mean(X, axis=0)
        alpha0 = m * 10  # init guess

        def equations(alpha):
            dig_sum = psi(np.sum(alpha))
            return psi(alpha) - dig_sum - np.mean(np.log(X), axis=0)

        sol = root(equations, alpha0, method='hybr')
        if sol.success and np.all(sol.x > 0):
            return sol.x
        return alpha0

    @staticmethod
    # def estimate_kl(data: list[list[float]]):
    def estimate_kl(data):
        X = np.array(data)
        n, k = X.shape

        def neg_log_likelihood(alpha):
            if np.any(alpha <= 0):
                return np.inf
            ll = n * (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)))
            ll += np.sum((alpha - 1) * np.sum(np.log(X), axis=0))
            return -ll / n

        m = np.mean(X, axis=0)
        alpha0 = m * 10
        res = minimize(neg_log_likelihood, x0=alpha0, method='L-BFGS-B', bounds=[(1e-6, None)] * X.shape[1])
        if res.success:
            return res.x
        return alpha0

def test():
    data_beta = [0.8, 0.9, 0.7 ,0.6, 0.75, 0.85]
    print("Beta MLE:", BetaEstimator.estimate_mle(data_beta))
    print("Beta KL:", BetaEstimator.estimate_kl(data_beta))

    data_dir = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.25, 0.25]]
    print("Dirichlet MLE:", DirichletEstimator.estimate_mle(data_dir))
    print("Dirichlet KL:", DirichletEstimator.estimate_kl(data_dir))

def test_plot():
    # Data
    data_beta = [0.8, 0.9, 0.7, 0.6, 0.75, 0.85]
    alpha_val, beta_val = BetaEstimator.estimate_moments(data_beta)
    BetaEstimator.plot(data_beta, alpha_val, beta_val)

    alpha_val, beta_val = BetaEstimator.estimate_mle(data_beta)
    BetaEstimator.plot(data_beta, alpha_val, beta_val)

    alpha_val, beta_val = BetaEstimator.estimate_kl(data_beta)
    BetaEstimator.plot(data_beta, alpha_val, beta_val)

# Example usage
if __name__ == "__main__":
    test_plot()
