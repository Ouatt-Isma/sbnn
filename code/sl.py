from typing import Dict
from estimator import BetaEstimator, DirichletEstimator
beta_estimate = BetaEstimator.estimate_mle
dirichlet_estimate = DirichletEstimator.estimate_kl

class BinomialOpinion:
    def __init__(self, belief: float, disbelief: float, uncertainty: float, base_rate: float = 0.5):
        if round(belief + disbelief + uncertainty, 6) != 1:
            raise ValueError("Belief, disbelief, and uncertainty must sum to 1.")
        self.b = belief
        self.d = disbelief
        self.u = uncertainty
        self.a = base_rate

    def expectation(self) -> float:
        return self.b + self.a * self.u

    def __repr__(self):
        return (f"BinomialOpinion(b={self.b:.3f}, d={self.d:.3f}, u={self.u:.3f}, a={self.a}, "
                f"E={self.expectation():.3f})")

    @staticmethod
    def from_beta(alpha: float, beta: float, base_rate: float = 0.5):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and Beta must be > 0 for Beta distribution.")
        r = alpha - 1
        s = beta - 1
        K = r + s
        b = r / (K + 2)
        d = s / (K + 2)
        u = 2 / (K + 2)
        return BinomialOpinion(b, d, u, base_rate)

    @staticmethod
    def from_list_prob(data_beta, base_rate: float = 0.5):
        if len(data_beta) <= 0:
            raise ValueError("len list data must be > 0 for Beta distribution.")
        alpha, beta = beta_estimate(data_beta)
        return BinomialOpinion.from_beta(alpha, beta, base_rate)
    


class MultinomialOpinion:
    def __init__(self, belief: Dict[str, float], uncertainty: float, base_rate: Dict[str, float]):
        if abs(sum(belief.values()) + uncertainty - 1) > 1e-6:
            raise ValueError("Sum of beliefs + uncertainty must equal 1.")
        if abs(sum(base_rate.values()) - 1) > 1e-6:
            raise ValueError("Base rates must sum to 1.")
        if set(belief.keys()) != set(base_rate.keys()):
            raise ValueError("Belief keys and base rate keys must match.")
        self.belief = belief
        self.u = uncertainty
        self.base_rate = base_rate
    
    def get_best_belief_uncertainty(self):
        max_key, max_value = max(self.belief.items(), key=lambda x: x[1])
        return (max_key, max_value, self.u)
    
    def expectation(self) -> Dict[str, float]:
        return {x: self.belief[x] + self.base_rate[x] * self.u for x in self.belief}

    def __repr__(self):
        exp = self.expectation()
        exp_str = ", ".join([f"{k}: {v:.3f}" for k, v in exp.items()])
        return (f"MultinomialOpinion(belief={self.belief}, u={self.u:.3f}, base_rate={self.base_rate}, "
                f"E={{ {exp_str} }})")

    @staticmethod
    def from_dirichlet(alpha: Dict[str, float], base_rate: Dict[str, float] = None):
        if any(v <= 0 for v in alpha.values()):
            raise ValueError("All alpha parameters must be > 0 for Dirichlet.")
        outcomes = list(alpha.keys())
        # r = {x: alpha[x] - 1 for x in outcomes}
        r = {x: alpha[x] for x in outcomes}
        K = sum(r.values())
        W = len(outcomes)
        belief = {x: r[x] / (K + W) for x in outcomes}
        u = W / (K + W)
        if base_rate is None:
            base_rate = {x: 1.0 / W for x in outcomes}
        return MultinomialOpinion(belief, u, base_rate)
    
    def from_list_prob(data_dirichlet, base_rate: Dict[str, float] = None):
        # data_dirichlet list of list of prob
        alphas = dirichlet_estimate(data_dirichlet)

        alpha = {}
        for i in range(len(alphas)):
            alpha[i] = alphas[i]
        return MultinomialOpinion.from_dirichlet(alpha)
       


# Example usage
if __name__ == "__main__":
    # Beta -> Binomial Opinion
    beta_op = BinomialOpinion.from_beta(alpha=3, beta=5)
    print(beta_op)

    # Dirichlet -> Multinomial Opinion
    alpha = {"Red": 2, "Green": 3, "Blue": 4}
    multi_op = MultinomialOpinion.from_dirichlet(alpha)
    print(multi_op)
