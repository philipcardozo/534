from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


def tune_sgtb(x, y, lst_nIter, lst_nu, lst_q, md):
    """
    Tune SGTB hyperparameters via GridSearchCV (5-fold CV).
    Returns dict with keys best-nIter, best-nu, best-q (and extras).
    """
    param_grid = {
        'nIter': lst_nIter,
        'nu':    lst_nu,
        'q':     lst_q,
        'md':    [md],
    }
    gs = GridSearchCV(
        SGTB(),
        param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1,
    )
    gs.fit(x, y)
    return {
        'best-nIter':  gs.best_params_['nIter'],
        'best-nu':     gs.best_params_['nu'],
        'best-q':      gs.best_params_['q'],
        'best-rmse':   -gs.best_score_,
        'cv_results':  gs.cv_results_,
    }


def compute_residual(y_true, y_pred):
    """
    Negative gradient of MSE loss: r_i = y_i - f(x_i).
    """
    return y_true - y_pred


class SGTB(RegressorMixin):
    def __init__(self, nIter=1, q=1, nu=0.1, md=3):
        self.nIter = nIter
        self.q = q
        self.nu = nu
        self.md = md
        self.train_dict = {}

    def get_params(self, deep=True):
        return {"nIter": self.nIter,
                "q":     self.q,
                "nu":    self.nu,
                "md":    self.md}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x, y):
        self.train_dict = {}
        self.trees_ = []

        # Iteration 0: predict mean of y
        self.f0_ = np.mean(y)
        f_pred = np.full(len(y), self.f0_)
        self.train_dict[0] = float(np.sqrt(np.mean((y - f_pred) ** 2)))

        n_samples = len(y)
        rng = np.random.RandomState(42)

        for m in range(1, self.nIter + 1):
            # Subsample at rate q
            n_sub = max(1, int(self.q * n_samples))
            idx = rng.choice(n_samples, size=n_sub, replace=False)
            x_sub = x[idx]
            y_sub = y[idx]
            f_sub = f_pred[idx]

            # Gradient residual on subsample
            residuals = compute_residual(y_sub, f_sub)

            # Fit regression tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.md, random_state=42)
            tree.fit(x_sub, residuals)
            self.trees_.append(tree)

            # Update predictions on full training set
            f_pred = f_pred + self.nu * tree.predict(x)
            self.train_dict[m] = float(np.sqrt(np.mean((y - f_pred) ** 2)))

        return self

    def predict(self, x):
        pred = np.full(len(x), self.f0_)
        for tree in self.trees_:
            pred = pred + self.nu * tree.predict(x)
        return pred
