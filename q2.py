import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_loan(path='loan_default.csv'):
    """
    Load and preprocess loan_default.csv.
    Returns X (numpy array), y (numpy array), feature_names.
    """
    df = pd.read_csv(path)

    # Drop id
    df = df.drop(columns=['id'])

    # Target
    y = df['class'].values
    df = df.drop(columns=['class'])

    # Convert 'term' to numeric (strip ' months')
    df['term'] = df['term'].str.strip().str.replace(' months', '').astype(float)

    # 'earliest_cr_line': extract year
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y',
                                             errors='coerce').dt.year
    df['earliest_cr_line'] = df['earliest_cr_line'].fillna(df['earliest_cr_line'].median())

    # 'emp_length': map to numeric
    emp_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10, 'n/a': np.nan
    }
    df['emp_length'] = df['emp_length'].map(emp_map)
    df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())

    # One-hot encode remaining categoricals
    cat_cols = ['grade', 'home_ownership', 'verification_status', 'purpose']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Fill remaining NaNs with median
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    feature_names = df.columns.tolist()
    X = df.values.astype(float)
    return X, y, feature_names


def partition_loan(X, y, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    Stratified 70/15/15 split.
    """
    from sklearn.model_selection import train_test_split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_frac / (1 - test_frac),
        stratify=y_tv, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Question 2c – tune_nn
# ---------------------------------------------------------------------------

def tune_nn(x, y, hiddenparams, actparams, alphaparams):
    """
    Grid-search a MLPClassifier over hidden layer sizes, activation functions,
    and L2 regularization parameters.  Uses 5-fold CV with AUC scoring.

    Parameters
    ----------
    x            : 2-D numpy array of features (already scaled)
    y            : 1-D numpy array of binary labels
    hiddenparams : list of tuples, e.g. [(100,), (100, 50)]
    actparams    : list of str, e.g. ['logistic', 'tanh', 'relu']
    alphaparams  : list of float, e.g. [0.0001, 0.001, 0.01]

    Returns
    -------
    dict with keys 'best-hidden', 'best-activation', 'best-alpha'
    (plus 'best-auc' and 'cv_results' for analysis)
    """
    param_grid = {
        'hidden_layer_sizes': hiddenparams,
        'activation':         actparams,
        'alpha':              alphaparams,
    }
    mlp = MLPClassifier(max_iter=500, random_state=42, early_stopping=True,
                        validation_fraction=0.1)
    gs = GridSearchCV(mlp, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    gs.fit(x, y)
    return {
        'best-hidden':     gs.best_params_['hidden_layer_sizes'],
        'best-activation': gs.best_params_['activation'],
        'best-alpha':      gs.best_params_['alpha'],
        'best-auc':        gs.best_score_,
        'cv_results':      gs.cv_results_,
    }


# ---------------------------------------------------------------------------
# Decision-tree tuner (kept from HW3)
# ---------------------------------------------------------------------------

def tune_dt(x, y, dparams, lsparams):
    param_grid = {'max_depth': dparams, 'min_samples_leaf': lsparams}
    dt = DecisionTreeClassifier(random_state=42)
    clf = GridSearchCV(dt, param_grid, scoring='roc_auc', cv=5,
                       return_train_score=False)
    clf.fit(x, y)
    return {
        'best-depth':        clf.best_params_['max_depth'],
        'best-leaf-samples': clf.best_params_['min_samples_leaf'],
        'best-auc':          clf.best_score_,
        'grid_results':      clf.cv_results_,
    }


# ---------------------------------------------------------------------------
# Main – runs all HW4 Q2 experiments and prints results
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------------ #
    # Load & preprocess
    # ------------------------------------------------------------------ #
    X, y, feat_names = preprocess_loan('loan_default.csv')
    X_train, y_train, X_val, y_val, X_test, y_test = partition_loan(X, y)

    # Scale for neural network (fit scaler on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Combined train+val for final DT training and NN CV
    X_tv   = np.vstack((X_train, X_val))
    y_tv   = np.concatenate((y_train, y_val))
    X_tv_s = np.vstack((X_train_s, X_val_s))
    y_tv_s = y_tv

    print("Dataset sizes – Train:", len(y_train),
          "| Val:", len(y_val), "| Test:", len(y_test))

    # ------------------------------------------------------------------ #
    # 2c/2d – Tune neural network
    # ------------------------------------------------------------------ #
    hiddenparams  = [(50,), (100,), (200,), (100, 50), (100, 25), (200, 100)]
    actparams     = ['logistic', 'tanh', 'relu']
    alphaparams   = [1e-4, 1e-3, 1e-2, 0.1]

    print("\nRunning NN grid search …")
    t0 = time.time()
    nn_res = tune_nn(X_tv_s, y_tv_s, hiddenparams, actparams, alphaparams)
    nn_tune_time = time.time() - t0
    print(f"NN tuning time: {nn_tune_time:.1f}s")
    print("Best hidden:", nn_res['best-hidden'])
    print("Best activation:", nn_res['best-activation'])
    print("Best alpha:", nn_res['best-alpha'])
    print("Best CV AUC:", nn_res['best-auc'])

    # ------------------------------------------------------------------ #
    # 2e – Evaluate on test set
    # ------------------------------------------------------------------ #

    # Retrain best NN on full train+val
    best_nn = MLPClassifier(
        hidden_layer_sizes=nn_res['best-hidden'],
        activation=nn_res['best-activation'],
        alpha=nn_res['best-alpha'],
        max_iter=500, random_state=42,
    )
    t0 = time.time()
    best_nn.fit(X_tv_s, y_tv_s)
    nn_train_time = time.time() - t0

    prob_nn = best_nn.predict_proba(X_test_s)[:, 1]
    pred_nn = best_nn.predict(X_test_s)
    auc_nn  = roc_auc_score(y_test, prob_nn)
    f1_nn   = f1_score(y_test, pred_nn)
    f2_nn   = fbeta_score(y_test, pred_nn, beta=2)

    # Tune & retrain decision tree on train+val (no scaling needed)
    dparams  = [2, 3, 4, 5, 7, 10, 15]
    lsparams = [1, 2, 5, 10, 20, 50, 100]
    dt_res = tune_dt(X_tv, y_tv, dparams, lsparams)
    best_dt = DecisionTreeClassifier(
        max_depth=dt_res['best-depth'],
        min_samples_leaf=dt_res['best-leaf-samples'],
        random_state=42,
    )
    t0 = time.time()
    best_dt.fit(X_tv, y_tv)
    dt_train_time = time.time() - t0

    prob_dt = best_dt.predict_proba(X_test)[:, 1]
    pred_dt = best_dt.predict(X_test)
    auc_dt  = roc_auc_score(y_test, prob_dt)
    f1_dt   = f1_score(y_test, pred_dt)
    f2_dt   = fbeta_score(y_test, pred_dt, beta=2)

    print("\n=== Test Results ===")
    print(f"{'Model':<15} {'AUC':>8} {'F1':>8} {'F2':>8} {'Train Time':>12}")
    print("-" * 55)
    print(f"{'Neural Net':<15} {auc_nn:>8.4f} {f1_nn:>8.4f} {f2_nn:>8.4f} {nn_train_time:>11.2f}s")
    print(f"{'Decision Tree':<15} {auc_dt:>8.4f} {f1_dt:>8.4f} {f2_dt:>8.4f} {dt_train_time:>11.2f}s")

    print("\nDT best params:", dt_res['best-depth'], dt_res['best-leaf-samples'])
    print("NN tuning time:", nn_tune_time, "s  | DT tuning time: included in GridSearchCV")
