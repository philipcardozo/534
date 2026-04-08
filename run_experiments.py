"""Run all Q3 experiments and print results."""
import numpy as np
import pandas as pd
import time
from sgb import SGTB

train = pd.read_csv('energydata/energy_train.csv')
val   = pd.read_csv('energydata/energy_val.csv')
test  = pd.read_csv('energydata/energy_test.csv')

feature_cols = [c for c in train.columns if c not in ['date', 'Appliances']]
X_train = train[feature_cols].values
y_train = train['Appliances'].values
X_val   = val[feature_cols].values
y_val   = val['Appliances'].values
X_test  = test[feature_cols].values
y_test  = test['Appliances'].values
md = 3

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Features ({len(feature_cols)}): {feature_cols}")

# -------------------------------------------------------------------------
# Q3d – q=1, vary nu and nIter
# -------------------------------------------------------------------------
nu_list   = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
iter_list = [10, 25, 50, 100, 200, 300, 500]

print("\n=== Q3d: q=1, md=3 – Validation RMSE ===")
header = f"{'nu':>6}  " + "  ".join([f"M={i:>4}" for i in iter_list])
print(header)

results_3d = {}
best3d_rmse = float('inf')
best3d_params = {}

for nu in nu_list:
    row = []
    for nIter in iter_list:
        m = SGTB(nIter=nIter, q=1.0, nu=nu, md=md)
        m.fit(X_train, y_train)
        pred_val = m.predict(X_val)
        rmse_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
        row.append(rmse_val)
        if rmse_val < best3d_rmse:
            best3d_rmse = rmse_val
            best3d_params = {'nu': nu, 'nIter': nIter}
    results_3d[nu] = row
    print(f"{nu:>6.3f}  " + "  ".join([f"{v:>8.2f}" for v in row]))

print(f"\nBest q=1: nu={best3d_params['nu']}, nIter={best3d_params['nIter']}, val_RMSE={best3d_rmse:.4f}")

# -------------------------------------------------------------------------
# Q3f – Full grid: nu x nIter x q
# -------------------------------------------------------------------------
nu_list_f   = [0.01, 0.05, 0.1, 0.2]
iter_list_f = [25, 50, 100, 200]
q_list_f    = [0.6, 0.7, 0.8, 0.9, 1.0]

print("\n=== Q3f: Grid search with subsampling ===")
print(f"{'nu':>5} {'nIter':>6} {'q':>5}  val_RMSE")

best_rmse = float('inf')
best_params = {}
all_results = []

for nu in nu_list_f:
    for nIter in iter_list_f:
        for q in q_list_f:
            m = SGTB(nIter=nIter, q=q, nu=nu, md=md)
            m.fit(X_train, y_train)
            pred_val = m.predict(X_val)
            rmse_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
            all_results.append((nu, nIter, q, rmse_val))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_params = {'nu': nu, 'nIter': nIter, 'q': q}

for (nu, nIter, q, rmse) in all_results:
    mark = " *BEST*" if (nu == best_params['nu'] and
                         nIter == best_params['nIter'] and
                         q == best_params['q']) else ""
    print(f"{nu:>5.2f} {nIter:>6} {q:>5.1f}  {rmse:>8.4f}{mark}")

print(f"\nBest: nu={best_params['nu']}, nIter={best_params['nIter']}, q={best_params['q']}")
print(f"Best val RMSE: {best_rmse:.4f}")

# -------------------------------------------------------------------------
# Q3g – Final model test error
# -------------------------------------------------------------------------
print("\n=== Q3g: Final model on test set ===")
final = SGTB(nIter=best_params['nIter'], q=best_params['q'],
             nu=best_params['nu'], md=md)
t0 = time.time()
final.fit(X_train, y_train)
final_time = time.time() - t0
pred_test  = final.predict(X_test)
rmse_test  = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))
print(f"Test RMSE (optimal stochastic): {rmse_test:.4f}  time={final_time:.2f}s")

# Best q=1 baseline
ref = SGTB(nIter=best3d_params['nIter'], q=1.0,
           nu=best3d_params['nu'], md=md)
t0 = time.time()
ref.fit(X_train, y_train)
ref_time   = time.time() - t0
pred_ref   = ref.predict(X_test)
rmse_ref   = float(np.sqrt(np.mean((y_test - pred_ref) ** 2)))
print(f"Test RMSE (q=1 best): {rmse_ref:.4f}  time={ref_time:.2f}s")

# -------------------------------------------------------------------------
# Q3h – Timing comparison stochastic vs non-stochastic
# -------------------------------------------------------------------------
print("\n=== Q3h: Timing (nIter=100, nu=0.05, md=3) ===")
print(f"{'q':>5}  {'train_time':>11}  {'val_RMSE':>10}  {'test_RMSE':>10}")
for qv in [0.6, 0.7, 0.8, 0.9, 1.0]:
    t0 = time.time()
    m_tmp = SGTB(nIter=100, q=qv, nu=0.05, md=3)
    m_tmp.fit(X_train, y_train)
    elapsed = time.time() - t0
    pv = m_tmp.predict(X_val)
    rv = float(np.sqrt(np.mean((y_val - pv) ** 2)))
    pt = m_tmp.predict(X_test)
    rt = float(np.sqrt(np.mean((y_test - pt) ** 2)))
    print(f"{qv:>5.1f}  {elapsed:>10.2f}s  {rv:>10.4f}  {rt:>10.4f}")
