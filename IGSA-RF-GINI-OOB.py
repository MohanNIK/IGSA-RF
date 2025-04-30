import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from rfpimp import permutation_importances

# ---------------- 数据预处理 -------------------
def column(path,i):
    return path.iloc[:, i].values.tolist()

def normalization(df, min_val, max_val):
    sheet = []
    for row in df:
        row_norm = []
        for j in range(len(row)):
            val = (row[j] - min_val[j]) / (max_val[j] - min_val[j]) if j in [1,6,9] else (max_val[j] - row[j]) / (max_val[j] - min_val[j])
            row_norm.append(val)
        sheet.append(row_norm)
    return sheet

df = pd.read_excel('训练数据.xlsx')
data = df.iloc[:,1:11].values
target = df.iloc[:,11].values

min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
data_norm = np.array(normalization(data, min_vals, max_vals))
name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
data_df = pd.DataFrame(data_norm, columns=name)

# ---------------- IGSA优化RF参数 -------------------
def fitness_function(params):
    n_estimators, max_features, min_samples_split = map(int, params)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max(1, min(max_features, data_df.shape[1])),
        min_samples_split=max(2, min(min_samples_split, 10)),
        random_state=42,
        n_jobs=-1
    )
    scores = []
    for _ in range(3):
        X_train, X_test, y_train, y_test = train_test_split(data_df, target, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores.append(r2_score(y_test, pred))
    return np.mean(scores)

# IGSA 参数
dim = 3
G0 = 100
alpha = 20
num_agents = 10
max_iter = 30
bounds = [(10, 200), (1, data_df.shape[1]), (2, 10)]

# 初始化
positions = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_agents, dim))
velocities = np.zeros_like(positions)

for t in range(max_iter):
    fitness = np.array([fitness_function(p) for p in positions])
    best_idx = np.argmax(fitness)
    best_pos = positions[best_idx]
    masses = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-6)
    masses /= masses.sum()
    G = G0 * np.exp(-alpha * t / max_iter)

    for i in range(num_agents):
        force = np.zeros(dim)
        for j in range(num_agents):
            if i != j:
                r = np.linalg.norm(positions[j] - positions[i]) + 1e-6
                force += np.random.rand() * G * (masses[j]) * (positions[j] - positions[i]) / r
        acc = force
        velocities[i] = np.random.rand() * velocities[i] + acc
        positions[i] = positions[i] + velocities[i]
        # 保持在边界内
        for d in range(dim):
            positions[i][d] = np.clip(positions[i][d], bounds[d][0], bounds[d][1])

# 最优参数训练最终模型
best_params = list(map(int, best_pos))
print("Best parameters:", best_params)

rf = RandomForestRegressor(
    n_estimators=best_params[0],
    max_features=best_params[1],
    min_samples_split=best_params[2],
    random_state=42,
    n_jobs=-1,
    oob_score=True,
    bootstrap=True
)
rf.fit(data_df, target)

print('GINI:')
print(dict(zip(name, rf.feature_importances_)))

print('OOB:')
def r2(rf, X, y): return r2_score(y, rf.predict(X))
perm_imp_rfpimp = permutation_importances(rf, data_df, target, r2)
perm_imp_rfpimp.reset_index(drop=False, inplace=True)
print(perm_imp_rfpimp)

# 模型评估
X_train, X_test, y_train, y_test = train_test_split(data_df, target, test_size=0.6, random_state=1)
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

print('MSE train: %.4f, test: %.4f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.4f, test: %.4f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))
