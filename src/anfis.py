import numpy as np
from sklearn.cluster import KMeans

LOG_SIGMA_MIN, LOG_SIGMA_MAX = np.log(0.15), np.log(8.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _bce(z, y):
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


class ANFIS:
    mf, order, bell_b = "gauss", 1, 2.0

    def __init__(self, n_rules=4, epochs=200, lr=0.01, batch_size=256, l2=1e-4,
                 patience=20, seed=0, mf="gauss", order=1):
        assert mf in ("gauss", "bell") and order in (0, 1)
        self.mf, self.order = mf, order
        self.n_rules = n_rules
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.l2 = l2
        self.patience = patience
        self.seed = seed

    def log_membership(self, X):
        u2 = ((X[:, None, :] - self.c[None, :, :]) / np.exp(self.log_s)[None, :, :]) ** 2
        if self.mf == "bell":
            return -np.log1p(u2 ** self.bell_b)
        return -0.5 * u2

    def _dlogmu_du2(self, u2):
        if self.mf == "bell":
            p = u2 ** self.bell_b
            return -self.bell_b * np.where(u2 > 0, p / np.maximum(u2, 1e-300), 0.0) / (1.0 + p)
        return np.full_like(u2, -0.5)

    def forward(self, X):
        log_w = self.log_membership(X).mean(axis=2)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        wbar = w / w.sum(axis=1, keepdims=True)
        f = X @ self.a.T + self.b
        return (wbar * f).sum(axis=1), wbar, f

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        for i in range(0, len(X), 20000):
            out[i:i + 20000] = self.forward(X[i:i + 20000])[0]
        return out

    def predict_proba(self, X):
        return _sigmoid(self.decision_function(X))

    def loss_and_grads(self, X, y):
        n, d = X.shape
        s2 = np.exp(2 * self.log_s)
        diff = X[:, None, :] - self.c[None, :, :]
        z, wbar, f = self.forward(X)
        g = (_sigmoid(z) - y) / n
        gw = g[:, None] * wbar
        g_logw = gw * (f - z[:, None])
        u2 = diff ** 2 / s2
        k = self._dlogmu_du2(u2) / d
        grads = {
            "a": (gw.T @ X + self.l2 * self.a) * self.order,
            "b": gw.sum(axis=0),
            "c": np.einsum("nr,nrj->rj", g_logw, k * (-2.0 * diff / s2)),
            "log_s": np.einsum("nr,nrj->rj", g_logw, k * (-2.0 * u2)),
        }
        loss = _bce(z, y) + 0.5 * self.l2 * np.sum(self.a ** 2)
        return loss, grads

    def _init_params(self, X, y, rng):
        n, d = X.shape
        sub = rng.choice(n, size=min(n, 20000), replace=False)
        Xs, ys = X[sub], y[sub]
        km = KMeans(n_clusters=self.n_rules, n_init=5, random_state=self.seed).fit(Xs)
        self.c = km.cluster_centers_.copy()
        sig = np.ones((self.n_rules, d))
        for r in range(self.n_rules):
            members = Xs[km.labels_ == r]
            if len(members) > 1:
                sig[r] = members.std(axis=0)
        self.log_s = np.log(np.clip(sig, 0.5, 3.0))
        self.a = np.zeros((self.n_rules, d))
        self.b = np.zeros(self.n_rules)
        wbar = self.forward(Xs)[1]
        target = 2.0 * (2.0 * ys - 1.0)
        if self.order == 0:
            self.b = np.linalg.solve(wbar.T @ wbar + 1.0 * np.eye(self.n_rules), wbar.T @ target)
            return
        Phi = np.concatenate([(wbar[:, :, None] * Xs[:, None, :]).reshape(len(Xs), -1), wbar], axis=1)
        theta = np.linalg.solve(Phi.T @ Phi + 1.0 * np.eye(Phi.shape[1]), Phi.T @ target)
        self.a = theta[:self.n_rules * d].reshape(self.n_rules, d)
        self.b = theta[self.n_rules * d:]

    def fit(self, X, y, X_val, y_val):
        X, y = np.asarray(X, float), np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        self._init_params(X, y, rng)
        names = ["a", "b", "c", "log_s"]
        m = {k: np.zeros_like(getattr(self, k)) for k in names}
        v = {k: np.zeros_like(getattr(self, k)) for k in names}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        best = (_bce(self.decision_function(X_val), y_val), self.get_params())
        self.history, stale = [], 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), self.batch_size):
                idx = order[i:i + self.batch_size]
                _, grads = self.loss_and_grads(X[idx], y[idx])
                t += 1
                for k in names:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    step = self.lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
                    setattr(self, k, getattr(self, k) - step)
                self.log_s = np.clip(self.log_s, LOG_SIGMA_MIN, LOG_SIGMA_MAX)
            val_loss = _bce(self.decision_function(X_val), y_val)
            self.history.append(val_loss)
            if val_loss < best[0] - 1e-5:
                best, stale = (val_loss, self.get_params()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.set_params(best[1])
        self.epochs_run = len(self.history)
        return self

    def get_params(self):
        return {k: getattr(self, k).copy() for k in ["a", "b", "c", "log_s"]}

    def set_params(self, p):
        for k, val in p.items():
            setattr(self, k, val.copy())

    @property
    def sigma(self):
        return np.exp(self.log_s)

    def describe_rules(self, split, max_terms=4, binary=None):
        rules = []
        for r in range(self.n_rules):
            order = np.argsort(self.sigma[r])[:max_terms]
            terms = []
            for j in order:
                centre = self.c[r, j] * split.std[j] + split.mean[j]
                zc = self.c[r, j]
                if binary is not None and binary[j]:
                    label = "NO" if centre < 1 / 3 else "YES" if centre > 2 / 3 else "MIXED"
                    terms.append(f"{split.features[j]} is {label}")
                else:
                    label = "LOW" if zc < -0.5 else "HIGH" if zc > 0.5 else "MEDIUM"
                    terms.append(f"{split.features[j]} is {label} (~{centre:.3g})")
            rules.append({
                "rule": r + 1,
                "if": " AND ".join(terms),
                "then_bias": float(self.b[r]),
                "then_top_coefficients": ", ".join(
                    f"{self.a[r, j]:+.2f}*{split.features[j]}"
                    for j in np.argsort(-np.abs(self.a[r]))[:max_terms]),
            })
        return rules
