import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture


def _shift_right(X):
    return torch.cat((torch.zeros_like(X[:, :1]), X[:, :-1]), dim=1)


def adaptive_threshold(values, seed=42):
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=seed).fit(x)
    means = gmm.means_.flatten()
    noise_idx, signal_idx = int(np.argmin(means)), int(np.argmax(means))
    lo, hi = means[noise_idx], means[signal_idx]
    grid = np.linspace(lo, hi, 1000).reshape(-1, 1)
    probs = gmm.predict_proba(grid)
    dominance = probs[:, signal_idx] > probs[:, noise_idx]
    if np.any(dominance):
        return float(grid[int(np.argmax(dominance))][0]), gmm
    return float((lo + hi) / 2), gmm


class LSTMDecoder(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.lstm = nn.GRU(num_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def step(self, X, h, causal_mask):
        X = X[:, :, np.where(causal_mask != 0)[0]]
        out, h = self.lstm(X, h)
        return self.fc(out).squeeze(-1), h

    def forward(self, X_right, z, causal_mask):
        return self.step(_shift_right(X_right), z, causal_mask)


class CausalLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, causal_graph):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_series = num_features
        self.causal_graph = np.asarray(causal_graph)

        self.encoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

        self.decoders = nn.ModuleList([
            LSTMDecoder(int(self.causal_graph[:, dim].sum()), hidden_size)
            for dim in range(num_features)
        ])

    def encode(self, X_left):
        h_0 = torch.zeros(1, X_left.shape[0], self.hidden_size, device=X_left.device)
        _, h = self.encoder(X_left, h_0)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, X_right, z):
        return torch.stack([head(X_right, z, self.causal_graph[:, d])[0]
                            for d, head in enumerate(self.decoders)], dim=2)

    def forward(self, X_left, X_right):
        mu, logvar = self.encode(X_left)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(X_right, z), mu, logvar

    @torch.no_grad()
    def rollout(self, z, future):
        x_t = torch.zeros(z.shape[1], 1, self.num_series, device=z.device)
        hidden = [z] * self.num_series
        preds = []
        for _ in range(future):
            outs = []
            for d, head in enumerate(self.decoders):
                op, hidden[d] = head.step(x_t, hidden[d], self.causal_graph[:, d])
                outs.append(op)
            x_t = torch.stack(outs, dim=-1)
            preds.append(x_t)
        return torch.cat(preds, dim=1)

    @torch.no_grad()
    def residual(self, X):
        half = X.shape[1] // 2
        mu, _ = self.encode(X[:, :half])
        return X[:, half:] - self.decode(X[:, half:], mu)

    def _strength_matrix(self):
        if any(h.lstm.weight_ih_l0.shape[1] != self.num_series for h in self.decoders):
            return None
        return torch.stack([torch.norm(h.lstm.weight_ih_l0, dim=0) for h in self.decoders])

    def get_causal_matrix(self, return_gmm=False):
        GC = self._strength_matrix()
        if GC is None:
            g = torch.tensor(self.causal_graph.T).float()
            return (g, g, None, None) if return_gmm else (g, g, None)
        raw = GC.detach().cpu().float()
        off = raw[~torch.eye(self.num_series, dtype=torch.bool)].numpy()
        thr, gmm = adaptive_threshold(off)
        binary = (raw > thr).float()
        return (raw, binary, thr, gmm) if return_gmm else (raw, binary, thr)


class ErrorCompensation(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.decoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_features)

    def _encode(self, E):
        h_0 = torch.zeros(1, E.shape[0], self.hidden_size, device=E.device)
        _, h = self.encoder(E, h_0)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, E):
        mu, logvar = self._encode(E)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        out, _ = self.decoder(_shift_right(E), self.tanh(self.fc_hidden(z)))
        return self.fc(out), mu, logvar

    @torch.no_grad()
    def forecast(self, E, future):
        mu, _ = self._encode(E)
        h = self.tanh(self.fc_hidden(mu))
        e_t = torch.zeros(E.shape[0], 1, E.shape[2], device=E.device)
        preds = []
        for _ in range(future):
            out, h = self.decoder(e_t, h)
            e_t = self.fc(out)
            preds.append(e_t)
        return torch.cat(preds, dim=1)
