import os
import random
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import torch
import torch.nn as nn
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
FS = 173.61  # fréquence d'échantillonnage de la base de Bonn (Hz)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = r"C:\Users\roman\OneDrive\Bureau\EEG\Bonn-EEG-Dataset-main"
 
# Ensembles de Bonn -> classe (0 = sain, 1 = crise d'épilepsie)
SET_LABELS = {"Z": 0, "O": 0, "N": 1, "F": 1, "S": 1}
 
TEST_SIZE = 0.2  # 20 % des segments réservés au test final
 
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def find_peaks_safe(x, height):
    return signal.find_peaks(x, height=height)

def extract_features(x):
    """Renvoie un dict {nom: valeur} de caractéristiques d'un segment EEG."""
    x = np.asarray(x, dtype=float)
    mu, sd = x.mean(), x.std()
    dx = np.diff(x)
    ddx = np.diff(dx)
 
    peaks, _ = find_peaks_safe(x, mu + sd / 2)
    peaks_neg, _ = find_peaks_safe(-x, -mu + sd / 2)  # seuil symétrique
 
    # Paramètres de Hjorth
    mobility = np.sqrt(np.var(dx) / np.var(x))
    complexity = np.sqrt(np.var(ddx) / np.var(dx)) / mobility
 
    feats = {
        "mean": mu,
        "std": sd,
        "ptp": np.ptp(x),
        "skew": stats.skew(x),
        "kurtosis": stats.kurtosis(x),
        "rms": np.sqrt(np.mean(x ** 2)),            # vrai RMS
        "line_length": np.sum(np.abs(dx)),           # variations brusques
        "zero_crossings": np.sum(np.diff(np.sign(x - mu)) != 0),
        "n_peaks": len(peaks),
        "n_peaks_neg": len(peaks_neg),
        "mean_peak_interval": np.mean(np.diff(peaks)) / FS if len(peaks) > 1 else 0.0,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }
 
    freqs, power = signal.welch(x, fs=FS, nperseg=512)
    total = power.sum()
    for name, (lo, hi) in BANDS.items():
        feats[f"ratio_{name}"] = power[(freqs >= lo) & (freqs < hi)].sum() / total
    return feats
 
FILE_RE = re.compile(r"^([ZONFS])(\d{3})\.txt$", re.IGNORECASE)

def load_bonn():
    """Parcourt BASE_DIR, charge chaque segment (Z001.txt, S042.txt...) dont
    l'ensemble figure dans SET_LABELS, et renvoie (features, labels, noms)."""
    rows, labels, names, seen = [], [], [], set()
    for root, _, files in os.walk(BASE_DIR):
        for name in sorted(files):
            m = FILE_RE.match(name)
            if not m or name.upper() in seen:
                continue
            letter = m.group(1).upper()
            if letter not in SET_LABELS:
                continue
            seen.add(name.upper())
            eeg = np.loadtxt(os.path.join(root, name))
            rows.append(extract_features(eeg))
            labels.append(SET_LABELS[letter])
            names.append(name)
    if not rows:
        raise FileNotFoundError(
            f"Aucun fichier de type Z001.txt / O001.txt / S001.txt trouvé dans {BASE_DIR}")
    return pd.DataFrame(rows), np.array(labels), np.array(names)

class EEGNet(nn.Module):
    """MLP plus compact que l'original (7 couches avec goulot à 4 neurones
    = risque de neurones morts et sur-apprentissage sur peu de données)."""
 
    def __init__(self, n_in, hidden=(32, 16), p=0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
 
def train_model(X, y, epochs=60, lr=1e-3, batch_size=16, seed=SEED):
    torch.manual_seed(seed)
    model = EEGNet(X.shape[1]).to(DEVICE)  # modèle NEUF à chaque appel
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
 
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step(total / len(loader.dataset))
    return model
 
 
def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return out.argmax(1).cpu().numpy()
 
 
def accuracy(y_true, y_pred):
    return 100.0 * np.mean(y_true == y_pred)

def cross_validate(X, y, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        # le scaler est ajusté sur le fold d'entraînement uniquement
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
 
        # 1) vraies étiquettes
        model = train_model(Xtr, y[tr], seed=SEED + fold)
        acc_true = accuracy(y[va], predict(model, Xva))
 
        # 2) étiquettes permutées -> NOUVEAU modèle (vraie référence "hasard")
        y_perm = np.random.RandomState(fold).permutation(y[tr])
        model_perm = train_model(Xtr, y_perm, seed=SEED + fold)
        acc_perm = accuracy(y[va], predict(model_perm, Xva))
 
        # 3) baseline classique
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
        acc_rf = accuracy(y[va], rf.fit(Xtr, y[tr]).predict(Xva))
 
        print(f"Fold {fold}/{k} | réseau {acc_true:.1f}% | "
              f"permuté {acc_perm:.1f}% | random forest {acc_rf:.1f}%")
        rows += [("réseau", fold, acc_true), ("permuté", fold, acc_perm),
                 ("random forest", fold, acc_rf)]
    return pd.DataFrame(rows, columns=["type", "fold", "score"])
 
 
def plot_cv(cv, y):
    chance = 100.0 * max(np.mean(y), 1 - np.mean(y))  # classe majoritaire
    plt.figure(figsize=(9, 6))
    sb.violinplot(data=cv, x="type", y="score", inner=None, cut=1, width=0.5,
                  hue="type", palette="pastel", legend=False)
    sb.swarmplot(data=cv, x="type", y="score", color="black", alpha=0.8)
    plt.axhline(chance, linestyle="--", color="red",
                label=f"Classe majoritaire ({chance:.0f}%)")
    plt.ylim(0, 100)
    plt.ylabel("Précision (%)")
    plt.title("Validation croisée : étiquettes réelles vs permutées")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    set_seed()
    X_df, y, names = load_bonn()
    feature_names = list(X_df.columns)
    print(f"{len(y)} segments chargés : {np.sum(y == 0)} sains, {np.sum(y == 1)} crises")
 
    # Séparation train/test AU NIVEAU DES SEGMENTS, stratifiée (mêmes proportions
    # sain/crise dans les deux jeux). Le test n'est utilisé qu'à la toute fin.
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Train : {X_train.shape}, Test : {X_test.shape}, "
          f"% malades train : {100 * y_train.mean():.0f}")
 
    # Validation croisée (sur le train uniquement)
    cv = cross_validate(X_train, y_train)
    print(cv.groupby("type")["score"].agg(["mean", "std"]))
    plot_cv(cv, y_train)
 
    # Modèle final : entraînement sur tout le train, évaluation sur le test
    scaler = StandardScaler().fit(X_train)
    model = train_model(scaler.transform(X_train), y_train)
    y_pred = predict(model, scaler.transform(X_test))
 
    print(f"\nPrécision du test : {accuracy(y_test, y_pred):.2f}%")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Sensibilité (malades détectés) : {100 * tp / (tp + fn):.1f}%  "
          f"| faux négatifs : {fn}")
    print(f"Spécificité (sains reconnus)   : {100 * tn / (tn + fp):.1f}%  "
          f"| faux positifs : {fp}")
    print(classification_report(y_test, y_pred, target_names=["sain", "malade"]))
 
    # Sauvegarde : le modèle est sauvegardé AVANT tout entraînement "permuté"
    joblib.dump({"scaler": scaler, "features": feature_names}, "scaler.pkl")
    torch.save(model.state_dict(), "eeg_model.pth")
 
 
if __name__ == "__main__":
    # Lancement : python eeg_bonn_classification.py
    main()
