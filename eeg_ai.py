# -*- coding: utf-8 -*-
"""Pont entre l'interface et le modèle : chargement, prédiction, mise à jour prudente.
 
Les caractéristiques (extract_features) et le réseau (EEGNet) sont importés du script
d'entraînement : une seule définition, donc aucun risque que l'application et
l'entraînement divergent.
"""
import copy
import json
import os
import shutil
from datetime import datetime
 
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
 
import eeg_bonn_classification as base
 
DEVICE = base.DEVICE
BACKUP_DIR = os.path.join(base.OUT_DIR, "model_backups")
EXPECTED_LEN = 4097   # longueur d'un segment de la base de Bonn
MIN_SAMPLES = 1024
 
 
# ----------------------------------------------------------------------
# Lecture et contrôle d'un fichier EEG
# ----------------------------------------------------------------------
def load_signal(path):
    """Lit un fichier texte (une valeur par ligne) et vérifie qu'il est exploitable."""
    try:
        signal = np.loadtxt(path, dtype=float)
    except ValueError:
        raise ValueError(
            "Le fichier doit contenir uniquement des nombres, une valeur par ligne.") from None
    signal = np.atleast_1d(signal)
    if signal.ndim != 1:
        raise ValueError("Le fichier doit contenir une seule colonne (un seul canal EEG).")
    if len(signal) < MIN_SAMPLES:
        raise ValueError(
            f"Signal trop court ({len(signal)} échantillons, minimum {MIN_SAMPLES}).")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Le signal contient des valeurs manquantes ou infinies.")
    if np.ptp(signal) == 0:
        raise ValueError("Le signal est constant : enregistrement inexploitable.")
    return signal
 
 
def signal_warnings(signal):
    warnings = []
    if len(signal) != EXPECTED_LEN:
        warnings.append(
            f"Attention : {len(signal)} échantillons au lieu de {EXPECTED_LEN} "
            "(durée différente des données d'entraînement) : résultat moins fiable.")
    return warnings
 
 
# ----------------------------------------------------------------------
# Informations sur le modèle (précision, sensibilité...)
# ----------------------------------------------------------------------
def read_info():
    try:
        with open(base.INFO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
 
 
def write_info(info):
    tmp = base.INFO_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, base.INFO_PATH)
 
 
def evaluate(model, Xs, y):
    """Précision, sensibilité et spécificité (en %) d'un modèle sur des données normalisées."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xs, dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
    tp = int(np.sum((pred == 1) & (y == 1)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    return {
        "accuracy": float(100 * np.mean(pred == y)),
        "sensitivity": float(100 * tp / max(tp + fn, 1)),
        "specificity": float(100 * tn / max(tn + fp, 1)),
    }
 
 
# ----------------------------------------------------------------------
# Détecteur
# ----------------------------------------------------------------------
class Detector:
    def __init__(self):
        self.model = self.scaler = self.features = None
        self.info = {}
        self.load()
 
    def load(self):
        if not (os.path.exists(base.MODEL_PATH) and os.path.exists(base.SCALER_PATH)):
            raise FileNotFoundError(
                "Modèle introuvable. Lance d'abord eeg_bonn_classification.py pour "
                f"l'entraîner (fichiers attendus dans {base.OUT_DIR}).")
        bundle = joblib.load(base.SCALER_PATH)
        self.scaler, self.features = bundle["scaler"], bundle["features"]
        model = base.EEGNet(len(self.features)).to(DEVICE)
        model.load_state_dict(torch.load(base.MODEL_PATH, map_location=DEVICE))
        model.eval()
        self.model = model
        self.info = read_info()
 
    def _vector(self, signal):
        feats = base.extract_features(signal)
        return np.array([[feats[name] for name in self.features]], dtype=float)
 
    def predict(self, signal):
        """Renvoie (classe, probabilité d'activité épileptique)."""
        x = self._vector(signal)
        if not np.all(np.isfinite(x)):
            raise ValueError("Signal inexploitable (caractéristiques non calculables).")
        x = self.scaler.transform(x)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32).to(DEVICE))
            proba = torch.softmax(logits, dim=1)[0, 1].item()
        return int(proba >= 0.5), proba
 
    # -- mise à jour du modèle ----------------------------------------
    def _bonn_splits(self):
        """Même découpage train/test que dans l'entraînement (mêmes graine et proportions)."""
        X_df, y, _ = base.load_bonn()
        return train_test_split(X_df[self.features].values, y, test_size=base.TEST_SIZE,
                                stratify=y, random_state=base.SEED)
 
    def retrain(self, signals, labels, epochs=30, lr=1e-4, batch_size=16):
        """Affine le modèle avec les cas validés par les praticiens, puis ne le
        remplace QUE s'il n'est pas moins bon que l'actuel sur le jeu de test Bonn.
        À exécuter hors du thread de l'interface. Ne touche pas à la base de données."""
        X_train, X_test, y_train, y_test = self._bonn_splits()
        X_new = np.vstack([self._vector(s) for s in signals])
        y_new = np.asarray(labels, dtype=int)
 
        Xs_fit = self.scaler.transform(np.vstack([X_train, X_new]))   # scaler inchangé
        y_fit = np.concatenate([y_train, y_new])
        Xs_test = self.scaler.transform(X_test)
 
        candidate = copy.deepcopy(self.model).to(DEVICE)
        torch.manual_seed(base.SEED)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(Xs_fit, dtype=torch.float32),
                          torch.tensor(y_fit, dtype=torch.long)),
            batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            candidate.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(candidate(xb), yb).backward()
                optimizer.step()
 
        before = evaluate(self.model, Xs_test, y_test)
        after = evaluate(candidate, Xs_test, y_test)
        # Règle prudente : aucune régression de la précision ni de la sensibilité.
        accepted = (after["accuracy"] >= before["accuracy"]
                    and after["sensitivity"] >= before["sensitivity"])
        if accepted:
            self._save_candidate(candidate, after, len(y_new), len(y_test))
        return {"accepted": accepted, "before": before, "after": after, "n_new": len(y_new)}
 
    def _save_candidate(self, candidate, metrics, n_new, n_test):
        # Sauvegarde de l'ancienne version, puis remplacement atomique des fichiers.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(BACKUP_DIR, stamp)
        os.makedirs(dest, exist_ok=True)
        for path in (base.MODEL_PATH, base.SCALER_PATH, base.INFO_PATH):
            if os.path.exists(path):
                shutil.copy2(path, dest)
        tmp = base.MODEL_PATH + ".tmp"
        torch.save(candidate.state_dict(), tmp)
        os.replace(tmp, base.MODEL_PATH)
        info = dict(self.info)
        info.update(metrics)
        info["n_test"] = int(n_test)
        info["version"] = int(info.get("version", 1)) + 1
        info["n_cas_ajoutes"] = int(info.get("n_cas_ajoutes", 0)) + int(n_new)
        info["date"] = datetime.now().isoformat(timespec="seconds")
        write_info(info)
 