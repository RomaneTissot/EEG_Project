# -*- coding: utf-8 -*-
"""Base de données SQLite : comptes praticiens, patients et diagnostics.
 
Différences avec la version d'origine :
- mots de passe hachés (PBKDF2 + sel), jamais stockés en clair ;
- chaque praticien ne voit que SES patients et SES diagnostics ;
- identifiants de diagnostic automatiques, clés étrangères activées ;
- le signal EEG est stocké dans la base (colonne BLOB) et non plus son chemin ;
- plus de listes globales (RPPS_lst, SS_lst) : l'unicité est garantie par la base.
"""
import hashlib
import hmac
import os
import sqlite3
import unicodedata
from datetime import datetime
 
import numpy as np
 
# Nouveau nom de fichier : le schéma a changé, l'ancien BDD.db n'est pas compatible.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg_app.db")
 
RPPS_LEN = 11
NIR_LEN = 13  # longueur retenue par le cahier des charges (un vrai NIR en compte 15)
MIN_PASSWORD_LEN = 8
PBKDF2_ITERATIONS = 600_000
 
ACCEPTE, REFUTE = "Accepté", "Réfuté"
 
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS comptes_prat (
    id_praticien TEXT PRIMARY KEY,
    nom TEXT NOT NULL,
    prenom TEXT NOT NULL,
    profession TEXT NOT NULL,
    laboratoire TEXT NOT NULL,
    mot_de_passe_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS Patients (
    id_patient TEXT NOT NULL,
    id_praticien TEXT NOT NULL,
    nom TEXT NOT NULL,
    prenom TEXT NOT NULL,
    PRIMARY KEY (id_patient, id_praticien),
    FOREIGN KEY (id_praticien) REFERENCES comptes_prat(id_praticien)
);
CREATE TABLE IF NOT EXISTS Diagnostics (
    id_diag INTEGER PRIMARY KEY AUTOINCREMENT,
    id_patient TEXT NOT NULL,
    id_praticien TEXT NOT NULL,
    date TEXT NOT NULL,
    labo TEXT,
    donnees BLOB NOT NULL,
    prediction INTEGER NOT NULL,
    probabilite REAL,
    statut TEXT NOT NULL CHECK (statut IN ('{ACCEPTE}', '{REFUTE}')),
    utilise_reentrainement INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (id_patient, id_praticien) REFERENCES Patients(id_patient, id_praticien)
);
"""
 
 
class ValidationError(Exception):
    """Erreur de saisie dont le message peut être affiché tel quel à l'utilisateur."""
 
 
# ----------------------------------------------------------------------
# Mots de passe
# ----------------------------------------------------------------------
def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"
 
 
def verify_password(password, stored):
    salt_hex, digest_hex = stored.split("$")
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS)
    return hmac.compare_digest(candidate.hex(), digest_hex)
 
 
# Haché une fois : sert à garder le même temps de réponse si le RPPS n'existe pas.
_DUMMY_HASH = hash_password("mot-de-passe-factice")
 
 
def _require_digits(value, length, label):
    if not (value.isascii() and value.isdigit() and len(value) == length):
        raise ValidationError(f"{label} doit comporter exactement {length} chiffres.")
 
 
def _normalize(text):
    """Minuscules sans accents, pour une recherche tolérante."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))
 
 
# ----------------------------------------------------------------------
# Accès aux données
# ----------------------------------------------------------------------
class Database:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
 
    def close(self):
        self.conn.close()
 
    # -- comptes -------------------------------------------------------
    def create_account(self, rpps, nom, prenom, profession, laboratoire, password):
        _require_digits(rpps, RPPS_LEN, "Le numéro RPPS")
        if not all([nom, prenom, profession, laboratoire]):
            raise ValidationError("Veuillez remplir tous les champs.")
        if len(password) < MIN_PASSWORD_LEN:
            raise ValidationError(
                f"Le mot de passe doit comporter au moins {MIN_PASSWORD_LEN} caractères.")
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO comptes_prat (id_praticien, nom, prenom, profession, "
                    "laboratoire, mot_de_passe_hash) VALUES (?, ?, ?, ?, ?, ?)",
                    (rpps, nom, prenom, profession, laboratoire, hash_password(password)))
        except sqlite3.IntegrityError:
            raise ValidationError("Ce numéro RPPS existe déjà.") from None
        return self.get_account(rpps)
 
    def get_account(self, rpps):
        return self.conn.execute(
            "SELECT * FROM comptes_prat WHERE id_praticien = ?", (rpps,)).fetchone()
 
    def authenticate(self, rpps, password):
        """Renvoie le compte si les identifiants sont corrects, sinon None.
        On ne distingue pas un mauvais RPPS d'un mauvais mot de passe."""
        row = self.get_account(rpps)
        stored = row["mot_de_passe_hash"] if row else _DUMMY_HASH
        ok = verify_password(password, stored)
        return row if (row is not None and ok) else None
 
    # -- patients ------------------------------------------------------
    def add_patient(self, id_praticien, ss, nom, prenom):
        _require_digits(ss, NIR_LEN, "Le numéro de sécurité sociale")
        if not (nom and prenom):
            raise ValidationError("Veuillez remplir tous les champs.")
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO Patients (id_patient, id_praticien, nom, prenom) "
                    "VALUES (?, ?, ?, ?)", (ss, id_praticien, nom, prenom))
        except sqlite3.IntegrityError:
            raise ValidationError("Ce patient existe déjà dans votre liste.") from None
        return self.get_patient(id_praticien, ss)
 
    def get_patient(self, id_praticien, ss):
        return self.conn.execute(
            "SELECT * FROM Patients WHERE id_praticien = ? AND id_patient = ?",
            (id_praticien, ss)).fetchone()
 
    def list_patients(self, id_praticien, query=""):
        rows = self.conn.execute(
            "SELECT * FROM Patients WHERE id_praticien = ? ORDER BY nom, prenom",
            (id_praticien,)).fetchall()
        query = _normalize(query.strip())
        if not query:
            return rows
        return [r for r in rows
                if query in _normalize(r["nom"]) or query in _normalize(r["prenom"])
                or query in r["id_patient"]]
 
    # -- diagnostics ---------------------------------------------------
    def add_diagnostic(self, id_praticien, id_patient, labo, signal,
                       prediction, proba, statut):
        blob = np.asarray(signal, dtype="<f8").tobytes()
        date = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO Diagnostics (id_patient, id_praticien, date, labo, donnees, "
                "prediction, probabilite, statut) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (id_patient, id_praticien, date, labo, blob, int(prediction),
                 float(proba), statut))
        return cur.lastrowid
 
    def list_diagnostics(self, id_praticien, id_patient):
        return self.conn.execute(
            "SELECT id_diag, date, prediction, probabilite, statut FROM Diagnostics "
            "WHERE id_praticien = ? AND id_patient = ? ORDER BY date DESC",
            (id_praticien, id_patient)).fetchall()
 
    # -- données pour la mise à jour du modèle -------------------------
    def count_pending_training(self):
        return self.conn.execute(
            "SELECT COUNT(*) FROM Diagnostics WHERE utilise_reentrainement = 0").fetchone()[0]
 
    def pending_training_samples(self):
        """Cas validés par un praticien et pas encore utilisés pour la mise à jour.
        Étiquette : la prédiction si elle a été acceptée, son contraire si elle a été réfutée."""
        rows = self.conn.execute(
            "SELECT id_diag, donnees, prediction, statut FROM Diagnostics "
            "WHERE utilise_reentrainement = 0 ORDER BY id_diag").fetchall()
        samples = []
        for r in rows:
            label = r["prediction"] if r["statut"] == ACCEPTE else 1 - r["prediction"]
            samples.append((r["id_diag"], np.frombuffer(r["donnees"], dtype="<f8").copy(), label))
        return samples
 
    def mark_used(self, ids):
        with self.conn:
            self.conn.executemany(
                "UPDATE Diagnostics SET utilise_reentrainement = 1 WHERE id_diag = ?",
                [(i,) for i in ids])
 