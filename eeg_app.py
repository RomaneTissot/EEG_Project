# -*- coding: utf-8 -*-
"""
Interface praticiens : comptes, patients, diagnostics EEG assistés par IA.
 
Lancement : python eeg_app.py
(après avoir entraîné le modèle avec eeg_bonn_classification.py)
 
Dépendances : customtkinter, fpdf2, matplotlib, torch, scikit-learn, joblib
"""
import io
import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox
 
import customtkinter as ctk
import numpy as np
from fpdf import FPDF
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
 
import eeg_ai
from eeg_bonn_classification import FS, OUT_DIR
from eeg_db import ACCEPTE, REFUTE, Database, ValidationError
 
REPORTS_DIR = os.path.join(OUT_DIR, "Diagnostics")
RETRAIN_THRESHOLD = 10   # nombre de cas validés avant de proposer une mise à jour
MAX_ATTEMPTS = 5         # tentatives de connexion avant blocage temporaire
LOCK_SECONDS = 30
 
DISCLAIMER = (
    "Outil d'aide à la décision issu d'un projet étudiant. Le modèle est entraîné sur la "
    "base publique de Bonn (segments de 23,6 s, un seul canal) et n'a pas été validé "
    "cliniquement. Il ne remplace pas l'avis d'un médecin."
)
 
 
# ----------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------
def fmt_date(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y à %H:%M")
    except ValueError:
        return iso
 
 
def diagnosis_text(prediction):
    return ("Activité épileptique détectée" if prediction == 1
            else "Aucune activité épileptique détectée")
 
 
def describe_model(info):
    if not info:
        return "Performances du modèle non renseignées (relancer l'entraînement)."
    return (f"Modèle v{info.get('version', 1)} : précision {info['accuracy']:.1f} %, "
            f"sensibilité {info['sensitivity']:.1f} %, spécificité {info['specificity']:.1f} % "
            f"(jeu de test Bonn, {info.get('n_test', '?')} segments).")
 
 
def latin(text):
    """Les polices standard de FPDF ne gèrent que le latin-1."""
    return str(text).encode("latin-1", "replace").decode("latin-1")
 
 
def make_figure(signal):
    fig = Figure(figsize=(10, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(signal)) / FS, signal, linewidth=0.7)
    ax.set_title("EEG du patient")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
 
 
def clear(widget):
    for child in widget.winfo_children():
        child.destroy()
 
 
def generate_pdf_report(account, patient, signal, prediction, proba, statut, info):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
 
    def write(text, size=12, bold=False, align="L"):
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.multi_cell(0, 7, latin(text), align=align, new_x="LMARGIN", new_y="NEXT")
 
    write("Rapport de diagnostic", 18, True, "C")
    write(datetime.now().strftime("%d/%m/%Y à %H:%M"), align="C")
    pdf.ln(4)
    write(f"Praticien : Dr {account['prenom']} {account['nom']} ({account['profession']})")
    write(f"Laboratoire : {account['laboratoire']}")
    write(f"Patient : {patient['nom']} {patient['prenom']}")
    write(f"N° de sécurité sociale : {patient['id_patient']}")
    pdf.ln(4)
    write("Résultat du diagnostic IA", 14, True)
    write(f"{diagnosis_text(prediction)} (décision du praticien : {statut})")
    write(f"Probabilité d'activité épileptique estimée : {proba * 100:.1f} %")
    pdf.ln(2)
    write("Modèle utilisé", 14, True)
    write(describe_model(info))
    write("Base d'entraînement : base de données EEG de Bonn (sujets sains et épileptiques).")
    pdf.ln(2)
    write(DISCLAIMER, 10)
    pdf.ln(4)
 
    buf = io.BytesIO()
    make_figure(signal).savefig(buf, format="png")
    buf.seek(0)
    pdf.image(buf, w=pdf.epw)
 
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORTS_DIR, f"Diagnostic_{patient['id_patient']}_{stamp}.pdf")
    pdf.output(path)
    return path
 
 
# ----------------------------------------------------------------------
# Fenêtres secondaires
# ----------------------------------------------------------------------
class BusyWindow(ctk.CTkToplevel):
    def __init__(self, parent, text):
        super().__init__(parent)
        self.title("Veuillez patienter")
        self.geometry("420x150")
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # impossible de la fermer à la main
        ctk.CTkLabel(self, text=text, wraplength=380).pack(pady=(25, 10))
        bar = ctk.CTkProgressBar(self, mode="indeterminate", width=320)
        bar.pack(pady=10)
        bar.start()
        self.transient(parent)
 
 
class CurveWindow(ctk.CTkToplevel):
    def __init__(self, parent, signal):
        super().__init__(parent)
        self.title("EEG du patient")
        self.geometry("950x420")
        canvas = FigureCanvasTkAgg(make_figure(signal), master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
 
 
class DiagnosisPopup(ctk.CTkToplevel):
    """Résultat de l'IA : le praticien l'accepte ou le réfute."""
 
    def __init__(self, parent, patient, signal, prediction, proba):
        super().__init__(parent)
        self.parent, self.patient, self.signal = parent, patient, signal
        self.prediction, self.proba = prediction, proba
        self.title("Résultat du diagnostic")
        self.geometry("740x480")
        self.transient(parent.winfo_toplevel())
        self.lift()
 
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        ctk.CTkLabel(frame, text=f"Patient : {patient['nom']} {patient['prenom']}").pack(
            pady=(10, 0))
        ctk.CTkLabel(frame, text="Résultat du diagnostic IA", font=("Arial", 16)).pack(pady=5)
        color = "#e74c3c" if prediction == 1 else "#2ecc71"
        ctk.CTkLabel(frame, text=diagnosis_text(prediction), font=("Arial", 20, "bold"),
                     text_color=color).pack(pady=5)
        ctk.CTkLabel(frame, text=f"Probabilité d'activité épileptique estimée : "
                                 f"{proba * 100:.1f} %").pack()
        if 0.2 < proba < 0.8:
            ctk.CTkLabel(frame, text="Résultat peu tranché : interprétation à faire avec "
                                     "une prudence particulière.",
                         text_color="#f39c12").pack(pady=5)
        ctk.CTkLabel(frame, text=describe_model(parent.app.detector.info), wraplength=640,
                     font=("Arial", 12)).pack(pady=(10, 0))
        ctk.CTkLabel(frame, text=DISCLAIMER, wraplength=640, justify="left",
                     text_color="gray", font=("Arial", 11)).pack(pady=10)
 
        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.pack(pady=10)
        ctk.CTkButton(buttons, text="Accepter", fg_color="green",
                      command=lambda: self._decide(ACCEPTE)).pack(side="left", padx=10)
        ctk.CTkButton(buttons, text="Réfuter", fg_color="red",
                      command=lambda: self._decide(REFUTE)).pack(side="left", padx=10)
        ctk.CTkButton(buttons, text="Voir la courbe",
                      command=lambda: CurveWindow(self, self.signal)).pack(side="left", padx=10)
 
    def _decide(self, statut):
        parent = self.parent
        self.destroy()
        parent.finish_diagnostic(self.patient, self.signal, self.prediction, self.proba, statut)
 
 
# ----------------------------------------------------------------------
# Page de connexion
# ----------------------------------------------------------------------
class LoginFrame(ctk.CTkFrame):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.failures = 0
 
        card = ctk.CTkFrame(self, corner_radius=10)
        card.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(card, text="Connexion à l'espace Praticiens", font=("Arial", 20)).pack(
            padx=40, pady=20)
        ctk.CTkLabel(card, text="N° de praticien (RPPS) :").pack()
        self.entry_user = ctk.CTkEntry(card, width=230)
        self.entry_user.pack()
        ctk.CTkLabel(card, text="Mot de passe :").pack(pady=(10, 0))
        self.entry_pwd = ctk.CTkEntry(card, show="*", width=230)
        self.entry_pwd.pack()
 
        buttons = ctk.CTkFrame(card, fg_color="transparent")
        buttons.pack(pady=20)
        self.btn_login = ctk.CTkButton(buttons, text="Se connecter", command=self.login)
        self.btn_login.pack(side="left", padx=10)
        ctk.CTkButton(buttons, text="Créer un compte",
                      command=app.show_create_account).pack(side="left", padx=10)
 
    def on_enter(self):
        self.login()
 
    def login(self):
        if self.btn_login.cget("state") == "disabled":
            return
        account = self.app.db.authenticate(self.entry_user.get().strip(), self.entry_pwd.get())
        self.entry_pwd.delete(0, "end")
        if account is not None:
            self.app.show_user(account)
            return
        self.failures += 1
        if self.failures >= MAX_ATTEMPTS:
            self.btn_login.configure(state="disabled")
            self.after(LOCK_SECONDS * 1000, self._unlock)
            messagebox.showerror(
                "Erreur", f"Trop de tentatives. Réessayez dans {LOCK_SECONDS} secondes.")
        else:
            messagebox.showerror("Erreur", "Identifiants incorrects")
 
    def _unlock(self):
        self.failures = 0
        self.btn_login.configure(state="normal")
 
 
# ----------------------------------------------------------------------
# Création de compte
# ----------------------------------------------------------------------
class CreateAccountFrame(ctk.CTkFrame):
    FIELDS = [("N° de praticien (RPPS, 11 chiffres)", False), ("Nom", False),
              ("Prénom", False), ("Profession", False), ("Laboratoire", False),
              ("Mot de passe (8 caractères minimum)", True),
              ("Confirmer le mot de passe", True)]
 
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        card = ctk.CTkFrame(self, corner_radius=10)
        card.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(card, text="Créer un compte", font=("Arial", 20)).pack(padx=40, pady=20)
 
        self.entries = []
        for label, secret in self.FIELDS:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(pady=4, padx=20)
            ctk.CTkLabel(row, text=f"{label} :", width=270, anchor="e").pack(side="left")
            entry = ctk.CTkEntry(row, width=220, show="*" if secret else "")
            entry.pack(side="left", padx=10)
            self.entries.append(entry)
 
        ctk.CTkButton(card, text="Confirmer", command=self.confirm).pack(pady=(15, 5))
        ctk.CTkButton(card, text="Retour", command=app.show_login).pack(pady=(5, 20))
 
    def on_enter(self):
        self.confirm()
 
    def confirm(self):
        values = [e.get().strip() for e in self.entries[:5]]
        password, confirmation = self.entries[5].get(), self.entries[6].get()
        if password != confirmation:
            messagebox.showerror("Erreur", "Les deux mots de passe ne correspondent pas.")
            return
        try:
            account = self.app.db.create_account(*values, password)
        except ValidationError as exc:
            messagebox.showerror("Erreur", str(exc))
            return
        messagebox.showinfo("Et voilà !", "Compte créé avec succès !")
        self.app.show_user(account)
 
 
# ----------------------------------------------------------------------
# Interface principale (praticien connecté)
# ----------------------------------------------------------------------
class UserFrame(ctk.CTkFrame):
    def __init__(self, app, account):
        super().__init__(app)
        self.app, self.db = app, app.db
        self.account = account
        self.praticien_id = account["id_praticien"]
        self.current_patient = None
        self.signal = None
 
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1, uniform="cols")
        self.grid_columnconfigure(1, weight=2, uniform="cols")
        self.grid_columnconfigure(2, weight=1, uniform="cols")
        self._build_top_bar()
 
        self.patient_list = ctk.CTkScrollableFrame(self, label_text="Patients")
        self.patient_list.grid(row=1, column=0, sticky="nswe", padx=10, pady=10)
        self.center = ctk.CTkFrame(self)
        self.center.grid(row=1, column=1, sticky="nswe", padx=10, pady=10)
        self.history = ctk.CTkScrollableFrame(self, label_text="Historique des diagnostics")
        self.history.grid(row=1, column=2, sticky="nswe", padx=10, pady=10)
 
        self._show_welcome()
        self.refresh_patients()
 
    # -- construction --------------------------------------------------
    def _build_top_bar(self):
        bar = ctk.CTkFrame(self, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        ctk.CTkButton(bar, text="≡", width=34, command=self.open_menu).pack(
            side="left", padx=10, pady=10)
        ctk.CTkButton(bar, text="Nouveau patient", command=self.open_new_patient).pack(
            side="left", padx=10)
        self.search_entry = ctk.CTkEntry(bar, placeholder_text="Rechercher un patient...",
                                         width=260)
        self.search_entry.pack(side="left", padx=10)
        self.search_entry.bind("<Return>", lambda event: self.search_patient())
        ctk.CTkButton(bar, text="Rechercher", command=self.search_patient).pack(side="left")
        ctk.CTkLabel(bar, text=self.account["laboratoire"]).pack(side="right", padx=10)
        ctk.CTkLabel(bar, text=self.account["profession"]).pack(side="right", padx=10)
        ctk.CTkLabel(bar, text=f"Dr {self.account['nom']}").pack(side="right", padx=10)
 
    def _show_welcome(self):
        clear(self.center)
        ctk.CTkLabel(self.center, text="Veuillez sélectionner un patient",
                     font=("Arial", 16)).place(relx=0.5, rely=0.5, anchor="center")
 
    # -- menu ----------------------------------------------------------
    def open_menu(self):
        pending = self.db.count_pending_training()
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label=f"Mettre à jour le modèle ({pending} cas en attente)",
                         command=self.app.start_retraining)
        menu.add_command(label="Informations sur le modèle", command=self.show_model_info)
        menu.add_separator()
        menu.add_command(label="Déconnexion", command=self.app.show_login)
        menu.add_command(label="Quitter", command=self.app.on_close)
        try:
            menu.tk_popup(self.winfo_rootx() + 10, self.winfo_rooty() + 50)
        finally:
            menu.grab_release()
 
    def show_model_info(self):
        messagebox.showinfo("Modèle", describe_model(self.app.detector.info))
 
    # -- patients ------------------------------------------------------
    def refresh_patients(self, query=""):
        clear(self.patient_list)
        patients = self.db.list_patients(self.praticien_id, query)
        if not patients:
            ctk.CTkLabel(self.patient_list, text="Aucun patient trouvé.").pack(pady=10)
            return
        for patient in patients:
            ctk.CTkButton(self.patient_list, text=f"{patient['nom']} {patient['prenom']}",
                          command=lambda p=patient: self.show_patient(p)).pack(
                fill="x", pady=3)
 
    def search_patient(self):
        self.refresh_patients(self.search_entry.get())
        self.focus()
 
    def open_new_patient(self):
        clear(self.center)
        clear(self.history)
        ctk.CTkLabel(self.center, text="Nouveau patient", font=("Arial", 18)).pack(pady=15)
        self.new_patient_entries = []
        for label in ("N° de sécurité sociale (13 chiffres)", "Nom", "Prénom"):
            row = ctk.CTkFrame(self.center, fg_color="transparent")
            row.pack(pady=5)
            ctk.CTkLabel(row, text=f"{label} :", width=240, anchor="e").pack(side="left")
            entry = ctk.CTkEntry(row, width=220)
            entry.pack(side="left", padx=10)
            entry.bind("<Return>", lambda event: self.add_new_patient())
            self.new_patient_entries.append(entry)
        ctk.CTkButton(self.center, text="Ajouter", command=self.add_new_patient).pack(pady=15)
 
    def add_new_patient(self):
        ss, nom, prenom = (e.get().strip() for e in self.new_patient_entries)
        try:
            patient = self.db.add_patient(self.praticien_id, ss, nom, prenom)
        except ValidationError as exc:
            messagebox.showerror("Erreur", str(exc))
            return
        self.refresh_patients()
        self.show_patient(patient)
 
    def show_patient(self, patient):
        self.current_patient = patient
        self.signal = None
        clear(self.center)
        ctk.CTkLabel(self.center, text=f"{patient['nom']} {patient['prenom']}",
                     font=("Arial", 18)).pack(pady=(25, 5))
        ctk.CTkLabel(self.center, text=f"N° de sécurité sociale : {patient['id_patient']}").pack(
            pady=5)
        ctk.CTkButton(self.center, text="Nouveau diagnostic",
                      command=self.open_new_diagnostic).pack(pady=15)
        self.refresh_history()
 
    def refresh_history(self):
        clear(self.history)
        diagnostics = self.db.list_diagnostics(
            self.praticien_id, self.current_patient["id_patient"])
        if not diagnostics:
            ctk.CTkLabel(self.history, text="Aucun diagnostic trouvé.").pack(pady=10)
            return
        for d in diagnostics:
            text = f"{fmt_date(d['date'])}\n{diagnosis_text(d['prediction'])} ({d['statut']})"
            ctk.CTkLabel(self.history, text=text, wraplength=230).pack(pady=6)
 
    # -- diagnostic ----------------------------------------------------
    def open_new_diagnostic(self):
        clear(self.center)
        self.signal = None
        patient = self.current_patient
        ctk.CTkLabel(self.center, text=f"Nouveau diagnostic : {patient['nom']} "
                                       f"{patient['prenom']}", font=("Arial", 18)).pack(pady=15)
        ctk.CTkButton(self.center, text="Déposer un fichier .txt",
                      command=self.choose_file).pack(pady=10)
        self.file_label = ctk.CTkLabel(self.center, text="Aucun fichier sélectionné")
        self.file_label.pack(pady=5)
        self.warn_label = ctk.CTkLabel(self.center, text="", text_color="#f39c12",
                                       wraplength=420)
        self.warn_label.pack(pady=5)
        self.diag_button = ctk.CTkButton(self.center, text="Lancer le diagnostic",
                                         command=self.run_diagnostic, state="disabled")
        self.diag_button.pack(pady=10)
 
    def choose_file(self):
        path = filedialog.askopenfilename(
            parent=self, title="Sélectionnez le fichier EEG",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")])
        if not path:
            return
        try:
            signal = eeg_ai.load_signal(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Fichier invalide", str(exc))
            return
        self.signal = signal
        self.file_label.configure(
            text=f"Fichier : {os.path.basename(path)} ({len(signal)} échantillons)")
        self.warn_label.configure(text="\n".join(eeg_ai.signal_warnings(signal)))
        self.diag_button.configure(state="normal")
 
    def run_diagnostic(self):
        if self.signal is None:
            messagebox.showerror("Erreur", "Veuillez d'abord sélectionner un fichier .txt")
            return
        try:
            prediction, proba = self.app.detector.predict(self.signal)
        except ValueError as exc:
            messagebox.showerror("Erreur", str(exc))
            return
        DiagnosisPopup(self, self.current_patient, self.signal, prediction, proba)
 
    def finish_diagnostic(self, patient, signal, prediction, proba, statut):
        """Enregistre la décision du praticien, génère le rapport PDF."""
        try:
            self.db.add_diagnostic(self.praticien_id, patient["id_patient"],
                                   self.account["laboratoire"], signal, prediction, proba,
                                   statut)
        except Exception as exc:  # noqa: BLE001 - on veut toujours informer l'utilisateur
            messagebox.showerror("Erreur", f"Le diagnostic n'a pas pu être enregistré : {exc}")
            return
        if self.current_patient is not None and \
                self.current_patient["id_patient"] == patient["id_patient"]:
            self.refresh_history()
        try:
            path = generate_pdf_report(self.account, patient, signal, prediction, proba,
                                       statut, self.app.detector.info)
            messagebox.showinfo("Rapport généré", f"Le rapport a été enregistré :\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("Rapport", "Le diagnostic est enregistré mais le PDF n'a "
                                              f"pas pu être créé : {exc}")
        self.check_retraining()
 
    def check_retraining(self):
        pending = self.db.count_pending_training()
        if pending and pending % RETRAIN_THRESHOLD == 0:
            if messagebox.askyesno(
                    "Mise à jour du modèle",
                    f"{pending} cas validés sont disponibles.\n"
                    "Lancer la mise à jour du modèle maintenant ?\n\n"
                    "Le nouveau modèle ne sera adopté que s'il n'est pas moins performant "
                    "que l'actuel sur le jeu de test."):
                self.app.start_retraining()
 
 
# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self, db, detector):
        super().__init__()
        self.db, self.detector = db, detector
        self.current = None
        self.retraining = False
        self.title("AI Diagnosis for EEG")
        self.geometry("1280x720")
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Return>", self._on_return)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_login()
 
    def toggle_fullscreen(self, event=None):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))
 
    def _on_return(self, event=None):
        handler = getattr(self.current, "on_enter", None)
        if handler:
            handler()
 
    def _show(self, frame):
        if self.current is not None:
            self.current.destroy()
        self.current = frame
        frame.pack(fill="both", expand=True)
 
    def show_login(self):
        self._show(LoginFrame(self))
 
    def show_create_account(self):
        self._show(CreateAccountFrame(self))
 
    def show_user(self, account):
        self._show(UserFrame(self, account))
 
    def on_close(self):
        if self.retraining:
            messagebox.showwarning("Mise à jour en cours",
                                   "Une mise à jour du modèle est en cours. Réessayez ensuite.")
            return
        if messagebox.askokcancel("Quitter", "Voulez-vous quitter l'application ?"):
            self.db.close()
            self.destroy()
 
    # -- mise à jour du modèle (en arrière-plan) -----------------------
    def start_retraining(self):
        if self.retraining:
            messagebox.showinfo("Mise à jour du modèle", "Une mise à jour est déjà en cours.")
            return
        samples = self.db.pending_training_samples()
        if not samples:
            messagebox.showinfo("Mise à jour du modèle", "Aucun nouveau cas validé en attente.")
            return
        ids = [s[0] for s in samples]
        signals = [s[1] for s in samples]
        labels = [s[2] for s in samples]
 
        self.retraining = True
        busy = BusyWindow(self, "Mise à jour du modèle en cours...\n"
                                "Cela peut prendre quelques minutes.")
        results = queue.Queue()
 
        def worker():
            try:
                results.put(("ok", self.detector.retrain(signals, labels)))
            except Exception as exc:  # noqa: BLE001
                results.put(("err", exc))
 
        threading.Thread(target=worker, daemon=True).start()
        self.after(300, self._poll_retraining, results, busy, ids)
 
    def _poll_retraining(self, results, busy, ids):
        try:
            kind, payload = results.get_nowait()
        except queue.Empty:
            self.after(300, self._poll_retraining, results, busy, ids)
            return
        busy.destroy()
        self.retraining = False
        if kind == "err":
            messagebox.showerror("Mise à jour impossible", str(payload))
            return
        before, after = payload["before"], payload["after"]
        summary = (f"Cas pris en compte : {payload['n_new']}\n"
                   f"Précision : {before['accuracy']:.1f} % → {after['accuracy']:.1f} %\n"
                   f"Sensibilité : {before['sensitivity']:.1f} % → {after['sensitivity']:.1f} %\n"
                   f"Spécificité : {before['specificity']:.1f} % → {after['specificity']:.1f} %")
        if payload["accepted"]:
            self.detector.load()
            self.db.mark_used(ids)
            messagebox.showinfo("Modèle mis à jour", summary)
        else:
            messagebox.showinfo(
                "Mise à jour non retenue",
                "Le modèle candidat est moins performant sur le jeu de test : "
                "l'ancien modèle est conservé.\n\n" + summary)
 
 
def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    try:
        detector = eeg_ai.Detector()
    except FileNotFoundError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Modèle introuvable", str(exc))
        root.destroy()
        return
    App(Database(), detector).mainloop()
 
 
if __name__ == "__main__":
    main()
 