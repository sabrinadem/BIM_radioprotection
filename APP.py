"""
APP.py
===========
Interface graphique PyQt5 pour le calcul d'attenuation de composite, quelle que soit la source de rayonnement.

"""

import sys
import os
import csv
import traceback
from datetime import datetime

from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QDoubleSpinBox, QPushButton, QLabel, QLineEdit,
    QFileDialog, QTabWidget, QMessageBox, QSplitter, QStackedWidget, QFrame,
)

# --- Modules physiques ---------------------------------
from sources import spectre_Pd103, spectre_Pd103mono, spectre_I125, charger_spectre_fichier
from composite_physics import densite_composite, mu_rho_composite, mu_rho_charge, mu_rho_matrice, CHARGES, MATRICES
from attenuation import (
    pourcentage_attenuation,
    epaisseur_pour_blocage,
    spectre_transmis,
    mu_rho_effectif,
)


# =========================================================================
#  Canvas matplotlib reutilisable
# =========================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1, figsize=(6, 5)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        if nrows * ncols == 1:
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.subplots(nrows, ncols)
        super().__init__(self.fig)

    def clear(self):
        for ax in np.atleast_1d(self.axes).ravel():
            ax.clear()


# =========================================================================
#  Fenetre principale
# =========================================================================
class MainWindow(QMainWindow):

    SOURCES_CONNUES = ["Pd-103", "Pd-103mono", "I-125", "Fichier spectre (SpekCalc, .spec)"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atténuation composite")
        self.resize(1300, 800)

        self.chemin_fichier_spectre = None
        self.chemins_fichiers_multiples = []  # pour la comparaison multi-spectres (onglet équivalence Pb)
        self.dernier_calcul = None  # rempli par _calculer(), utilise par _exporter_resultats()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_panneau_inputs())
        splitter.addWidget(self._build_panneau_graphiques())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 900])

        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Prêt. Configurez la source et le composite, puis cliquez sur Calculer.")
        self.showMaximized()

    # ------------------------------------------------------------------
    #  PANNEAU DE GAUCHE : inputs
    # ------------------------------------------------------------------
    def _build_panneau_inputs(self):
        panneau = QWidget()
        layout = QVBoxLayout(panneau)
        layout.setAlignment(Qt.AlignTop)

        # --- Groupe : source de rayonnement ---------------------------
        groupe_source = QGroupBox("Source de rayonnement")
        form_source = QFormLayout()

        self.combo_source = QComboBox()
        self.combo_source.addItems(self.SOURCES_CONNUES)
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        form_source.addRow("Type de source :", self.combo_source)

        self.ligne_fichier = QLineEdit()
        self.ligne_fichier.setPlaceholderText("Aucun fichier selectionné")
        self.ligne_fichier.setReadOnly(True)
        self.bouton_parcourir = QPushButton("Parcourir...")
        self.bouton_parcourir.clicked.connect(self._choisir_fichier_spectre)

        ligne_fichier_layout = QHBoxLayout()
        ligne_fichier_layout.addWidget(self.ligne_fichier)
        ligne_fichier_layout.addWidget(self.bouton_parcourir)
        self.widget_fichier = QWidget()
        self.widget_fichier.setLayout(ligne_fichier_layout)
        form_source.addRow("Fichier :", self.widget_fichier)

        groupe_source.setLayout(form_source)
        layout.addWidget(groupe_source)

        # --- Groupe : comparaison multi-spectres (onglet équivalence Pb) --
        groupe_multi = QGroupBox("Comparaison multi-spectres (onglet Équivalence Pb)")
        v_multi = QVBoxLayout()

        self.bouton_fichiers_multiples = QPushButton("Sélectionner plusieurs fichiers .spec...")
        self.bouton_fichiers_multiples.clicked.connect(self._choisir_fichiers_multiples)
        v_multi.addWidget(self.bouton_fichiers_multiples)

        self.label_fichiers_multiples = QLabel("Aucun fichier sélectionné (mode % charge utilisé)")
        self.label_fichiers_multiples.setWordWrap(True)
        v_multi.addWidget(self.label_fichiers_multiples)

        self.bouton_effacer_multiples = QPushButton("Effacer la sélection")
        self.bouton_effacer_multiples.clicked.connect(self._effacer_fichiers_multiples)
        v_multi.addWidget(self.bouton_effacer_multiples)

        groupe_multi.setLayout(v_multi)
        layout.addWidget(groupe_multi)

        # --- Groupe : composite (materiau) ----------------------------
        groupe_composite = QGroupBox("Composite (materiau NIST)")
        form_composite = QFormLayout()

        self.combo_charge = QComboBox()
        self.combo_charge.addItems(["Bi2O3", "W", "Pb"])  
        form_composite.addRow("Charge (element lourd) :", self.combo_charge)

        self.combo_matrice = QComboBox()
        self.combo_matrice.addItems(["PEEK", "TPU", "PLA", "PETG"])  
        form_composite.addRow("Matrice (polymere) :", self.combo_matrice)

        self.spin_fraction = QDoubleSpinBox()
        self.spin_fraction.setRange(0.0, 100.0)
        self.spin_fraction.setSuffix(" %")
        self.spin_fraction.setDecimals(1)
        self.spin_fraction.setValue(35.5)
        self.spin_fraction.setSingleStep(1.0)
        form_composite.addRow("Fraction massique charge :", self.spin_fraction)

        self.label_densite = QLabel("Densité composite : -- g/cm3")
        form_composite.addRow(self.label_densite)
        self.spin_fraction.valueChanged.connect(self._maj_densite)
        self.combo_charge.currentIndexChanged.connect(self._maj_densite)
        self.combo_matrice.currentIndexChanged.connect(self._maj_densite)
        self._maj_densite()

        groupe_composite.setLayout(form_composite)
        layout.addWidget(groupe_composite)

        # --- Groupe : parametres des graphiques -----------------------
        groupe_params = QGroupBox("Paramètres des graphiques")
        form_params = QFormLayout()

        self.spin_epaisseur_max = QDoubleSpinBox()
        self.spin_epaisseur_max.setRange(0.01, 100.0)
        self.spin_epaisseur_max.setSuffix(" mm")
        self.spin_epaisseur_max.setDecimals(2)
        self.spin_epaisseur_max.setValue(10.0)
        form_params.addRow("Épaisseur max (affichage axe) :", self.spin_epaisseur_max)

        self.spin_epaisseur_fixe = QDoubleSpinBox()
        self.spin_epaisseur_fixe.setRange(0.001, 100.0)
        self.spin_epaisseur_fixe.setSuffix(" mm")
        self.spin_epaisseur_fixe.setDecimals(3)
        self.spin_epaisseur_fixe.setValue(1.0)
        form_params.addRow("Épaisseur fixe (graph. 5-6) :", self.spin_epaisseur_fixe)

        self.spin_pct_cible = QDoubleSpinBox()
        self.spin_pct_cible.setRange(0.0, 99.9999)
        self.spin_pct_cible.setSuffix(" %")
        self.spin_pct_cible.setDecimals(3)
        self.spin_pct_cible.setValue(99.9)
        form_params.addRow("% attenuation cible :", self.spin_pct_cible)

        groupe_params.setLayout(form_params)
        layout.addWidget(groupe_params)

        # --- Bouton calculer --------------------------------------------
        self.bouton_calculer = QPushButton("Calculer / Mettre à jour les graphiques")
        self.bouton_calculer.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; background-color: #2d6cdf; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1f57b8; }"
        )
        self.bouton_calculer.clicked.connect(self._calculer)
        layout.addWidget(self.bouton_calculer)

        self.bouton_exporter = QPushButton("Exporter les résultats...")
        self.bouton_exporter.setEnabled(False)  # active seulement apres un calcul reussi
        self.bouton_exporter.clicked.connect(self._exporter_resultats)
        layout.addWidget(self.bouton_exporter)

        # --- Resultats -----------------------------------------------
        groupe_resultats = QGroupBox("Résultats")
        form_resultats = QFormLayout()
        self.label_flux = QLabel("--")
        self.label_pct_a_epaisseur_fixe = QLabel("--")
        self.label_epaisseur_solution = QLabel("--")
        self.label_mu_rho_eff = QLabel("--")
        form_resultats.addRow("Flux incident intégré :", self.label_flux)
        form_resultats.addRow("% attenuation (ep. fixe) :", self.label_pct_a_epaisseur_fixe)
        form_resultats.addRow("Épaisseur pour % cible :", self.label_epaisseur_solution)
        form_resultats.addRow("μ/ρ effectif:", self.label_mu_rho_eff)
        groupe_resultats.setLayout(form_resultats)
        layout.addWidget(groupe_resultats)

        layout.addStretch()
        self._on_source_changed()
        return panneau

    # ------------------------------------------------------------------
    #  PANNEAU DE DROITE : graphiques (onglets)
    # ------------------------------------------------------------------
    def _build_panneau_graphiques(self):
        self.onglets = QTabWidget()

        self.canvas_spectre = MplCanvas(figsize=(7, 5))
        self.onglets.addTab(self._wrap_canvas(self.canvas_spectre), "Spectre source")

        self.canvas_mu_rho = MplCanvas(figsize=(7, 5))
        self.onglets.addTab(self._wrap_canvas(self.canvas_mu_rho), "μ/ρ composite et elements purs vs énergie")

        self.canvas_mu_rho_eff = MplCanvas(figsize=(7, 5))
        self.onglets.addTab(self._wrap_canvas(self.canvas_mu_rho_eff), "μ/ρ effectif vs épaisseur")

        self.canvas_vs_epaisseur = MplCanvas(figsize=(7, 5))
        self.onglets.addTab(self._wrap_canvas(self.canvas_vs_epaisseur), "Atténuation vs épaisseur")

        self.canvas_vs_fraction = MplCanvas(figsize=(7, 5))
        self.onglets.addTab(self._wrap_canvas(self.canvas_vs_fraction), "Atténuation vs % charge")

        self.canvas_vs_energie = MplCanvas(figsize=(7, 5))
        self.onglets.addTab(self._wrap_canvas(self.canvas_vs_energie), "Atténuation vs énergie")

        self.canvas_equiv_plomb = MplCanvas(figsize=(7, 5))
        self.onglets.addTab(self._wrap_canvas(self.canvas_equiv_plomb), "Équivalence 0.5mm Pb")

        return self.onglets

    @staticmethod
    def _wrap_canvas(canvas):
        conteneur = QWidget()
        v = QVBoxLayout(conteneur)
        v.addWidget(canvas)
        return conteneur

    # ------------------------------------------------------------------
    #  Logique
    # ------------------------------------------------------------------
    def _on_source_changed(self):
        est_fichier = self.combo_source.currentText().startswith("Fichier")
        self.widget_fichier.setVisible(est_fichier)

    def _choisir_fichier_spectre(self):
        chemin, _ = QFileDialog.getOpenFileName(
            self, "Choisir un fichier de spectre", "", "Fichiers spectre (*.spec);;Tous les fichiers (*)"
        )
        if chemin:
            self.chemin_fichier_spectre = chemin
            self.ligne_fichier.setText(chemin)

    def _choisir_fichiers_multiples(self):
        """Selection de plusieurs fichiers .spec en une fois, pour comparer
        directement leurs epaisseurs equivalentes a 0.5 mm Pb sans avoir a
        changer la source dans le menu deroulant a chaque fois."""
        chemins, _ = QFileDialog.getOpenFileNames(
            self, "Choisir plusieurs fichiers de spectre (.spec)", "",
            "Fichiers spectre (*.spec);;Tous les fichiers (*)"
        )
        if chemins:
            self.chemins_fichiers_multiples = chemins
            self.label_fichiers_multiples.setText(
                f"{len(chemins)} fichier(s) sélectionné(s) pour la comparaison."
            )

    def _effacer_fichiers_multiples(self):
        self.chemins_fichiers_multiples = []
        self.label_fichiers_multiples.setText("Aucun fichier sélectionné (mode % charge utilisé)")

    def _maj_densite(self):
        w = self.spin_fraction.value() / 100
        nom_charge = self.combo_charge.currentText()
        nom_matrice = self.combo_matrice.currentText()
        rho = densite_composite(w, nom_charge=nom_charge, nom_matrice=nom_matrice)
        self.label_densite.setText(f"Densité composite : {rho:.3f} g/cm3")

    def _charger_source(self):
        """Retourne (E_array, I_array, mode) selon le choix du menu déroulant."""
        choix = self.combo_source.currentText()
        if choix == "Pd-103":
            return spectre_Pd103()
        elif choix == "Pd-103mono":
            return spectre_Pd103mono()
        elif choix == "I-125":
            return spectre_I125()
        else:
            if not self.chemin_fichier_spectre:
                raise ValueError("Veuillez selectionner un fichier de spectre (bouton Parcourir).")
            return charger_spectre_fichier(self.chemin_fichier_spectre)

    def _calculer(self):
        try:
            E_array, I_array, mode = self._charger_source()
        except Exception as exc:
            QMessageBox.critical(self, "Erreur - source", f"Impossible de charger la source :\n{exc}")
            traceback.print_exc()
            return

        w_frac = self.spin_fraction.value() / 100
        nom_charge = self.combo_charge.currentText()
        nom_matrice = self.combo_matrice.currentText()
        ep_max_mm = self.spin_epaisseur_max.value()
        ep_fixe_mm = self.spin_epaisseur_fixe.value()
        pct_cible = self.spin_pct_cible.value()

        try:
            self._tracer_spectre(E_array, I_array, mode)
            x_mm, pct_vs_ep = self._tracer_vs_epaisseur(
                E_array, I_array, mode, w_frac, nom_charge, nom_matrice, ep_max_mm, pct_cible)
            w_pct_range, pct_vs_frac = self._tracer_vs_fraction(
                E_array, I_array, mode, nom_charge, nom_matrice, ep_fixe_mm)
            E_mu_rho, mu_rho_vals, mu_rho_matrice_vals, mu_rho_charge_vals = self._tracer_mu_rho_composite(
                w_frac, nom_charge, nom_matrice)
            x_mm_eff, mu_rho_eff_vals = self._tracer_mu_rho_effectif(
                E_array, I_array, mode, w_frac, nom_charge, nom_matrice, ep_max_mm)
            resultats_num = self._maj_resultats(
                E_array, I_array, mode, w_frac, nom_charge, nom_matrice, ep_fixe_mm, pct_cible)
            E_plot, pct_vs_energie = self._tracer_vs_energie(
                mode, nom_charge, nom_matrice, w_frac, ep_fixe_mm)
            
            # --- Nouvel appel pour le graphique d'équivalence Plomb ---
            self._tracer_equivalence_kvp(E_array, I_array, mode, w_frac, nom_charge, nom_matrice)

            # On garde tout ce qu'il faut pour l'export, sans rien recalculer
            self.dernier_calcul = {
                "source": self.combo_source.currentText(),
                "mode": mode,
                "nom_charge": nom_charge,
                "nom_matrice": nom_matrice,
                "w_frac_pct": w_frac * 100,
                "ep_max_mm": ep_max_mm,
                "ep_fixe_mm": ep_fixe_mm,
                "pct_cible": pct_cible,
                "E_array": E_array,
                "I_array": I_array,
                "x_mm": x_mm,
                "pct_vs_epaisseur": pct_vs_ep,
                "w_pct_range": w_pct_range,
                "pct_vs_fraction": pct_vs_frac,
                "E_mu_rho": E_mu_rho,
                "mu_rho_composite": mu_rho_vals,
                "mu_rho_matrice": mu_rho_matrice_vals,
                "mu_rho_charge": mu_rho_charge_vals,
                "x_mm_mu_rho_eff": x_mm_eff,
                "mu_rho_effectif": mu_rho_eff_vals,
                "resultats_num": resultats_num,
            }
            self.bouton_exporter.setEnabled(True)

            self.statusBar().showMessage(
                f"Calcul terminé - source : {self.combo_source.currentText()}, "
                f"composite : {nom_charge}/{nom_matrice}, mode : {mode}, {len(E_array)} points/raies.")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur - calcul", f"Une erreur est survenue pendant le calcul :\n{exc}")
            traceback.print_exc()

    # --- Graphique 1 : spectre incident ------------------------------
    def _tracer_spectre(self, E_array, I_array, mode):
        c = self.canvas_spectre
        c.clear()
        ax = c.axes
        if mode == "continu":
            ax.plot(E_array, I_array, color="tab:blue")
            ax.fill_between(E_array, I_array, alpha=0.2, color="tab:blue")
        else:
            ax.stem(E_array, I_array, basefmt=" ")
        ax.set_xlabel("Énergie (keV)")
        ax.set_ylabel("Intensité")
        ax.set_title(f"Spectre incident ({mode})")
        ax.grid(True, alpha=0.3)
        c.draw()

    # --- Graphique 2 : % attenuation vs epaisseur --------------------
    def _tracer_vs_epaisseur(self, E_array, I_array, mode, w_frac, nom_charge, nom_matrice, ep_max_mm, pct_cible):
        x_mm = np.linspace(1e-4, ep_max_mm, 200)
        x_cm = x_mm / 10
        pct = np.array([
            pourcentage_attenuation(E_array, I_array, mode, w_frac, x,
                                     nom_charge=nom_charge, nom_matrice=nom_matrice)[0]
            for x in x_cm
        ])

        c = self.canvas_vs_epaisseur
        c.clear()
        ax = c.axes
        ax.plot(x_mm, pct, color="tab:green")
        ax.axhline(pct_cible, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Cible = {pct_cible:.2f} %")

        try:
            x_sol_cm = epaisseur_pour_blocage(
                E_array, I_array, mode, w_frac,
                pourcentage_cible=pct_cible, x_min=1e-5, x_max=max(ep_max_mm / 10, 5.0),
                nom_charge=nom_charge, nom_matrice=nom_matrice
            )
            x_sol_mm = x_sol_cm * 10
            if x_sol_mm <= ep_max_mm:
                ax.axvline(x_sol_mm, color="tab:red", linestyle="--", linewidth=0.8,
                           label=f"Solution = {x_sol_mm:.4f} mm")
        except Exception:
            pass  # pas de solution dans l'intervalle : on affiche quand meme la courbe

        ax.set_xlabel("Épaisseur (mm)")
        ax.set_ylabel("% atténuation")
        ax.set_title(f"Atténuation vs épaisseur ({nom_charge} = {w_frac*100:.1f} %)")
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(10.0))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        c.draw()
        return x_mm, pct

    # --- Graphique 3 : % attenuation vs % massique de la charge --------
    def _tracer_vs_fraction(self, E_array, I_array, mode, nom_charge, nom_matrice, ep_fixe_mm):
        w_pct_range = np.linspace(0.0, 100.0, 200)
        x_cm = ep_fixe_mm / 10
        pct = np.array([
            pourcentage_attenuation(E_array, I_array, mode, w / 100, x_cm,
                                     nom_charge=nom_charge, nom_matrice=nom_matrice)[0]
            for w in w_pct_range
        ])

        w_actuel = self.spin_fraction.value()

        c = self.canvas_vs_fraction
        c.clear()
        ax = c.axes
        ax.plot(w_pct_range, pct, color="tab:purple")
        ax.axvline(w_actuel, color="tab:red", linestyle="--", linewidth=0.8,
                   label=f"Valeur actuelle = {w_actuel:.1f} %")
        ax.set_xlabel(f"Fraction massique {nom_charge} (%)")
        ax.set_ylabel("% atténuation")
        ax.set_title(f"Atténuation vs % {nom_charge} (épaisseur = {ep_fixe_mm:.3f} mm)")
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(10.0))
        ax.yaxis.set_major_locator(MultipleLocator(10.0))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        c.draw()
        return w_pct_range, pct

    # --- Graphique 4 : mu/rho du composite, matrice pure et charge pure vs energie --
    def _tracer_mu_rho_composite(self, w_frac, nom_charge, nom_matrice):
        E_plot = np.linspace(4, 60, 400)
        mu_rho_vals = np.array([
            mu_rho_composite(E, w_frac, nom_charge=nom_charge, nom_matrice=nom_matrice) for E in E_plot
        ])
        mu_rho_matrice_vals = np.array([mu_rho_matrice(E, nom_matrice) for E in E_plot])
        mu_rho_charge_vals = np.array([mu_rho_charge(E, nom_charge) for E in E_plot])

        nom_charge_aff = CHARGES[nom_charge]["nom_affiche"]
        nom_matrice_aff = MATRICES[nom_matrice]["nom_affiche"]

        c = self.canvas_mu_rho
        c.clear()
        ax = c.axes
        ax.plot(E_plot, mu_rho_charge_vals, "--", label=f"{nom_charge_aff} pur", color="tab:red")
        ax.plot(E_plot, mu_rho_matrice_vals, "--", label=f"{nom_matrice_aff} pur", color="tab:blue")
        ax.plot(E_plot, mu_rho_vals, color="black", linewidth=2,
                label=f"Composite ({w_frac*100:.1f} % {nom_charge_aff})")
        if nom_charge == "Bi2O3":
            for seuil in (13.4186, 15.7111, 16.3875):
                ax.axvline(seuil, color="gray", linestyle=":", linewidth=0.8)
        elif nom_charge == "W":
            for seuil in (10.207, 11.544, 12.100):
                ax.axvline(seuil, color="gray", linestyle=":", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Énergie (keV)")
        ax.set_ylabel(r"$\mu/\rho$ (cm$^2$/g)")
        ax.set_title(f"μ/ρ : composite ({w_frac*100:.1f} % {nom_charge_aff}), "
                     f"{nom_matrice_aff} pur et {nom_charge_aff} pur vs énergie")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        c.draw()
        return E_plot, mu_rho_vals, mu_rho_matrice_vals, mu_rho_charge_vals

    # --- Graphique 5 : mu/rho effectif vs epaisseur --------------------
    def _tracer_mu_rho_effectif(self, E_array, I_array, mode, w_frac, nom_charge, nom_matrice, ep_max_mm):
        x_mm = np.linspace(ep_max_mm / 150, ep_max_mm, 150)
        x_cm = x_mm / 10
        mu_rho_eff_vals = np.array([
            mu_rho_effectif(E_array, I_array, mode, w_frac, x,
                             nom_charge=nom_charge, nom_matrice=nom_matrice)[1]
            for x in x_cm
        ])

        c = self.canvas_mu_rho_eff
        c.clear()
        ax = c.axes
        ax.plot(x_mm, mu_rho_eff_vals, color="tab:brown", linewidth=2)
        ax.set_xlabel("Épaisseur (mm)")
        ax.set_ylabel(r"$(\mu/\rho)_{eff}$ (cm$^2$/g)")
        ax.set_title("μ/ρ effectif vs épaisseur")
        ax.grid(True, alpha=0.3)
        c.draw()
        return x_mm, mu_rho_eff_vals


# --- Graphique 6 : % attenuation vs énergie -------------------------
    def _tracer_vs_energie(self, mode, nom_charge, nom_matrice, w_frac, ep_fixe_mm):
        E_plot = np.linspace(4, 60, 400)
        x_cm = ep_fixe_mm / 10

        nom_charge_aff = CHARGES[nom_charge]["nom_affiche"]
        nom_matrice_aff = MATRICES[nom_matrice]["nom_affiche"]

        pct = np.array([
            pourcentage_attenuation(np.array([E]), np.array([1.0]), mode, w_frac, x_cm,
                                     nom_charge=nom_charge, nom_matrice=nom_matrice)[0]
            for E in E_plot
        ])

        c = self.canvas_vs_energie
        c.clear()
        ax = c.axes
        ax.plot(E_plot, pct, color="tab:green",
                label=f"Composite ({w_frac*100:.1f} % {nom_charge_aff})")

        if nom_charge == "Bi2O3":
            for seuil in (13.4186, 15.7111, 16.3875):
                ax.axvline(seuil, color="gray", linestyle=":", linewidth=0.8)
        elif nom_charge == "W":
            for seuil in (10.207, 11.544, 12.100):
                ax.axvline(seuil, color="gray", linestyle=":", linewidth=0.8)

        ax.set_xlabel("Énergie (keV)")
        ax.set_ylabel("% atténuation")
        ax.set_title(f"Atténuation vs énergie ({w_frac*100:.1f} % {nom_charge_aff}, "
                     f"épaisseur = {ep_fixe_mm:.3f} mm)")
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.yaxis.set_major_locator(MultipleLocator(10.0))
        ax.set_ylim(0, 105)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        c.draw()
        return E_plot, pct


    # --- Panneau de resultats numeriques -------------------------------
    def _maj_resultats(self, E_array, I_array, mode, w_frac, nom_charge, nom_matrice, ep_fixe_mm, pct_cible):
        pct_fixe, flux_inc, flux_trans = pourcentage_attenuation(
            E_array, I_array, mode, w_frac, ep_fixe_mm / 10,
            nom_charge=nom_charge, nom_matrice=nom_matrice)
        self.label_flux.setText(f"{flux_inc:.4g}")
        self.label_pct_a_epaisseur_fixe.setText(f"{pct_fixe:.4f} % (a {ep_fixe_mm:.3f} mm)")

        x_sol_mm = None
        try:
            x_sol_cm = epaisseur_pour_blocage(
                E_array, I_array, mode, w_frac, pourcentage_cible=pct_cible,
                nom_charge=nom_charge, nom_matrice=nom_matrice)
            x_sol_mm = x_sol_cm * 10
            self.label_epaisseur_solution.setText(f"{x_sol_mm:.4f} mm (pour {pct_cible:.2f} %)")
        except Exception:
            self.label_epaisseur_solution.setText("Pas de solution dans l'intervalle testée")

        # mu/rho effectif a l'epaisseur fixe
        # pour signaler visuellement si un beam hardening notable est present)
        mu_eff, mu_rho_eff, rho = mu_rho_effectif(
            E_array, I_array, mode, w_frac, ep_fixe_mm / 10,
            nom_charge=nom_charge, nom_matrice=nom_matrice)
        mu_eff2, mu_rho_eff2, _ = mu_rho_effectif(
            E_array, I_array, mode, w_frac, (ep_fixe_mm / 2) / 10,
            nom_charge=nom_charge, nom_matrice=nom_matrice)
        ecart_pct = abs(mu_rho_eff - mu_rho_eff2) / mu_rho_eff * 100 if mu_rho_eff != 0 else 0.0
        texte_mu_rho = f"{mu_rho_eff:.4f} cm2/g (μ_eff = {mu_eff:.4f} cm-1)"
        self.label_mu_rho_eff.setText(texte_mu_rho)

        return {
            "flux_incident": flux_inc,
            "flux_transmis": flux_trans,
            "pct_attenuation_epaisseur_fixe": pct_fixe,
            "epaisseur_solution_mm": x_sol_mm,
            "mu_effectif": mu_eff,
            "mu_rho_effectif": mu_rho_eff,
            "rho_composite": rho,
        }

    def _tracer_equivalence_kvp(self, E_array, I_array, mode, w_frac, nom_charge, nom_matrice):
        """
        Épaisseur de composite équivalente à 0.5 mm de plomb.

        Deux modes, selon que des fichiers multiples ont ete choisis
        (bouton "Sélectionner plusieurs fichiers .spec...") ou non :

        - Mode COMPARAISON (fichiers multiples selectionnes) : un point/barre
          par fichier, epaisseur equivalente calculee pour CHAQUE spectre a
          la fraction/charge/matrice actuellement reglees dans l'interface.
          Utile pour comparer par ex. 8 spectres SpekCalc (differents kVp)
          sans avoir a changer la source dans le menu deroulant a chaque fois.

        - Mode FRACTION (aucun fichier multiple) : courbe epaisseur
          equivalente vs fraction massique de charge, pour la source UNIQUE
          deja chargee via le menu deroulant / "Parcourir...".
        """
        from attenuation import epaisseur_composite_equivalente_plomb
        import re

        c = self.canvas_equiv_plomb
        c.clear()
        ax = c.axes

        if self.chemins_fichiers_multiples:
            # ---------------- Mode comparaison multi-spectres -----------
            labels, epaisseurs = [], []
            for chemin in self.chemins_fichiers_multiples:
                nom_fichier = os.path.basename(chemin)
                m = re.search(r'(\d+(?:\.\d+)?)\s*kvp', nom_fichier, re.IGNORECASE)
                label = f"{m.group(1)} kVp" if m else nom_fichier
                try:
                    E_spec, I_spec, mode_spec = charger_spectre_fichier(chemin)
                    _, ep_mm = epaisseur_composite_equivalente_plomb(
                        E_spec, I_spec, mode_spec, w_frac, nom_charge, nom_matrice
                    )
                except Exception:
                    ep_mm = np.nan
                labels.append(label)
                epaisseurs.append(ep_mm)

            # Tri par kVp si tous les labels sont numeriques
            try:
                ordre = sorted(range(len(labels)), key=lambda i: float(labels[i].split()[0]))
                labels = [labels[i] for i in ordre]
                epaisseurs = [epaisseurs[i] for i in ordre]
            except Exception:
                pass

            x_pos = np.arange(len(labels))
            ax.bar(x_pos, epaisseurs, color="tab:blue")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Épaisseur composite équivalente à 0,5 mm Pb (mm)")
            ax.set_title(f"Équivalence plomb - {len(labels)} spectres "
                         f"({nom_charge} = {w_frac*100:.1f} %, matrice {nom_matrice})")
            for xi, ep in zip(x_pos, epaisseurs):
                if np.isfinite(ep):
                    ax.annotate(f"{ep:.3f}", (xi, ep), textcoords="offset points",
                                xytext=(0, 4), ha="center", fontsize=8)
            resultat = (labels, epaisseurs)
        else:
            # ---------------- Mode vs fraction (source unique) ----------
            w_pct_range = np.linspace(1.0, 100.0, 40)  # on evite 0% (matrice pure)
            epaisseurs = []
            for w_pct in w_pct_range:
                try:
                    _, ep_mm = epaisseur_composite_equivalente_plomb(
                        E_array, I_array, mode, w_pct / 100, nom_charge, nom_matrice
                    )
                except Exception:
                    ep_mm = np.nan
                epaisseurs.append(ep_mm)
            epaisseurs = np.array(epaisseurs)

            try:
                _, ep_actuelle_mm = epaisseur_composite_equivalente_plomb(
                    E_array, I_array, mode, w_frac, nom_charge, nom_matrice
                )
            except Exception:
                ep_actuelle_mm = None

            ax.plot(w_pct_range, epaisseurs, 'o-', color="tab:blue", linewidth=2, markersize=3)
            if ep_actuelle_mm is not None and np.isfinite(ep_actuelle_mm):
                ax.axvline(w_frac * 100, color="tab:red", linestyle="--", linewidth=0.8,
                           label=f"{w_frac*100:.1f} % -> {ep_actuelle_mm:.4f} mm")
                ax.legend()
            ax.set_xlabel(f"Fraction massique {nom_charge} (%)")
            ax.set_ylabel("Épaisseur composite équivalente à 0,5 mm Pb (mm)")
            ax.set_title(f"Équivalence plomb (0,5 mm) - source : {self.combo_source.currentText()}")
            resultat = (w_pct_range, epaisseurs)

        ax.grid(True, alpha=0.3)
        c.draw()
        return resultat
    # ------------------------------------------------------------------
    #  EXPORT DES RESULTATS
    # ------------------------------------------------------------------
    def _exporter_resultats(self):
        if not self.dernier_calcul:
            QMessageBox.warning(self, "Rien à exporter", "Veuillez d'abord cliquer sur Calculer.")
            return

        dossier = QFileDialog.getExistingDirectory(self, "Choisir un dossier de destination")
        if not dossier:
            return

        d = self.dernier_calcul
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        sous_dossier = os.path.join(dossier, f"resultats_attenuation_{horodatage}")

        try:
            os.makedirs(sous_dossier, exist_ok=True)

            # 1) Resume texte des parametres et resultats -----------------
            chemin_resume = os.path.join(sous_dossier, "resume.txt")
            with open(chemin_resume, "w", encoding="utf-8") as f:
                f.write("Résumé du calcul d'atténuation - composite\n")
                f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write("--- Paramètres ---\n")
                f.write(f"Source                         : {d['source']}\n")
                f.write(f"Mode                           : {d['mode']}\n")
                f.write(f"Charge                         : {d['nom_charge']}\n")
                f.write(f"Matrice                        : {d['nom_matrice']}\n")
                f.write(f"Fraction massique charge       : {d['w_frac_pct']:.2f} %\n")
                f.write(f"Épaisseur max (graph. 1)       : {d['ep_max_mm']:.3f} mm\n")
                f.write(f"Épaisseur fixe (graph. 2)      : {d['ep_fixe_mm']:.3f} mm\n")
                f.write(f"% atténuation cible            : {d['pct_cible']:.3f} %\n\n")
                f.write("--- Résultats ---\n")
                r = d["resultats_num"]
                f.write(f"Flux incident intégré          : {r['flux_incident']:.6g}\n")
                f.write(f"Flux transmis (ép. fixe)       : {r['flux_transmis']:.6g}\n")
                f.write(f"% atténuation (ép. fixe)       : {r['pct_attenuation_epaisseur_fixe']:.4f} %\n")
                if r["epaisseur_solution_mm"] is not None:
                    f.write(f"Épaisseur pour % cible         : {r['epaisseur_solution_mm']:.4f} mm\n")
                else:
                    f.write("Épaisseur pour % cible         : pas de solution dans l'intervalle testée\n")
                f.write(f"μ effectif (ép. fixe)          : {r['mu_effectif']:.6g} cm-1\n")
                f.write(f"μ/ρ effectif (ép. fixe)        : {r['mu_rho_effectif']:.6g} cm2/g\n")
                f.write(f"ρ composite                    : {r['rho_composite']:.4f} g/cm3\n")

            # 2) Donnees brutes en CSV -------------------------------------
            with open(os.path.join(sous_dossier, "spectre_source.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Énergie_keV", "Intensité"])
                w.writerows(zip(d["E_array"], d["I_array"]))

            with open(os.path.join(sous_dossier, "attenuation_vs_epaisseur.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Épaisseur_mm", "Pourcentage_atténuation"])
                w.writerows(zip(d["x_mm"], d["pct_vs_epaisseur"]))

            with open(os.path.join(sous_dossier, f"attenuation_vs_fraction_{d['nom_charge']}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([f"Fraction_massique_{d['nom_charge']}_pct", "Pourcentage_atténuation"])
                w.writerows(zip(d["w_pct_range"], d["pct_vs_fraction"]))

            with open(os.path.join(sous_dossier, "mu_rho_composite_vs_energie.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Énergie_keV", "mu_rho_composite_cm2_g",
                            f"mu_rho_{d['nom_matrice']}_cm2_g", f"mu_rho_{d['nom_charge']}_pur_cm2_g"])
                w.writerows(zip(d["E_mu_rho"], d["mu_rho_composite"], d["mu_rho_matrice"], d["mu_rho_charge"]))

            with open(os.path.join(sous_dossier, "mu_rho_effectif_vs_epaisseur.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Épaisseur_mm", "mu_rho_effectif_cm2_g"])
                w.writerows(zip(d["x_mm_mu_rho_eff"], d["mu_rho_effectif"]))

            # 3) Graphiques en PNG (haute resolution) ----------------------
            self.canvas_spectre.fig.savefig(os.path.join(sous_dossier, "graphique_spectre.png"), dpi=200)
            self.canvas_vs_epaisseur.fig.savefig(os.path.join(sous_dossier, "graphique_vs_epaisseur.png"), dpi=200)
            self.canvas_vs_fraction.fig.savefig(os.path.join(sous_dossier, "graphique_vs_fraction.png"), dpi=200)
            self.canvas_mu_rho.fig.savefig(os.path.join(sous_dossier, "graphique_mu_rho_composite.png"), dpi=200)
            self.canvas_mu_rho_eff.fig.savefig(os.path.join(sous_dossier, "graphique_mu_rho_effectif.png"), dpi=200)
            self.canvas_vs_energie.fig.savefig(os.path.join(sous_dossier, "graphique_vs_energie.png"), dpi=200)

            QMessageBox.information(
                self, "Export reussi",
                f"Résultats exportés avec succès dans :\n{sous_dossier}\n\n"
                "Contenu : resume.txt, 5 fichiers CSV et 5 graphiques PNG."
            )
            self.statusBar().showMessage(f"Exporté dans : {sous_dossier}")

        except Exception as exc:
            QMessageBox.critical(self, "Erreur - export", f"Impossible d'exporter les résultats :\n{exc}")
            traceback.print_exc()




def main():
    app = QApplication(sys.argv)
    fenetre = MainWindow()
    fenetre.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()