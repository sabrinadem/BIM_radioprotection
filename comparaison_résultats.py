"""
comparaison_mu_rho.py
======================
Outil INDEPENDANT (ne depend d'aucun module du projet d'attenuation) pour
superposer sur UN SEUL graphique plusieurs courbes mu/rho vs energie,
provenant de fichiers CSV deja exportes par APP.py (bouton "Exporter les
resultats...") :

    - mu_rho_composite_vs_energie.csv
        colonnes : Énergie_keV, mu_rho_composite_cm2_g,
                   mu_rho_<matrice>_cm2_g, mu_rho_<charge>_pur_cm2_g
    - mu_rho_composite_vs_spectre_source.csv
        colonnes : Énergie_keV, Intensité_source, mu_rho_composite_cm2_g

Fonctionne aussi avec n'importe quel autre CSV du moment qu'il contient :
    - une colonne d'en-tete contenant "nergie" (energie / Énergie)
    - une ou plusieurs colonnes d'en-tete contenant "mu_rho" (ou "mu/rho")

Usage :
    python comparaison_mu_rho.py

Dans la fenetre :
    1. Cliquer "Ajouter fichier(s)..." et choisir un ou plusieurs CSV.
    2. Si un fichier contient plusieurs colonnes mu/rho (ex : composite +
       matrice pure + charge pure), une boîte de dialogue permet de choisir
       lesquelles importer comme courbes separees.
    3. Chaque courbe apparaît dans la liste a gauche, avec une case a
       cocher (visible/invisible). Double-cliquer sur le nom pour le
       renommer. Bouton "Changer couleur" pour la recolorer.
    4. Le graphique (echelle log Y optionnelle) se met a jour automatiquement.
    5. Bouton "Exporter PNG..." pour sauvegarder l'image finale.
"""

import sys
import os
import csv
import re
import itertools

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog, QLabel,
    QCheckBox, QColorDialog, QMessageBox, QLineEdit, QDialog,
    QDialogButtonBox, QAbstractItemView, QGroupBox, QFormLayout,
    QDoubleSpinBox,
)


# =========================================================================
#  Lecture generique d'un CSV "energie + une ou plusieurs colonnes mu_rho"
# =========================================================================
def lire_colonnes_mu_rho(chemin):
    """
    Retourne (E_array, candidats) ou candidats est une liste de
    (nom_colonne, valeurs_array), triee par energie croissante.
    Leve ValueError si aucune colonne pertinente n'est detectee.
    """
    with open(chemin, "r", encoding="utf-8-sig", newline="") as f:
        lignes = list(csv.reader(f))

    if not lignes:
        raise ValueError("Fichier vide.")

    entete = lignes[0]
    corps = lignes[1:]

    idx_energie = next(
        (i for i, nom in enumerate(entete) if re.search(r"nergie", nom, re.IGNORECASE)),
        None
    )
    if idx_energie is None:
        raise ValueError(
            "Aucune colonne d'énergie détectée "
            "(en-tête attendu contenant 'nergie', ex : Énergie_keV)."
        )

    indices_mu_rho = [
        i for i, nom in enumerate(entete)
        if re.search(r"mu[_/]?rho", nom, re.IGNORECASE)
    ]
    if not indices_mu_rho:
        raise ValueError(
            "Aucune colonne μ/ρ détectée "
            "(en-tête attendu contenant 'mu_rho')."
        )

    E_vals = []
    mu_vals = {i: [] for i in indices_mu_rho}

    for ligne in corps:
        if not ligne or len(ligne) <= max([idx_energie] + indices_mu_rho):
            continue
        try:
            e = float(ligne[idx_energie])
        except ValueError:
            continue
        E_vals.append(e)
        for i in indices_mu_rho:
            try:
                mu_vals[i].append(float(ligne[i]))
            except ValueError:
                mu_vals[i].append(np.nan)

    if not E_vals:
        raise ValueError("Aucune ligne de données numériques exploitable dans ce fichier.")

    E = np.array(E_vals, dtype=float)
    candidats = [(entete[i], np.array(mu_vals[i], dtype=float)) for i in indices_mu_rho]

    idx_tri = np.argsort(E)
    E = E[idx_tri]
    candidats = [(nom, vals[idx_tri]) for nom, vals in candidats]
    return E, candidats


# =========================================================================
#  Boîte de dialogue : choisir quelles colonnes mu_rho importer
# =========================================================================
class SelectionColonnesDialog(QDialog):
    def __init__(self, nom_fichier, noms_colonnes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choisir les colonnes μ/ρ à importer")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Le fichier :\n{nom_fichier}\n\n"
            "contient plusieurs colonnes μ/ρ. Cochez celles à ajouter\n"
            "comme courbes séparées sur le graphique :"
        ))
        self.checkboxes = []
        for nom in noms_colonnes:
            cb = QCheckBox(nom)
            cb.setChecked(True)
            layout.addWidget(cb)
            self.checkboxes.append(cb)
        boutons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        boutons.accepted.connect(self.accept)
        boutons.rejected.connect(self.reject)
        layout.addWidget(boutons)

    def colonnes_selectionnees(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]


# =========================================================================
#  Canvas matplotlib
# =========================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(8, 6)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


# =========================================================================
#  Fenetre principale
# =========================================================================
class FenetreComparaison(QMainWindow):

    COULEURS_DEFAUT = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comparaison μ/ρ vs énergie - superposition multi-fichiers")
        self.resize(1200, 750)

        self.courbes = {}          # id (int) -> {"E":..., "mu_rho":..., "color":...}
        self._compteur_id = 0
        self._cycle_couleurs = itertools.cycle(self.COULEURS_DEFAUT)

        conteneur = QWidget()
        layout_principal = QHBoxLayout(conteneur)

        # ---------------- Panneau de gauche : liste des courbes ----------
        panneau_gauche = QWidget()
        v = QVBoxLayout(panneau_gauche)
        v.setAlignment(Qt.AlignTop)

        self.bouton_ajouter = QPushButton("Ajouter fichier(s)...")
        self.bouton_ajouter.clicked.connect(self._ajouter_fichiers)
        v.addWidget(self.bouton_ajouter)

        self.liste_courbes = QListWidget()
        self.liste_courbes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.liste_courbes.itemChanged.connect(self._on_item_change)
        self.liste_courbes.setMinimumWidth(300)
        v.addWidget(QLabel("Courbes chargées (cocher = visible, double-clic = renommer) :"))
        v.addWidget(self.liste_courbes)

        ligne_boutons = QHBoxLayout()
        self.bouton_couleur = QPushButton("Changer couleur")
        self.bouton_couleur.clicked.connect(self._changer_couleur)
        self.bouton_supprimer = QPushButton("Supprimer")
        self.bouton_supprimer.clicked.connect(self._supprimer_selection)
        ligne_boutons.addWidget(self.bouton_couleur)
        ligne_boutons.addWidget(self.bouton_supprimer)
        v.addLayout(ligne_boutons)

        # --- Options d'affichage ------------------------------------
        groupe_options = QGroupBox("Options du graphique")
        form = QFormLayout()

        self.case_log_y = QCheckBox("Échelle log (axe Y)")
        self.case_log_y.setChecked(True)
        self.case_log_y.stateChanged.connect(self._redessiner)
        form.addRow(self.case_log_y)

        self.champ_titre = QLineEdit("μ/ρ vs énergie - comparaison")
        self.champ_titre.textChanged.connect(self._redessiner)
        form.addRow("Titre :", self.champ_titre)

        self.case_limiter_x = QCheckBox("Limiter l'axe X")
        self.case_limiter_x.stateChanged.connect(self._redessiner)
        form.addRow(self.case_limiter_x)

        self.spin_x_min = QDoubleSpinBox()
        self.spin_x_min.setRange(0.0, 100000.0)
        self.spin_x_min.setValue(4.0)
        self.spin_x_min.setSuffix(" keV")
        self.spin_x_min.valueChanged.connect(self._redessiner)
        form.addRow("X min :", self.spin_x_min)

        self.spin_x_max = QDoubleSpinBox()
        self.spin_x_max.setRange(0.0, 100000.0)
        self.spin_x_max.setValue(150.0)
        self.spin_x_max.setSuffix(" keV")
        self.spin_x_max.valueChanged.connect(self._redessiner)
        form.addRow("X max :", self.spin_x_max)

        groupe_options.setLayout(form)
        v.addWidget(groupe_options)

        self.bouton_exporter_png = QPushButton("Exporter PNG...")
        self.bouton_exporter_png.clicked.connect(self._exporter_png)
        v.addWidget(self.bouton_exporter_png)

        v.addStretch()

        # ---------------- Panneau de droite : graphique -------------------
        self.canvas = MplCanvas()

        layout_principal.addWidget(panneau_gauche, 0)
        layout_principal.addWidget(self.canvas, 1)

        self.setCentralWidget(conteneur)
        self.statusBar().showMessage(
            "Prêt. Cliquez sur 'Ajouter fichier(s)...' pour charger vos CSV exportés."
        )
        self._redessiner()

    # ------------------------------------------------------------------
    #  Ajout de fichiers
    # ------------------------------------------------------------------
    def _ajouter_fichiers(self):
        chemins, _ = QFileDialog.getOpenFileNames(
            self, "Choisir un ou plusieurs fichiers CSV", "",
            "Fichiers CSV (*.csv);;Tous les fichiers (*)"
        )
        if not chemins:
            return

        for chemin in chemins:
            nom_fichier = os.path.basename(chemin)
            try:
                E, candidats = lire_colonnes_mu_rho(chemin)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Fichier ignoré",
                    f"Impossible de lire :\n{nom_fichier}\n\n{exc}"
                )
                continue

            if len(candidats) > 1:
                dialogue = SelectionColonnesDialog(
                    nom_fichier, [nom for nom, _ in candidats], self
                )
                if dialogue.exec_() != QDialog.Accepted:
                    continue
                colonnes_choisies = set(dialogue.colonnes_selectionnees())
            else:
                colonnes_choisies = {candidats[0][0]}

            for nom_colonne, valeurs in candidats:
                if nom_colonne not in colonnes_choisies:
                    continue
                label = f"{os.path.splitext(nom_fichier)[0]} — {nom_colonne}"
                self._ajouter_courbe(label, E, valeurs)

        self._redessiner()

    def _ajouter_courbe(self, label, E, mu_rho):
        id_courbe = self._compteur_id
        self._compteur_id += 1
        couleur = next(self._cycle_couleurs)
        self.courbes[id_courbe] = {"E": E, "mu_rho": mu_rho, "color": couleur}

        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
        item.setCheckState(Qt.Checked)
        item.setData(Qt.UserRole, id_courbe)
        self.liste_courbes.addItem(item)

    # ------------------------------------------------------------------
    #  Suppression / couleur / renommage
    # ------------------------------------------------------------------
    def _supprimer_selection(self):
        for item in self.liste_courbes.selectedItems():
            id_courbe = item.data(Qt.UserRole)
            self.courbes.pop(id_courbe, None)
            self.liste_courbes.takeItem(self.liste_courbes.row(item))
        self._redessiner()

    def _changer_couleur(self):
        items = self.liste_courbes.selectedItems()
        if not items:
            QMessageBox.information(self, "Aucune sélection", "Sélectionnez d'abord une courbe dans la liste.")
            return
        item = items[0]
        id_courbe = item.data(Qt.UserRole)
        couleur = QColorDialog.getColor()
        if couleur.isValid():
            self.courbes[id_courbe]["color"] = couleur.name()
            self._redessiner()

    def _on_item_change(self, item):
        # Cocher/decocher OU renommer declenchent tous les deux ce signal :
        # dans les deux cas, un simple redessin suffit (le nom affiche
        # vient directement de item.text()).
        self._redessiner()

    # ------------------------------------------------------------------
    #  Tracé
    # ------------------------------------------------------------------
    def _redessiner(self):
        ax = self.canvas.axes
        ax.clear()

        for i in range(self.liste_courbes.count()):
            item = self.liste_courbes.item(i)
            if item.checkState() != Qt.Checked:
                continue
            id_courbe = item.data(Qt.UserRole)
            courbe = self.courbes.get(id_courbe)
            if courbe is None:
                continue
            ax.plot(
                courbe["E"], courbe["mu_rho"],
                label=item.text(), color=courbe["color"], linewidth=2.2
            )

        if self.case_log_y.isChecked():
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        if self.case_limiter_x.isChecked():
            ax.set_xlim(self.spin_x_min.value(), self.spin_x_max.value())

        ax.set_xlabel("Énergie (keV)")
        ax.set_ylabel(r"$\mu/\rho$ (cm$^2$/g)")
        ax.set_title(self.champ_titre.text())
        if self.courbes:
            ax.legend(fontsize=16)
        ax.grid(True, which="both", alpha=0.3)
        self.canvas.draw()

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------
    def _exporter_png(self):
        if not self.courbes:
            QMessageBox.warning(self, "Rien à exporter", "Ajoutez d'abord au moins un fichier.")
            return
        chemin, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le graphique", "comparaison_mu_rho.png",
            "Image PNG (*.png)"
        )
        if not chemin:
            return
        try:
            self.canvas.fig.savefig(chemin, dpi=200)
            QMessageBox.information(self, "Export réussi", f"Graphique enregistré :\n{chemin}")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur - export", f"Impossible d'enregistrer le fichier :\n{exc}")


def main():
    app = QApplication(sys.argv)
    fenetre = FenetreComparaison()
    fenetre.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()