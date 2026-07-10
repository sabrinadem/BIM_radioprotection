"""
sources.py
==========
Module des SOURCES de rayonnement. Chaque fonction retourne :
    E_array, I_array, mode

- E_array : energies (keV)
- I_array : intensites (u.a. pour continu, % par desintegration pour discret)
- mode    : "continu"  -> spectre dense (ex: SpekCalc), integration = trapezes
            "discret"  -> raies isolees (ex: Pd-103), integration = somme ponderee

C'est ce "mode" qui permettra plus tard, dans PyQt, de proposer un menu
deroulant "type de source" sans dupliquer le moteur d'attenuation.
"""

import numpy as np


def charger_spectre_fichier(chemin_fichier):
    """
    Spectre CONTINU a partir d'un fichier 2 colonnes (Energie keV, Intensite u.a.)
    Ex: sortie de SpekCalc pour un tube a rayons X.
    """
    data = np.loadtxt(chemin_fichier, skiprows=18)
    E_array, I_array = data[:, 0], data[:, 1]
    idx = np.argsort(E_array)
    return E_array[idx], I_array[idx], "continu"


# ---------------------------------------------------------------------
# Sources DISCRETES (radio-isotopes de curietherapie)
# ---------------------------------------------------------------------

# (!) A RE-VERIFIER sur NNDC/ENSDF (nndc.bnl.gov/nudat3) avant publication.
# Seules les intensites RELATIVES comptent pour le calcul d'attenuation.
Pd103_lignes = [
    (20.074, 14.1),   # Rh Kalpha2
    (20.216, 25.4),   # Rh Kalpha1
    (22.699, 4.9),    # Rh Kbeta1
    (23.179, 1.0),    # Rh Kbeta2
    (39.75,  0.065),  # gamma (Rh-103m), quasi negligeable
]

def spectre_Pd103():
    """Pd-103 : rayons X caracteristiques du Rh (capture electronique)."""
    E = np.array([e for e, _ in Pd103_lignes])
    I = np.array([i for _, i in Pd103_lignes])
    idx = np.argsort(E)
    return E[idx], I[idx], "discret"


Pd103mono_lignes = [
    (21, 100),   # Rh Kalpha1
]

def spectre_Pd103mono():
    E = np.array([e for e, _ in Pd103mono_lignes])
    I = np.array([i for _, i in Pd103mono_lignes])
    idx = np.argsort(E)
    return E[idx], I[idx], "discret"

I125_lignes = [
     (27.4, ...),
     (31.0, ...),
     (35.5, ...),
 ]

def spectre_I125():
     E = np.array([e for e, _ in I125_lignes])
     I = np.array([i for _, i in I125_lignes])
     idx = np.argsort(E)
     return E[idx], I[idx], "discret"


SOURCES_DISPONIBLES = {
    "Pd-103": spectre_Pd103,
    "Pd-103mono": spectre_Pd103mono,
    "I-125": spectre_I125,
    # "Fichier SpekCalc": charger_spectre_fichier,  # necessite un argument, gere a part
}