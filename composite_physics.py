"""
composite_physics.py
=====================
Module COMMUN : tout ce qui concerne le MATERIAU (composite charge/matrice),
independant de la source de rayonnement utilisee.

Regroupe les etapes 1 a 4 de votre pipeline original :
  1. Donnees NIST mu/rho des elements
  2. Interpolation log-log
  3. Regle de melange (charge, matrice, composite)
  4. Densite et LAC du composite

Ce module ne sait rien des spectres (fichier SpekCalc, Pd-103, etc.) :
ca, c'est le role de sources.py

------------------------------------------------------------------------
CHARGES DISPONIBLES : "Bi2O3" (oxyde de bismuth), "W" (tungstene pur), "Pb" (plomb pur)
MATRICES DISPONIBLES : "PEEK", "TPU"
------------------------------------------------------------------------
"""

import numpy as np

# =====================================================================
# 1. DONNEES NIST : mu/rho (cm2/g) vs Energie (keV)
#    Source : https://physics.nist.gov/PhysRefData/XrayMassCoef/
# =====================================================================

# --- Bismuth (Z=83) : decoupe en segments a cause des seuils L3/L2/L1 --
Bi_segments = [
    (4.00, 13.4186, [(4.00, 1.296e3), (5.00, 7.580e2), (6.00, 4.855e2),
                      (8.00, 2.378e2), (10.00, 1.360e2), (13.4186, 6.491e1)]),
    (13.4186, 15.7111, [(13.4186, 1.560e2), (15.00, 1.160e2), (15.7111, 1.027e2)]),
    (15.7111, 16.3875, [(15.7111, 1.416e2), (16.0457, 1.351e2), (16.3875, 1.282e2)]),
    (16.3875, 60.00, [(16.3875, 1.478e2), (20.00, 8.952e1), (30.00, 3.152e1),
                       (40.00, 1.495e1), (50.00, 8.379e0), (60.00, 5.233e0)]),
]

# --- Plomb (Z=82) : donnees NIST officielles completes -----------------
# Source : NIST XCOM / X-Ray Mass Attenuation Coefficients (physics.nist.gov)
# Seuils : M5=2.48400, M4=2.58560, M3=3.06640, M2=3.55420, M1=3.85070,
#          L3=13.0352, L2=15.2000, L1=15.8608, K=88.0045 keV
# Plage : 1 keV -> 20 MeV.
Pb_segments = [
    (1.00000, 2.48400, [(1.00000, 5.210e3), (1.50000, 2.356e3), (2.00000, 1.285e3),
                          (2.48400, 8.006e2)]),
    (2.48400, 2.58560, [(2.48400, 1.397e3), (2.53429, 1.726e3), (2.58560, 1.944e3)]),
    (2.58560, 3.06640, [(2.58560, 2.458e3), (3.00000, 1.965e3), (3.06640, 1.857e3)]),
    (3.06640, 3.55420, [(3.06640, 2.146e3), (3.30130, 1.796e3), (3.55420, 1.496e3)]),
    (3.55420, 3.85070, [(3.55420, 1.585e3), (3.69948, 1.442e3), (3.85070, 1.311e3)]),
    (3.85070, 13.0352, [(3.85070, 1.368e3), (4.00000, 1.251e3), (5.00000, 7.304e2),
                          (6.00000, 4.672e2), (8.00000, 2.287e2), (10.00000, 1.306e2),
                          (13.0352, 6.701e1)]),
    (13.0352, 15.2000, [(13.0352, 1.621e2), (15.00000, 1.116e2), (15.2000, 1.078e2)]),
    (15.2000, 15.8608, [(15.2000, 1.485e2), (15.5269, 1.416e2), (15.8608, 1.344e2)]),
    (15.8608, 88.0045, [(15.8608, 1.548e2), (20.00000, 8.636e1), (30.00000, 3.032e1),
                          (40.00000, 1.436e1), (50.00000, 8.041e0), (60.00000, 5.021e0),
                          (80.00000, 2.419e0), (88.0045, 1.910e0)]),
    (88.0045, 20000.00, [(88.0045, 7.683e0), (100.00000, 5.549e0), (150.00000, 2.014e0),
                          (200.00000, 9.985e-1), (300.00000, 4.031e-1), (400.00000, 2.323e-1),
                          (500.00000, 1.614e-1), (600.00000, 1.248e-1), (800.00000, 8.870e-2),
                          (1000.00000, 7.102e-2), (1250.00000, 5.876e-2), (1500.00000, 5.222e-2),
                          (2000.00000, 4.606e-2), (3000.00000, 4.234e-2), (4000.00000, 4.197e-2),
                          (5000.00000, 4.272e-2), (6000.00000, 4.391e-2), (8000.00000, 4.675e-2),
                          (10000.00000, 4.972e-2), (15000.00000, 5.658e-2), (20000.00000, 6.206e-2)]),
]

# --- Oxygene, Carbone, Hydrogene : donnees NIST verifiees --------------
O_data = [(4.00, 9.315e1), (5.00, 4.790e1), (6.00, 2.770e1), (8.00, 1.163e1),
          (10.00, 5.952e0), (15.00, 1.836e0), (20.00, 8.651e-1), (30.00, 3.779e-1),
          (40.00, 2.585e-1), (50.00, 2.132e-1), (60.00, 1.907e-1)]

C_data = [(4.00, 3.778e1), (5.00, 1.912e1), (6.00, 1.095e1), (8.00, 4.576e0),
          (10.00, 2.373e0), (15.00, 8.071e-1), (20.00, 4.420e-1), (30.00, 2.562e-1),
          (40.00, 2.076e-1), (50.00, 1.871e-1), (60.00, 1.753e-1)]

H_data = [(4.00, 4.546e-1), (5.00, 4.193e-1), (6.00, 4.042e-1), (8.00, 3.914e-1),
          (10.00, 3.854e-1), (15.00, 3.764e-1), (20.00, 3.695e-1), (30.00, 3.570e-1),
          (40.00, 3.458e-1), (50.00, 3.355e-1), (60.00, 3.260e-1)]

# --- Tungstene (Z=74) : donnees NIST officielles completes -------------
# Source : NIST XCOM / X-Ray Mass Attenuation Coefficients (physics.nist.gov)
# Seuils : M5=1.80920, M4=1.87160, M3=2.28100, M2=2.57490, M1=2.81960,
#          L3=10.2068, L2=11.5440, L1=12.0998, K=69.5250 keV
# Plage : 1 keV -> 20 MeV. Remplace l'ancienne approximation 4-60 keV.
W_segments = [
    (1.00000, 1.80920, [(1.00000, 3.683e3), (1.50000, 1.643e3), (1.80920, 1.108e3)]),
    (1.80920, 1.87160, [(1.80920, 1.327e3), (1.84014, 1.911e3), (1.87160, 2.901e3)]),
    (1.87160, 2.28100, [(1.87160, 3.170e3), (2.00000, 3.922e3), (2.28100, 2.828e3)]),
    (2.28100, 2.57490, [(2.28100, 3.279e3), (2.42350, 2.833e3), (2.57490, 2.445e3)]),
    (2.57490, 2.81960, [(2.57490, 2.599e3), (2.69447, 2.339e3), (2.81960, 2.104e3)]),
    (2.81960, 10.2068, [(2.81960, 2.194e3), (3.00000, 1.902e3), (4.00000, 9.564e2),
                          (5.00000, 5.534e2), (6.00000, 3.514e2), (8.00000, 1.705e2),
                          (10.00000, 9.691e1), (10.2068, 9.201e1)]),
    (10.2068, 11.5440, [(10.2068, 2.334e2), (10.8548, 1.983e2), (11.5440, 1.689e2)]),
    (11.5440, 12.0998, [(11.5440, 2.312e2), (11.8186, 2.268e2), (12.0998, 2.065e2)]),
    (12.0998, 69.5250, [(12.0998, 2.382e2), (15.00000, 1.389e2), (20.00000, 6.573e1),
                          (30.00000, 2.273e1), (40.00000, 1.067e1), (50.00000, 5.949e0),
                          (60.00000, 3.713e0), (69.5250, 2.552e0)]),
    (69.5250, 20000.00, [(69.5250, 1.123e1), (80.00000, 7.810e0), (100.00000, 4.438e0),
                          (150.00000, 1.581e0), (200.00000, 7.844e-1), (300.00000, 3.238e-1),
                          (400.00000, 1.925e-1), (500.00000, 1.378e-1), (600.00000, 1.093e-1),
                          (800.00000, 8.066e-2), (1000.00000, 6.618e-2), (1250.00000, 5.577e-2),
                          (1500.00000, 5.000e-2), (2000.00000, 4.433e-2), (3000.00000, 4.075e-2),
                          (4000.00000, 4.038e-2), (5000.00000, 4.103e-2), (6000.00000, 4.210e-2),
                          (8000.00000, 4.472e-2), (10000.00000, 4.747e-2), (15000.00000, 5.384e-2),
                          (20000.00000, 5.893e-2)]),
]

# --- Azote (Z=7) : donnees NIST officielles completes ------------------
# Source : NIST XCOM / X-Ray Mass Attenuation Coefficients (physics.nist.gov)
# Pas de seuil d'absorption dans cette plage (K de l'azote ~ 0.4 keV, hors plage).
# Plage : 1 keV -> 20 MeV. Remplace l'ancien placeholder approximatif.
N_data = [(1.00, 3.311e3), (1.50, 1.083e3), (2.00, 4.769e2), (3.00, 1.456e2),
          (4.00, 6.166e1), (5.00, 3.144e1), (6.00, 1.809e1), (8.00, 7.562e0),
          (10.00, 3.879e0), (15.00, 1.236e0), (20.00, 6.178e-1), (30.00, 3.066e-1),
          (40.00, 2.288e-1), (50.00, 1.980e-1), (60.00, 1.817e-1), (80.00, 1.639e-1),
          (100.00, 1.529e-1), (150.00, 1.353e-1), (200.00, 1.233e-1), (300.00, 1.068e-1),
          (400.00, 9.557e-2), (500.00, 8.719e-2), (600.00, 8.063e-2), (800.00, 7.081e-2),
          (1000.00, 6.364e-2), (1250.00, 5.693e-2), (1500.00, 5.180e-2), (2000.00, 4.450e-2),
          (3000.00, 3.579e-2), (4000.00, 3.073e-2), (5000.00, 2.742e-2), (6000.00, 2.511e-2),
          (8000.00, 2.209e-2), (10000.00, 2.024e-2), (15000.00, 1.782e-2), (20000.00, 1.673e-2)]

M = {"Bi": 208.98038, "O": 15.999, "C": 12.011, "H": 1.008,
     "W": 183.84, "N": 14.007, "Pb": 207.2}

RHO_PEEK = 1.30
RHO_BI2O3 = 8.90
RHO_W = 19.25       # tungstene pur (element)
# ATTENTION - TPU : densite typique 1.10-1.25 g/cm3 selon le grade -> a ajuster
RHO_TPU = 1.20
RHO_PB = 11.35      # plomb pur (element)


# =====================================================================
# 1bis. DONNEES NIST : mu_en/rho (cm2/g), coefficient d'absorption
#       d'energie massique -> utile pour calculs de dose (radioprotection)
#       Meme structure/decoupage que les tables mu/rho ci-dessus.
# =====================================================================

W_muen_segments = [
    (1.00000, 1.80920, [(1.00000, 3.671e3), (1.50000, 1.632e3), (1.80920, 1.097e3)]),
    (1.80920, 1.87160, [(1.80920, 1.311e3), (1.84014, 1.883e3), (1.87160, 2.853e3)]),
    (1.87160, 2.28100, [(1.87160, 3.116e3), (2.00000, 3.853e3), (2.28100, 2.781e3)]),
    (2.28100, 2.57490, [(2.28100, 3.226e3), (2.42350, 2.786e3), (2.57490, 2.407e3)]),
    (2.57490, 2.81960, [(2.57490, 2.558e3), (2.69447, 2.301e3), (2.81960, 2.071e3)]),
    (2.81960, 10.2068, [(2.81960, 2.160e3), (3.00000, 1.873e3), (4.00000, 9.405e2),
                          (5.00000, 5.423e2), (6.00000, 3.428e2), (8.00000, 1.643e2),
                          (10.00000, 9.204e1), (10.2068, 8.724e1)]),
    (10.2068, 11.5440, [(10.2068, 1.966e2), (10.8548, 1.684e2), (11.5440, 1.444e2)]),
    (11.5440, 12.0998, [(11.5440, 1.889e2), (11.8186, 1.797e2), (12.0998, 1.699e2)]),
    (12.0998, 69.5250, [(12.0998, 1.948e2), (15.00000, 1.172e2), (20.00000, 5.697e1),
                          (30.00000, 1.991e1), (40.00000, 9.240e0), (50.00000, 5.050e0),
                          (60.00000, 3.070e0), (69.5250, 2.049e0)]),
    (69.5250, 20000.00, [(69.5250, 3.212e0), (80.00000, 2.879e0), (100.00000, 2.100e0),
                          (150.00000, 9.378e-1), (200.00000, 4.913e-1), (300.00000, 1.973e-1),
                          (400.00000, 1.100e-1), (500.00000, 7.440e-2), (600.00000, 5.673e-2),
                          (800.00000, 4.028e-2), (1000.00000, 3.276e-2), (1250.00000, 2.761e-2),
                          (1500.00000, 2.484e-2), (2000.00000, 2.256e-2), (3000.00000, 2.236e-2),
                          (4000.00000, 2.363e-2), (5000.00000, 2.510e-2), (6000.00000, 2.649e-2),
                          (8000.00000, 2.886e-2), (10000.00000, 3.072e-2), (15000.00000, 3.360e-2),
                          (20000.00000, 3.475e-2)]),
]

N_muen_data = [(1.00, 3.306e3), (1.50, 1.080e3), (2.00, 4.755e2), (3.00, 1.447e2),
               (4.00, 6.094e1), (5.00, 3.086e1), (6.00, 1.759e1), (8.00, 7.170e0),
               (10.00, 3.545e0), (15.00, 9.715e-1), (20.00, 3.867e-1), (30.00, 1.099e-1),
               (40.00, 5.051e-2), (50.00, 3.217e-2), (60.00, 2.548e-2), (80.00, 2.211e-2),
               (100.00, 2.231e-2), (150.00, 2.472e-2), (200.00, 2.665e-2), (300.00, 2.873e-2),
               (400.00, 2.952e-2), (500.00, 2.969e-2), (600.00, 2.956e-2), (800.00, 2.886e-2),
               (1000.00, 2.792e-2), (1250.00, 2.669e-2), (1500.00, 2.550e-2), (2000.00, 2.347e-2),
               (3000.00, 2.057e-2), (4000.00, 1.867e-2), (5000.00, 1.734e-2), (6000.00, 1.639e-2),
               (8000.00, 1.512e-2), (10000.00, 1.434e-2), (15000.00, 1.332e-2), (20000.00, 1.285e-2)]

# --- Plomb (Z=82) : mu_en/rho, memes seuils/breakpoints que Pb_segments -
Pb_muen_segments = [
    (1.00000, 2.48400, [(1.00000, 5.197e3), (1.50000, 2.344e3), (2.00000, 1.274e3),
                          (2.48400, 7.895e2)]),
    (2.48400, 2.58560, [(2.48400, 1.366e3), (2.53429, 1.682e3), (2.58560, 1.895e3)]),
    (2.58560, 3.06640, [(2.58560, 2.390e3), (3.00000, 1.913e3), (3.06640, 1.808e3)]),
    (3.06640, 3.55420, [(3.06640, 2.090e3), (3.30130, 1.748e3), (3.55420, 1.459e3)]),
    (3.55420, 3.85070, [(3.55420, 1.546e3), (3.69948, 1.405e3), (3.85070, 1.279e3)]),
    (3.85070, 13.0352, [(3.85070, 1.335e3), (4.00000, 1.221e3), (5.00000, 7.124e2),
                          (6.00000, 4.546e2), (8.00000, 2.207e2), (10.00000, 1.247e2),
                          (13.0352, 6.270e1)]),
    (13.0352, 15.2000, [(13.0352, 1.291e2), (15.00000, 9.100e1), (15.2000, 8.807e1)]),
    (15.2000, 15.8608, [(15.2000, 1.131e2), (15.5269, 1.083e2), (15.8608, 1.032e2)]),
    (15.8608, 88.0045, [(15.8608, 1.180e2), (20.00000, 6.899e1), (30.00000, 2.536e1),
                          (40.00000, 1.211e1), (50.00000, 6.740e0), (60.00000, 4.149e0),
                          (80.00000, 1.916e0), (88.0045, 1.482e0)]),
    (88.0045, 20000.00, [(88.0045, 2.160e0), (100.00000, 1.976e0), (150.00000, 1.056e0),
                          (200.00000, 5.870e-1), (300.00000, 2.455e-1), (400.00000, 1.370e-1),
                          (500.00000, 9.128e-2), (600.00000, 6.819e-2), (800.00000, 4.644e-2),
                          (1000.00000, 3.654e-2), (1250.00000, 2.988e-2), (1500.00000, 2.640e-2),
                          (2000.00000, 2.360e-2), (3000.00000, 2.322e-2), (4000.00000, 2.449e-2),
                          (5000.00000, 2.600e-2), (6000.00000, 2.744e-2), (8000.00000, 2.989e-2),
                          (10000.00000, 3.181e-2), (15000.00000, 3.478e-2), (20000.00000, 3.595e-2)]),
]


# =====================================================================
# 2. INTERPOLATION LOG-LOG
# =====================================================================
def _interp_loglog(E_keV, data_points):
    E_arr = np.array([p[0] for p in data_points])
    mr_arr = np.array([p[1] for p in data_points])
    log_E_query = np.log(E_keV)
    log_mr_query = np.interp(log_E_query, np.log(E_arr), np.log(mr_arr))
    return np.exp(log_mr_query)

def _interp_segments(E_keV, segments):
    """Interpolation log-log generique sur une liste de segments
    (E_min, E_max, [(E, valeur), ...]), avec extrapolation plate aux bornes."""
    for E_min, E_max, pts in segments:
        if E_min <= E_keV <= E_max:
            return _interp_loglog(E_keV, pts)
    if E_keV < segments[0][0]:
        return _interp_loglog(E_keV, segments[0][2])
    return _interp_loglog(E_keV, segments[-1][2])

def mu_rho_Bi(E_keV):
    for E_min, E_max, pts in Bi_segments:
        if E_min <= E_keV <= E_max:
            return _interp_loglog(E_keV, pts)
    if E_keV < Bi_segments[0][0]:
        return _interp_loglog(E_keV, Bi_segments[0][2])
    return _interp_loglog(E_keV, Bi_segments[-1][2])

def mu_rho_W(E_keV):
    """Donnees NIST officielles completes (1 keV - 20 MeV), seuils M5-K inclus."""
    return _interp_segments(E_keV, W_segments)

def mu_rho_O(E_keV):
    return _interp_loglog(E_keV, O_data)

def mu_rho_C(E_keV):
    return _interp_loglog(E_keV, C_data)

def mu_rho_H(E_keV):
    return _interp_loglog(E_keV, H_data)

def mu_rho_N(E_keV):
    """Donnees NIST officielles completes (1 keV - 20 MeV)."""
    return _interp_loglog(E_keV, N_data)

def mu_rho_Pb(E_keV):
    """Donnees NIST officielles completes (1 keV - 20 MeV), seuils M5-K inclus."""
    return _interp_segments(E_keV, Pb_segments)


# --- Coefficients d'absorption d'energie massique mu_en/rho (cm2/g) ----
# Disponibles pour l'instant pour N, W et Pb (donnees NIST fournies).
# Pour un calcul de dose complet sur le composite PEEK/Bi2O3, il manque
# encore Bi, O, C, H en mu_en/rho -> a ajouter si besoin (memes tableaux
# NIST, colonne "mu_en/rho").
def muen_rho_W(E_keV):
    """mu_en/rho du tungstene (cm2/g), donnees NIST 1 keV - 20 MeV."""
    return _interp_segments(E_keV, W_muen_segments)

def muen_rho_N(E_keV):
    """mu_en/rho de l'azote (cm2/g), donnees NIST 1 keV - 20 MeV."""
    return _interp_loglog(E_keV, N_muen_data)

def muen_rho_Pb(E_keV):
    """mu_en/rho du plomb (cm2/g), donnees NIST 1 keV - 20 MeV."""
    return _interp_segments(E_keV, Pb_muen_segments)


# Registre des elements disponibles pour composer charges/matrices (mu/rho)
ELEMENTS = {
    "Bi": mu_rho_Bi, "O": mu_rho_O, "C": mu_rho_C, "H": mu_rho_H,
    "W": mu_rho_W, "N": mu_rho_N, "Pb": mu_rho_Pb,
}

# Registre partiel pour mu_en/rho (voir note ci-dessus : Bi, O, C, H manquants)
MUEN_ELEMENTS = {
    "W": muen_rho_W, "N": muen_rho_N, "Pb": muen_rho_Pb,
}


# =====================================================================
# 3. REGLE DE MELANGE NIST : (mu/rho)_compose = somme(w_i * (mu/rho)_i)
# =====================================================================
def fractions_massiques(formule):
    M_tot = sum(n * M[el] for el, n in formule.items())
    return {el: (n * M[el]) / M_tot for el, n in formule.items()}

# --- Charges (element lourd) -------------------------------------------
w_Bi2O3 = fractions_massiques({"Bi": 2, "O": 3})
w_W = fractions_massiques({"W": 1})  # tungstene pur -> 100% W
w_Pb = fractions_massiques({"Pb": 1})  # plomb pur -> 100% Pb

CHARGES = {
    "Bi2O3": {"densite": RHO_BI2O3, "fractions": w_Bi2O3, "nom_affiche": "Bi2O3"},
    "W":     {"densite": RHO_W,     "fractions": w_W,     "nom_affiche": "Tungstène (W)"},
    "Pb":    {"densite": RHO_PB,    "fractions": w_Pb,    "nom_affiche": "Plomb (Pb)"},
}

# --- Matrices (polymere) ------------------------------------------------
w_PEEK = fractions_massiques({"C": 19, "H": 12, "O": 3})

# ATTENTION - Composition TPU APPROXIMATIVE - la formule/composition exacte
# varie beaucoup selon le grade (polyester vs polyether, durete Shore, etc.).
# Remplacer par les fractions massiques reelles de VOTRE TPU (fiche
# technique / analyse elementaire CHN) avant utilisation finale.
w_TPU = {"C": 0.63, "H": 0.08, "N": 0.04, "O": 0.25}

MATRICES = {
    "PEEK": {"densite": RHO_PEEK, "fractions": w_PEEK, "nom_affiche": "PEEK"},
    "TPU":  {"densite": RHO_TPU,  "fractions": w_TPU,  "nom_affiche": "TPU"},
}


def mu_rho_charge(E_keV, nom_charge="Bi2O3"):
    frac = CHARGES[nom_charge]["fractions"]
    return sum(w * ELEMENTS[el](E_keV) for el, w in frac.items())

def mu_rho_matrice(E_keV, nom_matrice="PEEK"):
    frac = MATRICES[nom_matrice]["fractions"]
    return sum(w * ELEMENTS[el](E_keV) for el, w in frac.items())

# --- Alias retro-compatibles (anciens noms utilises dans le reste du code) ---
def mu_rho_Bi2O3(E_keV):
    return mu_rho_charge(E_keV, "Bi2O3")

def mu_rho_PEEK(E_keV):
    return mu_rho_matrice(E_keV, "PEEK")

def mu_rho_composite(E_keV, w_frac_charge, nom_charge="Bi2O3", nom_matrice="PEEK"):
    return (w_frac_charge * mu_rho_charge(E_keV, nom_charge) +
            (1 - w_frac_charge) * mu_rho_matrice(E_keV, nom_matrice))


# =====================================================================
# 4. DENSITE ET LAC DU COMPOSITE
# =====================================================================
def densite_composite(w_frac_charge, nom_charge="Bi2O3", nom_matrice="PEEK"):
    rho_charge = CHARGES[nom_charge]["densite"]
    rho_matrice = MATRICES[nom_matrice]["densite"]
    inv_rho = (1 - w_frac_charge) / rho_matrice + w_frac_charge / rho_charge
    return 1 / inv_rho

def LAC_composite(E_keV, w_frac_charge, nom_charge="Bi2O3", nom_matrice="PEEK"):
    """Coefficient d'attenuation lineaire (cm^-1) a l'energie E_keV"""
    return (mu_rho_composite(E_keV, w_frac_charge, nom_charge, nom_matrice) *
            densite_composite(w_frac_charge, nom_charge, nom_matrice))


# =====================================================================
# GRAPHIQUES COMMUNS
# =====================================================================
def graphique_mu_rho_elements():
    import matplotlib.pyplot as plt
    E_plot = np.linspace(4, 60, 400)
    plt.figure(figsize=(8, 5))
    plt.plot(E_plot, [mu_rho_Bi(e) for e in E_plot], label="Bi", color="tab:red")
    plt.plot(E_plot, [mu_rho_Pb(e) for e in E_plot], label="Pb", color="tab:gray")
    plt.plot(E_plot, [mu_rho_W(e) for e in E_plot], label="W", color="tab:brown")
    plt.plot(E_plot, [mu_rho_O(e) for e in E_plot], label="O", color="tab:blue")
    plt.plot(E_plot, [mu_rho_C(e) for e in E_plot], label="C", color="tab:green")
    plt.plot(E_plot, [mu_rho_H(e) for e in E_plot], label="H", color="tab:orange")
    plt.plot(E_plot, [mu_rho_N(e) for e in E_plot], label="N", color="tab:cyan")
    for seuil in (13.4186, 15.7111, 16.3875, 10.207, 11.544, 12.100):
        plt.axvline(seuil, color="gray", linestyle="--", linewidth=0.6)
    plt.yscale("log")
    plt.xlabel("Energie (keV)")
    plt.ylabel(r"$\mu/\rho$ (cm$^2$/g)")
    plt.title("mu/rho des elements purs (seuils L en pointilles)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

def graphique_mu_rho_composite(w_frac_charge, nom_charge="Bi2O3", nom_matrice="PEEK"):
    import matplotlib.pyplot as plt
    E_plot = np.linspace(4, 60, 400)
    nom_charge_aff = CHARGES[nom_charge]["nom_affiche"]
    nom_matrice_aff = MATRICES[nom_matrice]["nom_affiche"]
    plt.figure(figsize=(8, 5))
    plt.plot(E_plot, [mu_rho_charge(e, nom_charge) for e in E_plot], "--",
             label=f"{nom_charge_aff} pur", color="tab:red")
    plt.plot(E_plot, [mu_rho_matrice(e, nom_matrice) for e in E_plot], "--",
             label=f"{nom_matrice_aff} pur", color="tab:blue")
    plt.plot(E_plot, [mu_rho_composite(e, w_frac_charge, nom_charge, nom_matrice) for e in E_plot],
             label=f"Composite ({w_frac_charge*100:.0f}% {nom_charge_aff})", color="black", linewidth=2)
    for seuil in (13.4186, 15.7111, 16.3875, 10.207, 11.544, 12.100):
        plt.axvline(seuil, color="gray", linestyle=":", linewidth=0.6)
    plt.yscale("log")
    plt.xlabel("Energie (keV)")
    plt.ylabel(r"$\mu/\rho$ (cm$^2$/g)")
    plt.title(f"mu/rho du composite ({nom_charge_aff}/{nom_matrice_aff}) vs phases pures")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

def graphique_LAC_composite(w_frac_charge, nom_charge="Bi2O3", nom_matrice="PEEK"):
    import matplotlib.pyplot as plt
    E_plot = np.linspace(4, 60, 400)
    nom_charge_aff = CHARGES[nom_charge]["nom_affiche"]
    plt.figure(figsize=(8, 5))
    plt.plot(E_plot, [LAC_composite(e, w_frac_charge, nom_charge, nom_matrice) for e in E_plot],
             color="tab:purple", linewidth=2)
    for seuil in (13.4186, 15.7111, 16.3875, 10.207, 11.544, 12.100):
        plt.axvline(seuil, color="gray", linestyle="--", linewidth=0.6)
    plt.yscale("log")
    plt.xlabel("Energie (keV)")
    plt.ylabel(r"LAC (cm$^{-1}$)")
    plt.title(f"LAC du composite ({w_frac_charge*100:.0f}% {nom_charge_aff}, "
              f"rho={densite_composite(w_frac_charge, nom_charge, nom_matrice):.2f} g/cm3)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()