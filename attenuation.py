"""
attenuation.py
==============
Moteur GENERIQUE d'attenuation spectrale. Fonctionne pour n'importe quelle
source (continue ou discrete) : c'est la meme physique (Beer-Lambert a
chaque energie), seule l'INTEGRATION du flux change selon le mode.

  mode = "continu" -> integrale (trapezes)      : le spectre est une densite
  mode = "discret" -> somme ponderee des raies   : chaque raie est un poids

Utilise composite_physics.py pour le calcul du LAC du materiau.
"""

import numpy as np
from scipy.optimize import brentq
from composite_physics import LAC_composite, densite_composite


def _flux_integre(E_array, I_array, mode):
    """Flux total, integre selon le mode physique du spectre."""
    if mode == "continu":
        return np.trapezoid(I_array, E_array)
    elif mode == "discret":
        return np.sum(I_array)
    else:
        raise ValueError(f"mode inconnu : {mode} (attendu 'continu' ou 'discret')")


def spectre_transmis(E_array, I_array, w_frac_charge, epaisseur_cm,
                      nom_charge="Bi2O3", nom_matrice="PEEK"):
    """
    Beer-Lambert applique a CHAQUE energie du spectre (continu ou discret) :
    I_transmis(E) = I_incident(E) * exp(-mu(E) * x)
    """
    mu_E = np.array([
        LAC_composite(E, w_frac_charge, nom_charge=nom_charge, nom_matrice=nom_matrice)
        for E in E_array
    ])
    I_transmis = I_array * np.exp(-mu_E * epaisseur_cm)
    return I_transmis, mu_E


def pourcentage_attenuation(E_array, I_array, mode, w_frac_charge, epaisseur_cm,
                             nom_charge="Bi2O3", nom_matrice="PEEK"):
    """
    % d'attenuation global du flux de photons, quel que soit le type de source.
    """
    I_transmis, mu_E = spectre_transmis(
        E_array, I_array, w_frac_charge, epaisseur_cm,
        nom_charge=nom_charge, nom_matrice=nom_matrice
    )
    flux_incident = _flux_integre(E_array, I_array, mode)
    flux_transmis = _flux_integre(E_array, I_transmis, mode)   
    pct = (1 - flux_transmis / flux_incident) * 100
    return pct, flux_incident, flux_transmis

# Ajout/Modification dans attenuation.py
def epaisseur_composite_equivalente_plomb(E_spec, I_spec, mode, w_frac, nom_charge, nom_matrice):
    """
    Calcule l'épaisseur composite nécessaire pour égaler 0.5 mm de plomb.
    """
    from composite_physics import LAC_composite
    
    # 1. Calculer l'atténuation cible avec 0.5 mm de Plomb (Pb)
    # On force une matrice neutre (ex: "PLA") et w_frac=1.0 pour simuler le Pb pur
    pct_cible, _, _ = pourcentage_attenuation(
        E_spec, I_spec, mode, w_frac_charge=1.0, epaisseur_cm=0.05, 
        nom_charge="Pb", nom_matrice="PLA"
    )
    
    # 2. Sécurité : Plafonnement à 99.9%
    pct_cible = min(pct_cible, 99.9999)
    
    # 3. Calcul de l'épaisseur composite
    x_cm = epaisseur_pour_blocage(
        E_spec, I_spec, mode, w_frac, 
        pourcentage_cible=pct_cible, x_min=0.0001, x_max=5.0,
        nom_charge=nom_charge, nom_matrice=nom_matrice
    )
    return pct_cible, x_cm * 10 # Retourne l'épaisseur en mm



def epaisseur_pour_blocage(E_array, I_array, mode, w_frac_charge,
                            pourcentage_cible=99.0, x_min=1e-5, x_max=5.0,
                            nom_charge="Bi2O3", nom_matrice="PEEK"):
    """
    Bissection (brentq) : epaisseur (cm) necessaire pour atteindre le
    % d'attenuation cible, valable pour n'importe quel type de source.
    """
    def f(x):
        pct, _, _ = pourcentage_attenuation(
            E_array, I_array, mode, w_frac_charge, x,
            nom_charge=nom_charge, nom_matrice=nom_matrice
        )
        return pct - pourcentage_cible

    return brentq(f, x_min, x_max)

"""
chemin_fichier = r"F:\code_BIM_Sabrina\BIM_radioprotection\spekcalc\40kVp 30deg 1000Air 0.8Be 1.2Al 0Cu 0Sn 0W 0Ta 0Ti 0C 0Wa.spec"
data = np.loadtxt(chemin_fichier, skiprows=18)
E_array, I_array = data[:, 0], data[:, 1]
idx = np.argsort(E_array)

kvp40 = epaisseur_composite_equivalente_plomb(E_array, I_array, "continu", 0.80, "W", "TPU")
"""




# =====================================================================
# MU/RHO EFFECTIF (coefficient d'attenuation massique effectif)
# =====================================================================
def mu_effectif(E_array, I_array, mode, w_frac_charge, epaisseur_cm,
                nom_charge="Bi2O3", nom_matrice="PEEK"):
    """
    Coefficient d'attenuation lineaire EFFECTIF (cm^-1), obtenu en
    inversant Beer-Lambert sur le flux total transmis (et non sur une
    seule energie) :

        T = flux_transmis / flux_incident
        mu_eff = -ln(T) / x

    C'est la quantite qu'on mesurerait experimentalement en placant un
    detecteur "somme tout le flux" derriere une epaisseur x du composite.
    """
    pct, flux_inc, flux_trans = pourcentage_attenuation(
        E_array, I_array, mode, w_frac_charge, epaisseur_cm,
        nom_charge=nom_charge, nom_matrice=nom_matrice
    )
    T = flux_trans / flux_inc
    return -np.log(T) / epaisseur_cm


def mu_rho_effectif(E_array, I_array, mode, w_frac_charge, epaisseur_cm,
                     nom_charge="Bi2O3", nom_matrice="PEEK"):
    """
    (mu/rho)_effectif (cm^2/g) = mu_effectif / rho_composite.

    IMPORTANT (physique du "beam hardening") :
    -------------------------------------------
    Pour une source MONOENERGETIQUE (une seule raie), mu_eff est rigoureusement
    constant et egal au LAC(E) reel, quelle que soit l'epaisseur x -> le
    (mu/rho)_eff calcule ici sera identique peu importe x.

    Pour une source POLYENERGETIQUE (plusieurs raies ou spectre continu, ex :
    Pd-103 qui emet a plusieurs energies, ou un spectre SpekCalc), le spectre
    transmis se "durcit" en traversant la matiere : les photons de basse
    energie (mu eleve) sont attenues preferentiellement, donc le spectre
    restant est de plus en plus "dur" (energie moyenne plus elevee) a mesure
    que x augmente. Consequence : mu_eff DIMINUE legerement quand x augmente.
    Ce n'est PAS un bug -- c'est le meme phenomene qui explique le "beam
    hardening" en radiologie. Si votre spectre est tres etroit (quasi
    monoenergetique), la variation sera negligeable ; si les raies sont
    tres espacees en energie, la variation sera plus visible.

    Retourne (mu_eff, mu_rho_eff, rho_composite).
    """
    mu_eff = mu_effectif(
        E_array, I_array, mode, w_frac_charge, epaisseur_cm,
        nom_charge=nom_charge, nom_matrice=nom_matrice
    )
    rho = densite_composite(w_frac_charge, nom_charge=nom_charge, nom_matrice=nom_matrice)
    mu_rho_eff = mu_eff / rho
    return mu_eff, mu_rho_eff, rho


# =====================================================================
# GRAPHIQUES 
# =====================================================================
def graphique_spectre_incident(E_array, I_array, mode):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    if mode == "continu":
        plt.plot(E_array, I_array, color="tab:blue")
        plt.fill_between(E_array, I_array, alpha=0.2, color="tab:blue")
    else:
        plt.stem(E_array, I_array, basefmt=" ")
    plt.xlabel("Energie (keV)")
    plt.ylabel("Intensite")
    plt.title("Spectre incident")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def graphique_spectre_transmis(E_array, I_array, mode, w_frac_charge, epaisseur_cm,
                                nom_charge="Bi2O3", nom_matrice="PEEK"):
    import matplotlib.pyplot as plt
    I_transmis, mu_E = spectre_transmis(
        E_array, I_array, w_frac_charge, epaisseur_cm,
        nom_charge=nom_charge, nom_matrice=nom_matrice
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    if mode == "continu":
        ax1.plot(E_array, I_array, label="Incident", color="tab:blue")
        ax1.plot(E_array, I_transmis, label="Transmis", color="tab:red")
        ax1.fill_between(E_array, I_array, alpha=0.15, color="tab:blue")
        ax1.fill_between(E_array, I_transmis, alpha=0.15, color="tab:red")
    else:
        largeur = (E_array.max() - E_array.min()) * 0.02 if len(E_array) > 1 else 0.3
        ax1.bar(E_array - largeur/2, I_array, width=largeur, label="Incident", color="tab:blue")
        ax1.bar(E_array + largeur/2, I_transmis, width=largeur, label="Transmis", color="tab:red")

    ax1.set_ylabel("Intensite")
    ax1.set_title(f"Spectre incident vs transmis (x = {epaisseur_cm*10:.4f} mm)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(E_array, mu_E, "o-" if mode == "discret" else "-", color="tab:purple")
    ax2.set_xlabel("Energie (keV)")
    ax2.set_ylabel(r"LAC (cm$^{-1}$)")
    ax2.set_yscale("log")
    ax2.set_title("LAC applique a chaque energie du spectre")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


def graphique_convergence_epaisseur(E_array, I_array, mode, w_frac_charge,
                                     x_solution, pourcentage_cible,
                                     nom_charge="Bi2O3", nom_matrice="PEEK"):
    import matplotlib.pyplot as plt
    x_range = np.linspace(1e-5, max(x_solution * 2, 0.02), 150)
    pct_range = [
        pourcentage_attenuation(E_array, I_array, mode, w_frac_charge, x,
                                 nom_charge=nom_charge, nom_matrice=nom_matrice)[0]
        for x in x_range
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(x_range * 10, pct_range, color="tab:green")
    plt.axhline(pourcentage_cible, color="gray", linestyle="--", linewidth=0.8,
                label=f"Cible = {pourcentage_cible:.2f}%")
    plt.axvline(x_solution * 10, color="tab:red", linestyle="--", linewidth=0.8,
                label=f"Solution = {x_solution*10:.4f} mm")
    plt.xlabel("Epaisseur (mm)")
    plt.ylabel("% attenuation")
    plt.title("Convergence de la bissection (brentq)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def graphique_mu_rho_effectif(E_array, I_array, mode, w_frac_charge,
                               epaisseur_max_cm=1.0, n_points=150,
                               nom_charge="Bi2O3", nom_matrice="PEEK"):
    """
    Trace (mu/rho)_eff en fonction de l'epaisseur, pour VERIFIER visuellement
    s'il reste constant (source monoenergetique) ou s'il varie legerement
    (beam hardening, source polyenergetique). Voir docstring de
    mu_rho_effectif() pour l'explication physique.
    """
    import matplotlib.pyplot as plt
    x_range_cm = np.linspace(epaisseur_max_cm / n_points, epaisseur_max_cm, n_points)
    mu_rho_vals = [
        mu_rho_effectif(E_array, I_array, mode, w_frac_charge, x,
                         nom_charge=nom_charge, nom_matrice=nom_matrice)[1]
        for x in x_range_cm
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(x_range_cm * 10, mu_rho_vals, color="tab:brown", linewidth=2)
    plt.xlabel("Epaisseur (mm)")
    plt.ylabel(r"$(\mu/\rho)_{eff}$ (cm$^2$/g)")
    plt.title("mu/rho effectif vs epaisseur")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()