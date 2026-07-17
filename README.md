# Simulateur d'atténuation par rayons X pour composites matrice/charge radio-opaque

## Contexte

Ce projet a été développé dans le cadre d'un stage de recherche en radioprotection, visant à évaluer la capacité de **filaments composites 3D imprimables** à atténuer un faisceau de rayons X, pour une application en **plaques épisclérales de curiethérapie** (traitement du mélanome uvéal).

Le principe : imprimer en 3D des composants (ex. plaques, coques de protection) à partir d'une **matrice polymère** (PEEK, TPU, etc.) chargée en **particules radio-opaques** (Bi₂O₃, tungstène, etc.). Plus la charge est dense en électrons, plus elle atténue les rayons X, mais le défi est de trouver le bon compromis entre :
- pourcentage massique de charge (dispersion, imprimabilité, densité)
- épaisseur de matériau disponible
- énergie du faisceau utilisé (diagnostique, thérapeutique, curiethérapie)

L'application permet de **simuler numériquement** cette atténuation avant même de fabriquer et tester physiquement un échantillon, afin d'orienter les choix de formulation (fraction massique, matrice, charge, épaisseur).

## Principe physique

### Loi d'atténuation (Beer-Lambert)

L'atténuation d'un faisceau de rayons X traversant un matériau d'épaisseur `x` suit :

```
I(x) = I₀ · exp(−(μ/ρ) · ρ · x)
```

où :
- `μ/ρ` est le **coefficient d'atténuation massique** (cm²/g), qui dépend de l'énergie du photon et de la composition chimique du matériau (données obtenus sur le site du [NIST](https://physics.nist.gov/PhysRefData/XrayMassCoef/tab3.html))
- `ρ` est la densité du matériau (g/cm³)
- `x` est l'épaisseur traversée (cm)

Le **pourcentage d'atténuation** rapporté dans l'application se calcule comme :

```
% atténuation = (1 − I(x)/I₀) × 100
```

### Composite matrice + charge

Pour un composite, le coefficient massique effectif est obtenu par une **loi de mélange pondérée** entre les coefficients massiques de la matrice pure et de la charge pure, selon leurs fractions massiques respectives (`w` pour la charge, `1 − w` pour la matrice) :

```
(μ/ρ)_composite = w · (μ/ρ)_charge + (1 − w) · (μ/ρ)_matrice
```

La densité du composite (nécessaire pour convertir `μ/ρ` en `μ`) est aussi estimée à partir des fractions massiques et des densités des constituants purs.

### Spectre polyénergétique

Un faisceau de rayons X réel n'est pas monoénergétique : il est composé d'un **spectre** de photons à différentes énergies (`E_array`) pondéré par une intensité relative (`I_array`), typiquement issu d'un tube à rayons X (spectre de bremsstrahlung + raies caractéristiques). L'atténuation totale est donc calculée en intégrant l'effet d'atténuation sur l'ensemble du spectre, plutôt qu'à une seule énergie.

### Seuils d'absorption K (K-edges)

Les courbes de `μ/ρ` présentent des discontinuités abruptes aux **énergies de seuil K** des éléments lourds, des sauts brusques d'atténuation quand l'énergie du photon devient suffisante pour éjecter un électron de la couche K de l'atome. Ces seuils sont visibles et marqués dans les graphiques pour :
- **Bi₂O₃** : ~13.4, 15.7 et 16.4 keV (sous-couches L du bismuth, proches en énergie)
- **W (tungstène)** : ~10.2, 11.5 et 12.1 keV

Ces seuils sont importants en curiethérapie car ils déterminent des plages d'énergie où le matériau devient soudainement beaucoup plus (ou moins) protecteur.


### Paramètres ajustables
- **Nature de la charge** (ex. Bi₂O₃, W) et **de la matrice** (ex. PEEK, TPU)
- **Fraction massique** de la charge dans le composite
- **Épaisseur** du matériau
- **Mode de calcul** (monoénergétique ou spectral, selon le contexte simulé)
- **Équivalence plomb** épaisseur de composite équivalente à 0.5 mm de plomb



## Utilisation

Afin d'ouvrir l'application, exécuter `APP.py`.

### Ajouter une source de radioisotope
Ajouter l'information nécessaire dans `sources.py`, puis importer le nom de la source dans `APP.py` à la ligne 28.

### Ajouter un spectre Spekcalc
Insérer un fichier `.spec` dans le dossier `spekcalc`.

### Ajouter une nouvelle charge ou matrice de composite
Ajouter l'information nécessaire dans `composite_physics.py`, puis ajouter le matériau dans la liste :
- ligne 119 pour une charge
- ligne 123 pour une matrice

## Contact

Pour toute question : sabrina.demers.6@ulaval.ca
