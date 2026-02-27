# src/data/utilitaires.py
# Script utililtaires

import pandas as pd
import os

def check_else_create_dirs():
    """Crée les répertoires nécessaires s'ils n'existent pas."""

    # Liste basée sur l'arborescence présentée dans l'énoncé
    PATHS = [
        'data/raw',
        'data/processed',
        'models',
        'metrics',
        'src/data',
        'src/models'
    ]
    for path in PATHS:
        os.makedirs(path, exist_ok=True)
        # NB: Git ignore les dossiers vides, par défaut, il ne suit que les fichiers, pas les dossiers.
        # Création d'un .gitkeep pour que Git suive les dossiers vides au début
        with open(os.path.join(path, '.gitkeep'), 'a'):
            pass

def load_processed_data(subset='train', scaled=False):
    """Charge les données X et y selon le subset et l'état de normalisation."""

    #--- Préparation
    stockage_dir = 'data/processed'
    suffix = "_scaled" if scaled else ""

    X = pd.read_csv(f'{stockage_dir}/X_{subset}{suffix}.csv')

    # .squeeze("columns") transforme le DataFrame 1-colonne en Série
    # On retrouve aiinsi exactement le format post-split !
    y = pd.read_csv(f'{stockage_dir}/y_{subset}.csv').squeeze("columns")

    return X, y
