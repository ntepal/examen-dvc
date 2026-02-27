# src/data/split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Import relatif au projet pour le batch à partir de examen-dvc
# NB: export PYTHONPATH=$PYTHONPATH:. à mettre dans le .sh qui l'appelle
import src.data.utilitaires as utilitaires

def run_split():
    #--- Vérification de la présence des répertoires sinon création
    utilitaires.check_else_create_dirs()

    #--- Dataframe du csv
    df = pd.read_csv('data/raw/raw.csv')

    # Conversion et extraction de la date en colonnes numériques
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['Hour'] = df['date'].dt.hour

    # Suppression de la colonne date originale (texte)
    df = df.drop(columns=['date'])

    #--- Préparation
    stockage_dir = 'data/processed'

    # Même s'il est mentionné "variable cible est silica_concentrate et
    # se trouve dans la dernière colonne du dataset", il est plus judicieux
    # d'être explicite pour anticiper toute potentielle modification du dataset
    # Définition explicite de la cible
    target_column = 'silica_concentrate'

    #--- Dataframe pour variables et pour cible
    X = df.drop(columns=[target_column])
    y = df[target_column]

    #--- Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #--- Sauvegarde dans les csv en excluant la colonne d'indexation
    X_train.to_csv(f'{stockage_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{stockage_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{stockage_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{stockage_dir}/y_test.csv', index=False)

    print(f"✅ Données divisées avec succès dans {stockage_dir}")

if __name__ == "__main__":
    run_split()
