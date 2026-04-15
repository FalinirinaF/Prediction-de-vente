import pandas as pd
from sqlalchemy import create_engine

# 1. Vos données sous forme de liste
data = [
    ["2023-01-05", "Magasin_A", "Produit_1", 120],
    ["2023-01-12", "Magasin_A", "Produit_1", 150],
    ["2023-02-03", "Magasin_A", "Produit_1", 180],
    ["2023-03-01", "Magasin_A", "Produit_1", 210],
    ["2023-04-05", "Magasin_A", "Produit_1", 230],
    ["2023-05-02", "Magasin_A", "Produit_1", 240],
    ["2023-06-01", "Magasin_A", "Produit_1", 270],
    ["2023-07-01", "Magasin_A", "Produit_1", 300],
    ["2023-08-01", "Magasin_A", "Produit_1", 320],
]

# 2. Création d'un DataFrame (Tableau virtuel)
# Attention : les colonnes doivent avoir les mêmes noms que dans votre table SQL
df = pd.DataFrame(data, columns=['date', 'store', 'product', 'sales'])

# 3. Connexion à PostgreSQL
# Remplacez 'Faly0007' par votre vrai mot de passe
engine = create_engine('postgresql://postgres:Faly0007@localhost:5432/pred_vente')

try:
    # 4. Envoi des données vers la table 'ventes'
    # if_exists='append' permet d'ajouter les lignes sans effacer celles déjà présentes
    df.to_sql('ventes', engine, if_exists='append', index=False)
    print("✅ Données insérées avec succès dans la table 'ventes' !")
except Exception as e:
    print(f"❌ Erreur lors de l'insertion : {e}")
