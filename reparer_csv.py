import pandas as pd

# Les données propres
data = [
    ["2023-01-05", "Magasin_A", "Produit_1", 120],
    ["2023-01-12", "Magasin_A", "Produit_1", 150],
    ["2023-02-03", "Magasin_A", "Produit_1", 180],
    ["2023-03-01", "Magasin_A", "Produit_1", 210],
    ["2023-04-05", "Magasin_A", "Produit_1", 230],
    ["2023-05-02", "Magasin_A", "Produit_1", 240],
    ["2023-06-01", "Magasin_A", "Produit_1", 270], # Mois 6
    ["2023-07-01", "Magasin_A", "Produit_1", 300], # Mois 7
    ["2023-08-01", "Magasin_A", "Produit_1", 320], # Mois 8
]

# Création du DataFrame
df = pd.DataFrame(data, columns=['date', 'store', 'product', 'sales'])

# Sauvegarde SANS guillemets et avec le bon séparateur
df.to_csv('data_propre.csv', index=False)

print("✅ Fichier 'data_propre.csv' créé avec succès !")
