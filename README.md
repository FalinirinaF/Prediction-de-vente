# Prediction-de-vente
git clone https://github.com/FalinirinaF/Prediction-de-vente.git
cd Prediction-de-vente

python3 -m venv venv
source venv/bin/activate

# Mise à jour de pip
pip install --upgrade pip

# Installation des bibliothèques de base et ML
pip install streamlit pandas numpy matplotlib scikit-learn xgboost sqlalchemy psycopg2-binary

# Installation de la version stable de TensorFlow pour CPU
pip install tensorflow-cpu
streamlit run app.py
