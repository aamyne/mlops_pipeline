# Définition des variables
PYTHON = python
PIP = pip
REQ_FILE = requirements.txt
MODEL_DIR = artifacts
TRAIN_DATA = churn-bigml-80.csv
TEST_DATA = churn-bigml-20.csv

# Installation des dépendances
install:
	$(PIP) install -r $(REQ_FILE)

# Vérification de la qualité du code
lint:
	flake8 --max-line-length=100 data_processing.py main.py model_evaluation.py model_persistence.py model_training.py

# Formatage du code (auto-correction)
format:
	black data_processing.py main.py model_evaluation.py model_persistence.py model_training.py

# Vérification de la sécurité du code
security:
	bandit -r data_processing.py main.py model_evaluation.py model_persistence.py model_training.py

# Exécution complète du pipeline (préparation, entraînement, évaluation, sauvegarde)
run:
	$(PYTHON) main.py all

# Exécution de toutes les tâches (installation, formatage, linting, sécurité, pipeline)
all: install format lint security run

# Lancement de l'interface MLflow pour visualiser les expériences
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# Servir le modèle enregistré depuis MLflow
serve-model:
	mlflow models serve -m runs:/<run_id>/bagging_model --port 5000
	@echo "Remplacez <run_id> par l'ID de la run MLflow souhaitée (voir mlflow-ui)"

# Nettoyage des fichiers temporaires et artefacts
clean:
	rm -rf $(MODEL_DIR)/*.joblib
	rm -rf artifacts/confusion_matrix.csv
	rm -f mlflow.db
	rm -rf mlruns
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -exec rm -f {} +

# CI/CD - Exécuter toutes les vérifications et le pipeline
ci: install lint format security run

# Aide pour afficher les commandes disponibles
help:
	@echo "Commandes disponibles dans le Makefile:"
	@echo "  install      - Installe les dépendances du projet"
	@echo "  lint         - Vérifie la qualité du code avec flake8 (max-line-length=100)"
	@echo "  format       - Formate le code avec black"
	@echo "  security     - Analyse de sécurité avec bandit"
	@echo "  run          - Exécute le pipeline complet (préparation, entraînement, évaluation, sauvegarde)"
	@echo "  all          - Exécute toutes les tâches (install, format, lint, security, run)"
	@echo "  mlflow-ui    - Lance l'interface MLflow pour visualiser les expériences"
	@echo "  serve-model  - Sert un modèle enregistré via MLflow (remplacez <run_id>)"
	@echo "  clean        - Supprime les fichiers temporaires, artefacts, et caches"
	@echo "  ci           - Exécute toutes les étapes CI/CD (install, lint, format, security, run)"
	@echo "  help         - Affiche cette aide"