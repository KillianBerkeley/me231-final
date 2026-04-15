# ME 231 — Burnout modeling

Multiple models on an employee burnout dataset from Kaggle (`tech_mental_health_burnout.csv`).

- **Single model script** (`nn_model.py`, `kmeans_model.py`, `svm_model.py`, `boosted_tree_model.py`, `regression_model.py`): run one file → outputs for that model only.
- **`run_all_models.py`**: runs all models, builds two ensembles (equal-weight vote and weighted vote). **K-means is excluded from the ensembles** because it performed poorly.
