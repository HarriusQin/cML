# Ensemble Methods

Combines multiple weak learners to create a strong classifier.

## Models

| Model | File | Key Algorithm |
|-------|------|--------------|
| AdaBoost | [adaboost.md](adaboost.md) | Sequential boosting with reweighting |
| Random Forest | [randomforest.md](randomforest.md) | Parallel bagging with trees |
| XGBoost | [xgboost.md](xgboost.md) | Gradient boosting |
| CatBoost | [catboost.md](catboost.md) | Ordered boosting |

## Common Concepts

### Weak Learner
A model that performs slightly better than random guessing. In this library, decision stumps (depth-1 trees) are commonly used.

### Bagging vs Boosting
- **Bagging** (Random Forest): Train learners in parallel on bootstrap samples
- **Boosting** (AdaBoost): Train learners sequentially, reweighting misclassified samples

### Feature Randomness
Random Forest randomly selects a subset of features at each split, decorrelating trees.
