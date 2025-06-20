
# Exercise 3: Boosted Decision Tree (BDT) for Higgs classification

Similarly to Multi-Layer Perceptrons (MLP), and Boosted Decision Trees (BDTs) can be used for classification purposes. In this second exercise, we will train this alternative network to classify the same $H\to W^+ W^-$ signal events from backgrounds as in exercise 1.


To run the BDT training, simply set in [scripts/train.py](../scripts/train.py) the following parameters:
```python
do_nn = False
do_bdt = True
```

and then run the training with the usual command:
```bash
python scripts/train.py
```

To make plots, simply run
```bash
python scripts/plot_bdt.py
```

## Exercises
### 1) Hyperparameter tuning

- Try different values for `max_depth`, `n_estimators`, `learning_rate`
- Use `GridSearchCV` or `RandomizedSearchCV` from `sklearn` to find the best parameters

### 2) Compare the performance of different models
- Compare the AUC scores of BDT and MLP models.
- Compare the training time of BDT and MLP models.

### 3) Try other BDT implementations
- `SKLearn`'s `GBDT`
- `LightGBM`