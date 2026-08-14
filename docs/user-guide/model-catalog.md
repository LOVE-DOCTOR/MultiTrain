# Built-in model catalog

Pass any of these names in `custom_models`. Names are case-sensitive because they map directly to estimator factories.

## Classifiers

| Family | Names |
| --- | --- |
| Linear | `LogisticRegression`, `LogisticRegressionCV`, `SGDClassifier`, `PassiveAggressiveClassifier`, `RidgeClassifier`, `RidgeClassifierCV`, `Perceptron` |
| Support vector | `LinearSVC`, `NuSVC`, `SVC` |
| Neighbors and neural network | `KNeighborsClassifier`, `MLPClassifier` |
| Naive Bayes | `GaussianNB`, `BernoulliNB`, `MultinomialNB`, `ComplementNB` |
| Trees and ensembles | `DecisionTreeClassifier`, `ExtraTreeClassifier`, `GradientBoostingClassifier`, `ExtraTreesClassifier`, `BaggingClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `HistGradientBoostingClassifier` |
| External gradient boosting | `CatBoostClassifier`, `LGBMClassifier`, `XGBClassifier` |

## Regressors

| Family | Names |
| --- | --- |
| Linear and regularized | `LinearRegression`, `Ridge`, `RidgeCV`, `Lasso`, `LassoCV`, `ElasticNet`, `ElasticNetCV`, `Lars`, `LarsCV`, `OrthogonalMatchingPursuit`, `OrthogonalMatchingPursuitCV`, `BayesianRidge`, `ARDRegression` |
| Robust and generalized linear | `HuberRegressor`, `TheilSenRegressor`, `RANSACRegressor`, `PoissonRegressor`, `GammaRegressor`, `TweedieRegressor` |
| Online linear | `SGDRegressor`, `PassiveAggressiveRegressor` |
| Neighbors, neural network, and support vector | `KNeighborsRegressor`, `MLPRegressor`, `SVR`, `LinearSVR`, `NuSVR` |
| Trees and ensembles | `DecisionTreeRegressor`, `ExtraTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `AdaBoostRegressor`, `BaggingRegressor`, `HistGradientBoostingRegressor` |
| External gradient boosting | `CatBoostRegressor`, `LGBMRegressor`, `XGBRegressor` |

## Select a smaller comparison

```python
train = MultiClassifier(
    custom_models=[
        "LogisticRegression",
        "RandomForestClassifier",
        "LGBMClassifier",
    ]
)
```

An empty list is rejected. Duplicate names are rejected because they would produce ambiguous result and artifact keys.

The exact underlying defaults are versioned with MultiTrain. Use {doc}`custom-models` when an experiment needs explicit estimator parameters.
