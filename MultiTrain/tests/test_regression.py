"""
test cases for Classification class
"""

import pandas as pd
import numpy as np

# import MultiTrain
from MultiTrain.regression.regression_models import MultiRegressor

df = pd.DataFrame(
    {"X": np.arange(0, 10), "X2": np.linspace(2, 3, 10), "Y": np.linspace(0.1, 1, 10)}
)
features = df.drop("Y", axis=1)
target = round(df.Y, 2)
train_reg = MultiRegressor(
    cores=-1,
    # this parameter works exactly the same as setting n_jobs to -1, this uses all the cpu cores to make training faster
    random_state=42,  # setting random state here automatically sets a unified random state across function imports
)
split = train_reg.split(
    X=features, y=target, sizeOfTest=0.1, randomState=42, shuffle_data=True
)


def test_fit():
    """
    Test Regression methods
    """

    """Check return type"""
    fitted_reg = train_reg.fit(
        splitting=True, split_data=split, show_train_score=True, excel=True
    )
    assert pd.DataFrame == type(fitted_reg)
