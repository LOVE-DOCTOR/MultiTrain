def feature_label_type_error(X_, y_):
    if isinstance(X_, int or bool) or isinstance(y_, int or bool):
        raise ValueError(
            f"{X_} and {y_} are not valid arguments for 'split'."
            f"Try using the standard variable names e.g split(X, y) instead of split({X_}, {y_})"
        )


def strat_error(strat_):
    if isinstance(strat_, bool) is False:
        raise TypeError(
            "argument of type int or str is not valid. Parameters for strat is either False or True"
        )


def dimensionality_reduction_type_error(dimension_):
    if isinstance(dimension_, bool) is False:
        raise TypeError(
            f'dimensionality_reduction should be set to True or False, received "{dimension_}"'
        )


def test_size_error(size_):
    if size_ < 0 or size_ > 1:
        raise ValueError(
            f"value of sizeOfTest should be between 0 and 1, received {size_}"
        )


def missing_values_error(missing_values):
    if isinstance(missing_values, dict):
        for i, j in missing_values.items():
            if i not in ["cat", "num"]:
                raise KeyError(
                    f'{i} is an invalid key for param missing_values, valid keys are one of ["cat", "num"]'
                )

        if missing_values["cat"] != "most_frequent":
            raise ValueError(
                f"Received value '{missing_values['cat']}', you can only use 'most_frequent' for "
                f"categorical columns"
            )
        elif missing_values["num"] not in [
            "mean",
            "median",
            "most_frequent",
        ]:
            raise ValueError(
                f"Received value '{missing_values['num']}', you can only use one of ['mean', 'median', "
                f"'most_frequent'] for numerical columns"
            )
    else:
        raise TypeError(
            f"missing_values parameter can only be {dict}, {type(missing_values)} with argument {missing_values}"
            f" received"
        )
