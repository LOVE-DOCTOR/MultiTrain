def raise_text_error(text_, vectorizer_, ngrams_):
    if text_:
        if isinstance(text_, bool) is False:
            raise TypeError("parameter text is of type bool only. set to true or false")

        if text_ is False:
            if vectorizer_ is not None:
                raise Exception(
                    "parameter vectorizer can only be accepted when parameter text is True"
                )

            if ngrams_ is not None:
                raise Exception(
                    "parameter ngrams can only be accepted when parameter text is True"
                )


def raise_imbalanced_error(imbalanced_, sampling_):
    if imbalanced_ is False:
        if sampling_:
            raise Exception(
                'this parameter can only be used if "imbalanced" is set to True'
            )


def raise_kfold1_error(kf_, splitting_, split_data_):
    if kf_:
        if isinstance(kf_, bool) is False:
            raise TypeError(
                f"You can only declare object type 'bool' in kf. Try kf = False or kf = True "
                f"instead of kf = {kf_}"
            )

        if splitting_:
            raise ValueError(
                "KFold cross validation cannot be true if splitting is true and splitting cannot be "
                "true if KFold is true"
            )

        if split_data_:
            raise ValueError(
                "split_data cannot be used with kf, set splitting to True to use param "
                "split_data"
            )


def raise_split_data_error(split_data_, splitting_):
    if split_data_:
        if isinstance(split_data_, tuple) is False:
            raise TypeError(
                "You can only pass in the return values of the split method to split_data"
            )

        if splitting_ is None:
            raise ValueError(
                "You must set splitting to True or False if the split_data parameter is used"
            )


def raise_splitting_error(splitting_, split_data_):
    if splitting_:
        if isinstance(splitting_, bool):
            if split_data_ is None:
                raise ValueError(
                    "You must pass in the return values of the split method to split_data if splitting "
                    "is True"
                )

            if isinstance(split_data_, tuple) is False:
                raise TypeError(
                    "You can only pass in the return values of the split method to split_data"
                )

        elif isinstance(splitting_, bool) is False:
            raise ValueError(
                f"splitting can only be set to True or False, received {splitting_}"
            )


def raise_fold_type_error(fold_):
    if isinstance(fold_, int) is False:
        raise TypeError(
            "param fold is of type int, pass a integer to fold e.g fold = 5, where 5 is number of "
            "splits you want to use for the cross validation procedure"
        )


def raise_kfold2_error(kf_, X_, y_):
    if kf_ is True and (X_ is None or y_ is None or (X_ is None and y_ is None)):
        raise ValueError("Set the values of features X and target y when kf is True")
