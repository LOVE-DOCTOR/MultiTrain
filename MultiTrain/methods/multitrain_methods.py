import numpy as np
import pandas as pd
import os
import shutil
import logging

from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pandas.core.series import Series
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_to_excel(name: any, file: any) -> None:
    """
    If the name is True, then write the file to an excel file called "Training_results.xlsx"

    :param name: This is the name of the file you want to save
    :param file: the name of the file you want to read in
    """
    if name is True:
        file.to_excel("Training_results.xlsx")
    else:
        pass


def directory(FOLDER_NAME):
    """
    If the folder doesn't exist, create it

    :param FOLDER_NAME: The name of the folder you want to create
    """
    if not os.path.exists(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)
        return FOLDER_NAME
    # The above code is checking if the folder exists. If it does, it asks the user if they want to overwrite the
    # current directory or specify a new folder name. If the user chooses to overwrite the current directory,
    # the code deletes the current directory and creates a new one.
    elif os.path.exists(FOLDER_NAME):
        print("Directory exists already")
        print(
            "Do you want to overwrite current directory(y) or specify a new folder name(n)."
        )
        confirmation_values = ["y", "n"]
        while True:
            confirmation = input("y/n: ").lower()
            if confirmation in confirmation_values:
                if confirmation == "y":
                    shutil.rmtree(FOLDER_NAME)
                    os.mkdir(FOLDER_NAME)

                    return FOLDER_NAME

                # The above code is checking if the user has entered a valid folder name.
                elif confirmation == "n":
                    INVALID_CHAR = [
                        "#",
                        "%",
                        "&",
                        "{",
                        "}",
                        "<",
                        "<",
                        "/",
                        "$",
                        "!",
                        "'",
                        '"',
                        ":",
                        "@",
                        "+",
                        "`",
                        "|",
                        "=",
                        "*",
                        "?",
                    ]
                    while True:
                        FOLDER_NAME_ = input("Folder name: ")
                        folder_name = list(FOLDER_NAME_.split(","))
                        compare_names = all(
                            item in INVALID_CHAR for item in folder_name
                        )
                        if compare_names:
                            raise ValueError(
                                "Invalid character specified in folder name"
                            )
                        else:
                            os.mkdir(FOLDER_NAME_)
                            logger.info(
                                f"Directory {FOLDER_NAME_} successfully created"
                            )
                            return FOLDER_NAME_

            else:
                logger.info("Select from y/n")


def img_plotly(
    figure: any, name: any, label: str, FILENAME: str, FILE_PATH: any
) -> None:
    SOURCE_FILE_PATH = FILE_PATH + f"/{name}"
    DESTINATION_FILE_PATH = FILE_PATH + f"/{FILENAME}" + f"/{name}"
    figure.write_image(name, width=1280, height=720)
    shutil.move(src=SOURCE_FILE_PATH, dst=DESTINATION_FILE_PATH)


def img(FILENAME: any, FILE_PATH: any, type_="file") -> None:
    """
    It takes a filename and a type, and saves all the figures in the current figure list to a pdf file or a picture
    file

    :param FILE_PATH:
    :param FILENAME: The name of the file you want to save
    :type FILENAME: any
    :param type_: 'file' or 'picture', defaults to file (optional)
    """
    if type_ == "file":
        FILE = PdfPages(FILENAME)
        figureCount = plt.get_fignums()
        fig = [plt.figure(n) for n in figureCount]

        for i in fig:
            i.savefig(
                FILE,
                format="pdf",
                dpi=550,
                papertype="a4",
                bbox_inches="tight",
            )

        FILE.close()

    elif type_ == "picture":
        FILE = directory(FILENAME)

        figureCount = plt.get_fignums()
        fig = [plt.figure(n) for n in figureCount]
        fig_dict = {}
        fig_num = list(range(6))
        for i in range(len(fig_num)):
            fig_dict.update({fig_num[i]: fig[i]})

        for key, value in fig_dict.items():
            add_path = key
            FINAL_PATH = FILE_PATH + f"/{FILE}" + f"/{add_path}"
            value.savefig(FINAL_PATH, dpi=1080, bbox_inches="tight")


def _check_target(target):
    target_class = "binary" if target.value_counts().count() == 2 else "multiclass"
    return target_class


def _get_cat_num(dictionary):
    categorical_values = ""
    numerical_values = ""
    for i, j in dictionary.items():
        if i == "cat":
            categorical_values = j
        elif i == "num":
            numerical_values = j
        else:
            raise Exception

    return categorical_values, numerical_values


def _fill(value1, value2):
    cat = SimpleImputer(strategy=value1, missing_values=np.nan)
    num = SimpleImputer(strategy=value2, missing_values=np.nan)
    return cat, num


def _fill_columns(cat_init, num_init, features):
    for i in features.columns:
        if features[i].dtypes == "object":
            imputer = cat_init.fit(features[[i]])
            features[[i]] = imputer.transform(features[[i]])
        else:
            imputer = num_init.fit(features[[i]])
            features[[i]] = imputer.transform(features[[i]])
    return features


def _dummy(features, encoder):
    label = LabelEncoder()
    for i in features.columns:
        if features[i].dtypes == "object":
            if encoder == "labelencoder":
                features[i] = label.fit_transform(features[i])
                return features

            elif encoder == "onehotencoder":
                features = pd.get_dummies(features)
                return features

            elif isinstance(encoder, dict):
                for keys, values in encoder.items():
                    if keys == "labelencoder":
                        if isinstance(values, list):
                            for i in values:
                                features[i] = label.fit_transform(features[i])
                        else:
                            raise TypeError(
                                f"received a {type(values)} in dictionary values, pass a list instead"
                            )

                    elif keys == "onehotencoder":
                        if isinstance(values, list):
                            features = pd.get_dummies(features, columns=[values])
                        else:
                            raise TypeError(
                                f"received a {type(values)} in dictionary values, pass a list instead"
                            )

                    else:
                        raise ValueError(
                            f"received {keys}, dictionary keys must be one of 'labelencoder' or 'onehotencoder' "
                        )

                return features

            else:
                raise ValueError(
                    f'the encoder parameter only supports "labelencoder", "onehotencoder", or a dictionary, '
                    f"received {encoder} "
                )
        return features


def show_best(data: DataFrame, best: str, high: list, low: list):

    if best in high:
        df = data.sort_values(by=best, ascending=False)
        retrieve_df = df.reset_index()
        logger.info(
            f'The best model based on the {best} metric is {retrieve_df["index"][0]}'
        )
        display(df)
        return df

    elif best in low:
        df = data.sort_values(by=best, ascending=True)
        retrieve_df = df.reset_index()
        logger.info(
            f'The best model based on the {best} metric is {retrieve_df["index"][0]}'
        )
        display(df)
        return df
