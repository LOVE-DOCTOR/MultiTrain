from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    desc = f.read()

setup(
    name='MultiTrain',  # name of package
    version="0.0.1",
    author='Shittu Samson',
    author_email='tunexo885@gmail.com',
    description="Test package",
    long_description=desc,
    long_description_content_type='text/markdown',
    url="https://github.com/LOVE-DOCTOR/train-with-models",
    packages=find_packages(where="MultiTrain", include=['MultiTrain', 'MultiTrain.tests', 'MultiTrain.methods',
                                                        'MultiTrain.regression', 'MultiTrain.classification']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=["matplotlib",
                      "pandas",
                      "scikit-learn",
                      "numpy",
                      "plotly",
                      "xgboost",
                      "catboost",
                      "imbalanced-learn",
                      "seaborn",
                      "scikit-optimize",
                      "lightgbm"]

)
