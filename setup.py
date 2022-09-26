from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    desc = f.read()

setup(
    name='MultiTrain',  # name of package
    version="0.12.3",
    author='Shittu Samson',
    author_email='tunexo885@gmail.com',
    maintainer='Shittu Samson',
    maintainer_email='tunex885@gmail.com',
    description="MultiTrain allows you to train multiple machine learning algorthims on a dataset all at once to determine the best for that particular use case",
    long_description=desc,
    long_description_content_type='text/markdown',
    keywords=['multitrain', 'multi', 'train', 'MultiTrain', 'multiclass', 'classifier', 'automl', 'AutoML', 'train multiple models'],
    url="https://github.com/LOVE-DOCTOR/train-with-models",
    packages=find_packages(include=['MultiTrain', 'MultiTrain.tests', 'MultiTrain.methods',
                                    'MultiTrain.regression', 'MultiTrain.classification']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Framework :: Jupyter",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    py_modules=['MultiTrain'],
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=["matplotlib==3.5.3",
                      "pandas==1.4.4",
                      "scikit-learn==1.1.2",
                      "numpy==1.23.3",
                      "plotly==5.10.0",
                      "ipython==8.4.0",
                      "xgboost==1.6.2",
                      "catboost==1.0.6",
                      "imbalanced-learn==0.9.1",
                      "seaborn==0.12.0",
                      "scikit-optimize==0.9.0",
                      "lightgbm==3.3.2",
                      "kaleido==0.2.1"]
)