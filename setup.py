from setuptools import setup, find_packages

with open("README.md", "r") as f:
    desc = f.read()

setup(
    name='MultiTrain',  # name of package
    version="1.0.0",
    author='Shittu Samson',
    author_email='tunexo885@gmail.com',
    maintainer='Shittu Samson',
    maintainer_email='tunex885@gmail.com',
    description="MultiTrain is a user-friendly tool that lets you train several machine learning models at once on your dataset, helping you easily find the best model for your needs.",
    long_description=desc,
    long_description_content_type="text/markdown",
    keywords=[
        "multitrain",
        "multi",
        "train",
        "MultiTrain",
        "multiclass",
        "classifier",
        "automl",
        "AutoML",
        "train multiple models",
    ],
    url="https://github.com/LOVE-DOCTOR/MultiTrain",
    packages=find_packages(
        include=[
            "MultiTrain",
            "MultiTrain.test",
            "MultiTrain.utils",
            "MultiTrain.errors",
            "MultiTrain.regression",
            "MultiTrain.classification",
        ]
    ),
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
        "Development Status :: 5 - Production/Stable",
        "Framework :: Jupyter",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    py_modules=["MultiTrain"],
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        "catboost==1.2.7",
        "imbalanced-learn==0.13.0",
        "lightgbm==4.5.0",
        "setuptools==75.8.0",
        "matplotlib==3.10.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "plotly==6.0.0",
        "pytest==8.3.4",
        "scikit-learn==1.6.1",
        "scipy==1.13.1",
        "seaborn==0.13.2",
        "xgboost==1.6.2",
        "tqdm==4.64.0",
        "ipython==8.32.0"
        
    ]
)
