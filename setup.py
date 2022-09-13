import setuptools

with open('README.md', 'r') as f:
    desc = f.read()

setuptools.setup(
    name='MultiTrain',  # name of package
    version="0.0.1",
    author='Shittu Samson',
    description="Test package",
    long_description=desc,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=["MultiTrain"],
    package_dir={'': 'src/MultiTrain'},
    install_requires=[]

)
