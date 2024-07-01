from setuptools import setup, find_packages


setup(
    name='histofex',
    version='0.1.0',
    description='HistoFeatEx',
    url='https://github.com/ezgiogulmus/HistoFeatEx',
    author='FEO',
    author_email='',
    license='GPLv3',
    packages=find_packages(exclude=['assets', 'datasets_csv', "splits"]),
    install_requires=[
        "torch>=2.3.0",
        "torchvision",
        "numpy==1.23.4", 
        "pandas==1.4.3",
        "h5py",
        "transformers==4.31.0",
        "timm==0.9.8",
        "conch @ git+https://github.com/mahmoodlab/CONCH",
        "uni @ git+https://github.com/mahmoodlab/UNI"
    ],

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
    ]
)