from setuptools import setup, find_packages

setup(
    name="number-sequence-predictor",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "numpy",
        "matplotlib",
        "tensorflow",
    ],
)
