from setuptools import setup, find_packages

setup(
    name="market_analyzer",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "statsmodels",
        "xgboost",
        "tensorflow",
        "prophet",
        "requests",
        "pytest",
    ],
    python_requires=">=3.8",
)