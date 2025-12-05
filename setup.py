from setuptools import setup, find_packages

setup(
    name="rfm",
    version="0.1.0",
    description="Routing Foundation Model (RFM): neural optimization framework for routing MILPs",
    author="Ritwika Kancharla",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "networkx",
        "pyyaml",
        "tqdm",
        "matplotlib",
    ],
)
