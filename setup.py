from setuptools import find_packages, setup

setup(
    name="systems2atoms",
    version="0.0.0",
    description="Models built for cross-scale optimization for the hydrogen economy",
    url="https://github.com/LLNL/systems2atoms",
    packages=find_packages(),
    include_package_data=True,
    package_data = {'systems2atoms':['systems2atoms/*/inputs/*.csv']}
)