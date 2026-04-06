from setuptools import setup, find_packages

setup(
    name="mac-tensor",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mac-tensor=mac_tensor.cli:main",
        ],
    },
    python_requires=">=3.9",
)
