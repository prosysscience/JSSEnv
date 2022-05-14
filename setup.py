from setuptools import setup, find_packages
from JSSEnv.version import VERSION

setup(
    name="JSSEnv",
    version=VERSION,
    author="Pierre Tassel",
    author_email="pierre.tassel@aau.at",
    description="An optimized OpenAi gym's environment to simulate the Job-Shop Scheduling problem.",
    url="https://github.com/prosysscience/JSSEnv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "gym",
        "pandas",
        "numpy",
        "plotly",
        "imageio",
        "psutil",
        "requests",
        "kaleido",
        "pytest",
        "codecov",
    ],
    include_package_data=True,
)
