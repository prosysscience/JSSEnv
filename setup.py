from setuptools import setup, find_packages

setup(
    name="JSSEnv",
    version="1.1.0",
    author="Pierre Tassel",
    author_email="pierre.tassel@aau.at",
    description="An optimized OpenAi gym's environment to simulate the Job-Shop Scheduling problem.",
    url="https://github.com/prosysscience/JSSEnv",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.20.0,<1.24.0",
        "pandas>=1.3.0,<2.1.0",
        "plotly",
        "imageio",
        "psutil",
        "requests",
        "kaleido",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "codecov>=2.1.0",
            "build>=1.0.0",
            "wheel>=0.40.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
