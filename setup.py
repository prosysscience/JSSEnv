from setuptools import setup

setup(name='JSSEnv',
      version='1.0.0',
      install_requires=['gym', 'pandas', 'numpy', 'plotly', 'imageio', 'psutil', 'requests', 'kaleido', 'pytest', 'codecov'],
      include_package_data=True
)