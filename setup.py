from setuptools import setup, find_packages

pkgs = ['pandas~=1.3.3',
        'plotly~=5.3.1',
        'numpy~=1.21.2',
        'scikit-image~=0.18.3',
        'pyyaml~=6.0'
        ]

setup(name='SMLM Simulator',
      version='0.1',
      description='SMLM Simulator building on SuReSim',
      python_requires='>=3.6',
      author='Diana Mindroc',
      license='MIT',
      install_requires=pkgs,
      packages=find_packages(exclude=['docs', 'tests*']))
