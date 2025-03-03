from setuptools import setup, find_packages

pkgs = [
    'pandas~=1.5.3',
    'plotly~=5.3.1',
    'numpy~=1.24.2',
    'scikit-image~=0.20.0',
    'pyyaml~=6.0',
    'torch==2.4.0+cu124',
    'torchvision==0.19.0+cu124',
    'torchaudio==2.4.0+cu124',
    'matplotlib~=3.7',
    'einops~=0.6',
    'imageio~=2.30',
    'timm~=0.9',
    'ninja~=1.7',
    'easydict~=1.1'
]

setup(
    name='SMLM Simulator',
    version='0.1',
    description='SMLM Simulator building on SuReSim',
    python_requires='>=3.9',
    author='Diana Mindroc',
    license='MIT',
    install_requires=pkgs,
    packages=find_packages(exclude=['docs', 'tests*'])
)

