from setuptools import setup, find_packages

setup(name='testperanto',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'torch',
          'torchvision==0.10.0',
          'transformers==4.10.0',
          'sklearn',
          'datasets==1.11.0',
          'matplotlib==3.4.2',
          'seaborn==0.11.2',
          'jupyter'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      )

