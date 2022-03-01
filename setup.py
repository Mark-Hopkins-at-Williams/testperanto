from setuptools import setup, find_packages

setup(name='testperanto',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'spacy==3.2.3',
          'pyconll==3.1.0',
          'nltk==3.7',
          'matplotlib==3.5.1',
          'pandas==1.4.1',
          'seaborn==0.11.2',
          'jupyter'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      )

