from setuptools import setup, find_packages

setup(name='testperanto',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'spacy',
          'pyconll',
          'nltk',
          'matplotlib',
          'pandas',
          'seaborn',
          'jupyter'
      ],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      )

