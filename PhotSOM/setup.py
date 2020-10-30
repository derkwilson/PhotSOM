from setuptools import setup

setup(name='PhotSOM',
      version='0.1',
      description='Compute photometric redshifts using self-organizing maps',
      url='',
      author='Derek Wilson',
      author_email='dnwilson@uci.edu',
      license='MIT',
      packages=['PhotSOM'],
      install_requires=['numpy','pandas','matplotlib'],
      zip_safe=False)
