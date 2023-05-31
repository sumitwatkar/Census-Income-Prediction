from setuptools import find_packages, setup
from typing import List

HYPHON_E_DOT = "-e ."

def get_requriments(filepath: str) -> List[str]:
    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [ req.replace("\n", "") for req in requirements]

        if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)
            return requirements


setup(name='Census-Income-Prediction',
      version='0.0.1',
      description='Machine Learning Pipeline Project for Census Income Prediction',
      author='Sumit Watkar',
      author_email='watkar.sumit@gmail.com',
      packages=find_packages(),
      install_requires = get_requriments("requirements.txt")
     )