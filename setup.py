import os
from setuptools import setup, find_packages

setup(
    name='sc2ai',
    version="0.1",
    author='Yifei Cheng',
    author_email='ycheng517@gmail.com',
    description='pysc2 agents',
    license='Private',
    url = "https://github.com/ycheng517/sc2ai",
    packages=find_packages(),
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    classifiers=[],
    install_requires=[
        # internal
        'pysc2',
        'numpy',
        'pandas',
    ]
)
