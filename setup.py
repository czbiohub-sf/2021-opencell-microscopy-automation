#!/usr/bin/env python

import setuptools

setuptools.setup(
    name='dragonfly-automation',
    description='Automation scripts for OpenCell imaging',
    url='https://github.com/czbiohub/dragonfly-automation',
    packages=setuptools.find_packages(),
    python_requires='>3.7',
    zip_safe=False,
    entry_points={
        'console_scripts': [
        ]
    }
)
