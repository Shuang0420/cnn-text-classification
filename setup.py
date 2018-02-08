# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

'''
Build the cnn-text-classification Py2/Py3 Chinese version package
'''

from setuptools import setup, find_packages

VERSION = "0.1.0"


setup_args = dict( name="classifier",
                   version=VERSION,
                   author="Shuang0420",
                   author_email="sxu1@alumni.cmu.edu",

                   description="CNN Text Classification",
                   long_description="""a slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.
""",
                   url="https://github.com/Shuang0420/cnn-text-classification.git",
                   platforms=["any"],
                   classifiers=["Development Status :: 1 - Beta",
                                "Environment :: Console",
                                "Intended Audience :: Developers",
                                "Programming Language :: Python",
                                "Programming Language :: Python :: 3",
                                "Programming Language :: Python :: 3.4",
                                "Programming Language :: Python :: 3.5",
                                "Programming Language :: Python :: 3.6",
                                "License :: OSI Approved :: BSD License",
                                "Operating System :: OS Independent",
                                "Topic :: Communications :: Chat",
                                "Topic :: Scientific/Engineering :: Artificial Intelligence"
                                ],

                   install_requires = [ 'setuptools',
                                        ],

                    packages = find_packages(),


                   include_package_data = False,       # otherwise package_data is not used
                   )

if __name__ == '__main__':
    setup( **setup_args )
