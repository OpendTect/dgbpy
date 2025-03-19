""" dgbpy contains generic deep learning python tools for seismic interpretation. """

from setuptools import setup, find_packages
import re

with open('dgbpy/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='dgbpy',
    packages=find_packages(exclude=[]),
    version=version,
    url='https://github.com/OpendTect/dgbpy',
    license='Apache 2.0',
    author='dGB Earth Sciences',
    author_email='info@dgbes.com',
    description='Deep Learning tools for seismic interpretation using OpendTect',
    long_description='',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'bokeh>=2.1.1',
        'h5py>=2.10.0',
        'joblib>=1.0.1',
        'keras>=2.3.1,<2.15.0',
        'numpy>=1.19.2,<2.0',
        'odpy>=1.1.0',
        'scikit-learn>=0.24.2',
        'psutil>=5.7.0',
        'tensorflow>=2.1.4,<2.15.0',
        'torch>=1.9.0',
        'fastprogress>=1.0.0',
        'onnxruntime-gpu>=1.0.0',
    ],
    extras_require={
        'onnxmltools': ['onnxmltools>=1.9.0'],
        'pydot': ['pydot>=1.4.1'],
        'scikit-learn-intelex': ['scikit-learn-intelex>=1.0.0'],
        'skl2onnx': ['skl2onnx>=1.0.0'],
        'xgboost': ['xgboost>=1.1.1'],
        'boto3': ['boto3>=1.34.60'],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
    ],
)
