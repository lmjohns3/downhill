import os
import setuptools

setuptools.setup(
    name='downhill',
    version='0.1.0',
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@lmjohns3.com',
    description='SGD-based optimization routines for theano',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.rst')).read(),
    license='MIT',
    url='http://github.com/lmjohns3/theanopt',
    keywords=('adadelta '
              'adagrad '
              'gradient-descent '
              'nesterov '
              'optimization '
              'rmsprop '
              'sgd '
              'theano '
              ),
    install_requires=['theano', 'climate'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        ],
    )
