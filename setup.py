import os
import setuptools

setuptools.setup(
    name='downhill',
    version='0.3.1',
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@lmjohns3.com',
    description='Stochastic optimization routines for Theano',
    long_description=open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'README.rst')).read(),
    license='MIT',
    url='http://github.com/lmjohns3/downhill',
    keywords=('adadelta '
              'adam '
              'esgd '
              'gradient-descent '
              'nesterov '
              'optimization '
              'rmsprop '
              'sgd '
              'theano '
              ),
    install_requires=['theano', 'climate'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        ],
    )
