from setuptools import setup
from setuptools import find_packages


setup(
    name='Icarus',

    version='2.3.2',

    # description
    description=('Icarus is a stellar binary light curve synthesis tool '
                 'initially developed by Rene Breton'),
    long_description=open('README.md', encoding='utf-8').read(),

    # The project's main homepage.
    url='https://github.com/bretonr/Icarus',

    # The project's download url.
    download_url='https://github.com/bretonr/Icarus/tarball/v2.3.2',

    # Author details
    author='Dr Rene Breton',
    author_email='superluminique@gmail.com',

    # license
    license='BSD',

    classifiers=[

        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',

        # relevant topics
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Astronomy',

        # license
        'License :: OSI Approved :: BSD License',

        # python versions this library supports
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    keywords=['astrophysics','cosmology', 'photometry', 'binary', 'modeling',
              'space', 'models', 'spectroscopy', 'astronomy', 'science',
              'research', 'stars', 'physics'],

    # includes everything except the examples
    packages=find_packages(exclude=['Examples']),

    # as stated on https://github.com/bretonr/Icarus
    install_requires=['numpy', 'scipy', 'astropy'],

    python_requires='>=3.6',


    # including the geodesic data files.
    include_package_data=True,
    package_data={
        '': ['*.txt'],
    }
)

# recommended libraries
try:
    import matplotlib
except:
    print('matlibplot is not installed. Although not a requirement, please '
          'install it in order to get better graphs.')

try:
    import PyGTS
except:
    print('PyGTS is not installed. Although not a requirement, please '
          'install it in order to generate surface geodesic primitives '
          'instead of reading the pre-generated one, and calculate '
          'occultations and transits in eclipsing binaries.')
