from setuptools import setup, find_packages

setup(
    name='RiboHMM',
    version='1.0.0',
    description='Algorithm for inferring coding sequences from Riboseq data',
    license='Apache',
    author='Dominic Fitzgerald/Anil Raj',
    author_email='dominicfitzgerald11@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    package_data={
            'ribohmm': ['include/*']
        },
    entry_points={
        'console_scripts': ['ribohmm = ribohmm:execute_from_command_line']
    },
    install_requires=['pysam', 'cvxopt', 'numpy', 'scipy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development :: Libraries'
    ]
)