from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='bismet',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Bathymetry interpolation smoothing and meshing tools using implicit functions, targetting fdaPDE in R.",
    license="MIT",
    author="Eli Ateljevich",
    author_email='Eli.Ateljevich@water.ca.gov',
    url='https://github.com/water-e/bismet',
    packages=['bismet'],
    
    install_requires=requirements,
    keywords='bismet',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
