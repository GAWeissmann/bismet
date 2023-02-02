from setuptools import setup
import versioneer

requirements = ['schimpy','scipy>=1.4','numpy',
                'matplotlib','scikit-image','scikit-fmm','gdal',
                'nodepy','pydistmesh']

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
        'Programming Language :: Python :: 3.8'
    ],
    entry_points={
        'console_scripts': [
            'contour_smooth=bismet.contour_smooth:main',
            'prep_mesh_sdist=bismet.prep_mesh_sdist:main',
            'remove_skewed_cells=bismet.remove_skewed_cells:main',
            'sdist_to_direction=bismet.sdist_to_direction:main'
        ],
    }
)
