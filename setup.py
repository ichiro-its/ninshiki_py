from setuptools import setup, find_packages

package_name = 'ninshiki_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ichiro',
    maintainer_email='nathanterbaik@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = ninshiki_py.ninshiki_py_detector:main',
            'viewer = ninshiki_py.ninshiki_py_viewer:main',
        ],
    },
)
