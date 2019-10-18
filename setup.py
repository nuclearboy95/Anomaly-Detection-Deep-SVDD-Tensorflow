from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_reqs = [
        s for s in [
            line.strip(' \n') for line in f
        ] if not s.startswith('#') and s != ''
    ]

setup(name='deep-svdd',
      version='1.3',
      url='https://github.com/nuclearboy95/Deep-SVDD-Tensorflow',
      license='MIT',
      author='Jihun Yi',
      author_email='t080205@gmail.com',
      description='Tensorflow implementation of Deep SVDD',
      packages=find_packages(exclude=['dist', 'build']),
      include_package_data=True,
      long_description=open('README.md').read(),
      zip_safe=False,
      setup_requires=['nose>=1.0'],
      install_requires=install_reqs,
      test_suite='nose.collector')
