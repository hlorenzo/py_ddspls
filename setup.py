from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()
    
setup(name='py_ddspls',
      version='1.0.3',
      description='The multi data driven sparse pls package',
      long_description=readme(),
      url='http://github.com/hlorenzo/py_ddspls',
      author='Hadrien Lorenzo',
      author_email='hadrien.lorenzo.2015@gmail.com',
      license='MIT',
      packages=['py_ddspls'],
      install_requires=['numpy','pandas','sklearn'],
      dependency_links=['http://github.com/hlorenzo/py_ddspls'],
      scripts=['bin/example.py'],
      zip_safe=False,
      include_package_data=True,
      package_data={
        'data':
             ['data/penicilliumYES.npy',
             'data/liverToxicity.npy']}
)