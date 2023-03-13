from setuptools import setup
    
setup(name='py_ddspls',
      version='1.1.1',
      description='The multi data driven sparse pls package',
      long_description=open('README.txt').read(),
      url='http://github.com/hlorenzo/py_ddspls',
	  download_url='https://github.com/hlorenzo/py_ddspls/archive/v_111.tar.gz',
      author='Hadrien Lorenzo',
      author_email='hadrien.lorenzo.2015@gmail.com',
      license='MIT',
      packages=['py_ddspls'],
      install_requires=['numpy','pandas','scikit-learn'],
      dependency_links=['http://github.com/hlorenzo/py_ddspls'],
      scripts=['bin/example.py'],
      zip_safe=False,
      include_package_data=True,
      package_data={
        'data':
             ['data/penicilliumYES.npy',
             'data/liverToxicity.npy']}
)
