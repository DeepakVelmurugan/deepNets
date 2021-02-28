from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: MacOS',
  'Operating System :: Microsoft :: Windows',
  'Operating System :: POSIX :: Linux',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3.7',
  'Programming Language :: Python :: 3.8'
]
 
setup(
  name='deepNets',
  version='0.1.7',
  description='A basic deep learning tool',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Deepak Velmurugan',
  author_email='deepazlions@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='deepLearning', 
  packages=find_packages(),
  setup_requires=['numpy'],
  install_requires=['numpy']
)
