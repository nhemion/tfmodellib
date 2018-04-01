from setuptools import setup


with open('README.md') as f:
    readme = f.read()

install_requires = ['numpy>=1.14.0']
try:
    import tensorflow
except ImportError:
    install_requires.append('tensorflow>=1.4.0')
else:
    if tensorflow.test.is_built_with_cuda():
        install_requires.append('tensorflow-gpu>=1.4.0')
    else:
        install_requires.append('tensorflow>=1.4.0')

setup(
    name='tfmodellib',
    version='0.1.0',
    description='A collection of neural network models implemented in Tensorflow.',
    long_description=readme,
    author='Nikolas Hemion',
    author_email='nikolas@hemion.org',
    packages=['tfmodellib'],
    install_requires=install_requires
)

