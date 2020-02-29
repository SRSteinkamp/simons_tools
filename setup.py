from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
# taken from https://uoftcoders.github.io/studyGroup/lessons/python/packages/lesson/
    name='random_ni_tools',
    #url='https://github.com/jladan/package_demo',
    author='Simon Steinkamp',
    author_email='simon.steinkamp@googlemail.com',
    # Needed to actually package something
    packages=['random_ni_tools'],
    # Needed for dependencies
    install_requires=['numpy', 'nilearn', 'nibabel', 'sklearn', 'networkx', 'pandas', 'scipy', ],
    # *strongly* suggested for sharing
    version='0.01',
    # The license can be anything you like
    #license='MIT',
    description='Just some tools for visualizations, doing some work on VOIs (or ROIs), get some information from atlases.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)