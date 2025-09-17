from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='eeg-sleep-analysis',
    version='0.1.0',
    author='S.H. Noteboom',
    author_email='s.h.noteboom@amsterdamumc.nl',
    description='EEG sleep stage analysis pipeline',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=required,
    include_package_data=True,
)


