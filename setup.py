from setuptools import setup, find_packages

setup(
    name='featureExtractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'Pillow',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A custom DRNet-based feature extractor for medical imaging and classification tasks.',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/featureExtractor',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
