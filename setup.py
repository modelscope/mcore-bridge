# Copyright (c) ModelScope Contributors. All rights reserved.
from setuptools import find_packages, setup

import re


def get_readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'src/mcore_bridge/version.py'


def get_version():
    with open(version_file, 'r', encoding='utf-8') as f:
        return re.search(r'^__version__\s*=\s*["\'](.+?)["\']', f.read(), re.M).group(1)


def parse_requirements(path='requirements.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    requirements = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        requirements.append(line)
    return requirements


if __name__ == '__main__':
    install_requires = parse_requirements('requirements.txt')

    setup(
        name='mcore_bridge',
        version=get_version(),
        description='MCore-Bridge: Making Megatron training as simple as Transformers',
        long_description=get_readme(),
        long_description_content_type='text/markdown',
        author='ModelScope teams',
        author_email='contact@modelscope.cn',
        keywords=['transformers', 'LLM', 'lora', 'megatron', 'peft'],
        url='https://github.com/modelscope/mcore-bridge',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        # The Megatron patch for GLM-5.3 ships inside the package so an installed mcore-bridge can
        # apply it (see mcore_bridge/tools/apply_megatron_patch.py).
        package_data={'mcore_bridge': ['patches/*.patch', 'patches/README.md']},
        python_requires='>=3.8.0',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
        ],
        license='Apache License 2.0',
        install_requires=install_requires,
        zip_safe=False)
