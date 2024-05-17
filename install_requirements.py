# Description: Install required packages listed in the requirements.txt file, and upgrade pip.

import os


def install_requirements(requirements_file):
    """
    Install required packages listed in the requirements.txt file, and upgrade pip.
    """
    os.system('python -m pip install --upgrade pip')
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()
        for req in requirements:
            req = req.strip()           # Remove leading/trailing whitespace
            if req.startswith('#'):
                continue                # Skip comments
            os.system(f'pip install --no-cache {req}')
