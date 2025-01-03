from setuptools import setup, find_packages, setup

hyphen_e_dot = '-e .'

def get_requirements(file_path:str) -> list[str]:
    '''
    Get requirements from requirements.txt
    :param file_path:
    :return:
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements


setup(
    name='insurance-prediction-regression',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    author='Gaurav',
    author_email='gaurav.vjadhav01@gmail.com',
    install_requires=get_requirements('Requirements.txt')
)
