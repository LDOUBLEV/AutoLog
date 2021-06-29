from setuptools import setup
# python3.7 setup.py bdist_wheel

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    requirements.append('tqdm')

      
setup(name='AutoLog',
      version='1.0.0',
      install_requires=requirements,
      description='Auto logger',
      author='Liuweiwei',
      author_email='liuvv0203@gmail.com',
      packages=['auto_log'],
      )
