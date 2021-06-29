from setuptools import setup
# python3.7 setup.py bdist_wheel

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

      
setup(name='auto_log',
      version='1.0.0',
      install_requires=requirements,
      license='Apache License 2.0',
      keywords='auto log',
      description="The AutoLog Contains automatic timing, statistics on CPU memory, GPU memory and other information, since generating logs and other functions.",
      url='https://github.com/LDOUBLEV/AutoLog',
      author='DoubleV',
      author_email='liuvv0203@gmail.com',
      packages=['auto_log'],
      )
