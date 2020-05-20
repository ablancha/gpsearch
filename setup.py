from setuptools import setup

with open("requirements.txt", "r") as req:
    requires = req.read().split("\n")


setup(name="gpsearch",
      version="0.1",
      description="Active learning with output-weighted importance sampling",
     #url="http://github.com/storborg/funniest",
      author="Antoine Blanchard",
      author_email="ablancha@mit.edu",
      install_requires=requires,
      packages=setuptools.find_packages(),
      include_package_data=True,
      license="MIT"
    )
