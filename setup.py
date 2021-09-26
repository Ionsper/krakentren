from setuptools import setup

setup(name="krakentren",
      version="0.1.0",
      description="A python package to interact with Kraken.com REST API",
      url="https://github.com/Ionsper/krakentren",
      author="Ionsper",
      author_email="ionsper@outlook.com",
      license="MIT",
      packages=["krakentren"],
      install_requires=["numpy == 1.17.4",
                        "pandas == 1.1.4",
                        "requests == 2.25.0"])
