from setuptools import setup

setup(name="krakentren",
      version="0.11.1",
      description="A python package to interact with Kraken.com REST API",
      url="https://github.com/Ionsper/krakentren",
      author="Ionsper",
      author_email="ionsper@outlook.com",
      license="MIT",
      packages=["krakentren"],
      python_requires='>=3.9.2',
      install_requires=["numpy==1.22.3",
                        "pandas==1.4.2",
                        "requests==2.27.1"])
