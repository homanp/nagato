from setuptools import find_packages, setup

setup(
    name="nagato-ai",
    version="0.0.1",
    packages=find_packages(),
    description="The open framework for finetuning LLMs on private data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ismail Pelaseyed",
    author_email="ismail@superagent.sh",
    url="https://github.com/homanp/nagato",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
