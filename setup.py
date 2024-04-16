import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flimanalyzer",
    author="Karsten Siller",
    author_email="khsiller@gmail.com",
    description="FLIM Analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uvaKCCI/flimanalyzer",
    packages=setuptools.find_packages(),
    package_data={"": ["resources"]},
    include_package_data=True,
    entry_points={"console_scripts": ["flimanalyzer = flim.analyzerapp:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
