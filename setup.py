
import setuptools

#requirements = ["numpy", "scipy", "sklearn", "nibabel", "nilearn", "pandas", "statsmodels", "bctpy", "matplotlib", "h5py"]

reqs = ["bctpy", \
        "h5py", \
        "matplotlib", \
        "nibabel", \
        "nilearn", \
        "nltools", \
        "numpy", \
        "pandas", \
        "pingouin", \
        "scikit-learn", \
        "scipy", \
        "seaborn", \
        "statsmodels"]

setuptools.setup(
    name="OCD_clinical_trial",
    version="0.0.1",
    author="Sebastien Naze",
    author_email="sebastien.naze@gmail.com",
    description="OCD Clinical Trial Analysis",
    url="https://github.com/sebnaze/OCD_clinical_trial",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=reqs,
    include_package_data=True,
)
