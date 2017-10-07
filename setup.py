from setuptools import setup

setup(
      name = "mass",
      version = 0.1,
      packages = ["mass"],
      description = "MCMC Massive Sample - Bayesian inference for very large samples.",
      author = "Manuel Silva",
      author_email = "madusilva@gmail.com",
      license="GPLv2",
      classifiers=[
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering :: Astronomy"

      ],
      install_requires = ["numpy"],
      package_data = {
          '' : []
        },
      zip_safe=False
)