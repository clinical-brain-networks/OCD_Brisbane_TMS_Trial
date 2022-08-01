Clinical Trial OCD Neuroimaging
=========================================
Structural and Functional Neuroimaging analysis of OCD CT data derived from baseline analysis.

<!-- dependencies: pybct, h5py, nibabel, nilearn, pandas, scipy, sklearn, statsmodel.
     insert badges instead -->

Table of contents
-----------------
* [Installation](#installation)
* [Usage](#usage)
  - [Workflow](#workflow)
    + [Functional analysis](#functional-analysis)
    + [Structural analysis](#structural-analysis)
    + [Effective connectivity](#effective-connectivity)
  - [Code structure](#code-structure)
    + [Main modules](#main-modules)
    + [Old (deprecated) modules](#old-deprecated-modules)
    + [Utility](#utility)
* [License](#license)
* [Authors and history](#authors-and-history)
* [Acknowledgments](#acknowledgments)

Installation
------------
> Tested on Ubuntu 20.04
> Linux-5.8.0

It is strongly advised to install the OCDbaseline project into a new virtual environment using python 3.9:

    pyenv install 3.9.7 OCDenv
    pyenv activate OCDenv

Then from the root of the OCDbaseline source repository (where the `setup.py` is located), type

    pip install -e .


Usage
-----

[Workflow](#workflow) provides an overall walkthrough of each of the 3 analysis ([functional](#functional-analysis), [structural](#structural-analysis) and [effective](#effective-analysis)) to reproduce the results of the study.


[Code structure](#code-structure) provides a more specific description of each module.

For more details about each module, refer to each file separately.

## Workflow

This project contains 3 "*streams*" of analysis: functional, structural, and effective connectivity analysis.
> _n.b. technically the effective connectivity analysis is also functional_

### Functional analysis
> The functional analysis assumes that [fMRIPrep](https://github.com/nipreps/fmriprep) has already been run. Before running the following scripts, ensure that the path to the project directory `proj_dir` is correctly set in those scripts and that the output folder `derivatives` has been generated from fMRIPrep with its adequate content.

To perform several preprocessing steps (denoising, filtering, global signal regression, scrubbing, etc.), and the first-level SPM analysis; from the HPC cluster run the following PBS script

    prep_seed-to-voxel.pbs

This calls `preprocessing/post_fmriprep_denoising.py` with a set of default preprocessing parameters. See this file for more details about the preprocessing pipeline and the [fmripop](https://github.com/brain-modelling-group/fmripop) package.

The second-level SPM analysis is performed by running the following command:

    python seed_to_voxel_analysis.py --min_time_after_scrubbing 2 --plot_figs --run_second_level --brain_smoothing_fwhm 8 --fdr_threshold 0.05 --save_outputs

Here, the arguments indicate to discard subjects with less than 2 minutes of data after scrubbing was performed, use the 8mm spatially smoothed data (need to be preprocessed accordingly above) and to use a FDR corrected p-value threshold of 0.05.

The output of the script should look like this (only shown for one pathway):

![seed_to_voxel_analysis](screenshots/seed_to_voxel_analysis.001.jpeg)


## Code structure


This is a quick description of each module, for more details, refers to the docstrings within each file.

> Note: each *_analysis.py script can be run individually with a set of default parameters.

### Main modules

&ensp;&ensp;&ensp;&ensp; **seed_to_voxel_analysis.py**: Main script for the functional analysis. It prepare files for SPM and also performs _first_ and _second_ level analysis in nilearn. The _second level_ routine is performed in 2 phases: 1) extract within-group masks and join them to create _first-level_ mask; 2) re-do _second-level_ using this _first level_ mask. It displays cluster table statistics and color-coded brain maps of difference between groups.


Licence
-------

This work is licensed under a Creative Commons Attribution 4.0 International License.


Authors and history
-------------------

This code was contributed by Sebastien Naze for QIMR Berghofer in 2021-2022.


Acknowledgments
---------------

Australian NHMRC Grant number # GN2001283
