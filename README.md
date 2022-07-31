# [Toward Improving Effectiveness of Crowdsourced, On-Demand Assistance From Authors in Online Learning Platforms](https://doi.org/10.17605/osf.io/zcbjx)

[![GitHub DOI](https://zenodo.org/badge/519476273.svg)](https://zenodo.org/badge/latestdoi/519476273)
[![OSF DOI](https://img.shields.io/badge/OSF-10.17605%2Fosf.io%2Fzcbjx-blue)](https://doi.org/10.17605/osf.io/zcbjx)


[![Docker](https://img.shields.io/docker/automated/ahaim5357/10.17605-osf.io-zcbjx)](https://hub.docker.com/repository/docker/ahaim5357/10.17605-osf.io-zcbjx)
[![Docker DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6946374.svg)](https://doi.org/10.5281/zenodo.6946374)

[*Toward Improving Effectiveness of Crowdsourced, On-Demand Assistance From Authors in Online Learning Platforms*](https://doi.org/10.17605/osf.io/zcbjx) is a project that ran within the [ASSISTments Platform](assistments.org) from February 16th, 2022 to June 1st, 2022. This attempted to determine between any two authors within the platform which was better at generating on-demand assistance (e.g. hints and explanations). It contains the preregistration and updates of this project through this time period, the datasets compiled, constructed, and used, and the analysis ran to obtain the specified results.

## External Datasets

This project uses datasets compiled outside of this study:

* [Public Elementary/Secondary School Local Codes](https://nces.ed.gov/ccd/CCDLocaleCode.asp)
* [Inferred Gender of Students ASSISTments](https://osf.io/pm4ux/?view_only=84a5ee4af2f24bbb82a65bc99235684b)

## Analysis

### Method 1: Docker

The [Docker Container](https://hub.docker.com/r/ahaim5357/10.17605-osf.io-zcbjx) can be run using:

    docker run --name <container_name> ahaim5357/10.17605-osf.io-zcbjx:xprize
    docker cp <container_name>:app/results ./results

You can also clone this repository and run using the following [Docker](https://www.docker.com/) commands:

    docker build -t <image_name> .
    docker run --name <container_name> <image_name>
    docker cp <container_name>:app/results ./results

Where `image_name` and `container_name` are specified as whatever identifiers the user desires.

#### Rerunning the Container

If you need to rerun the container for any reason, you can do so by finding the container id using:

    docker ps -a

Afterwards, you can create a new container and run it:

    docker commit $<container_id> <new_image_name>
    docker run -it --entrypoint=sh <new_image_name>

### Method 2: Python

#### Libraries

The analysis code is written in Python 3.8.10. The following libraries are required to run the analysis. Additionally, the following versions were used to provide the results:

* pandas: 0.25.3
* numpy: 1.17.4
* statsmodels: 0.13.2
* matplotlib: 3.1.2
* seaborn: 0.11.2

#### Datasets

The [Inferred Gender of Students ASSISTments Dataset](https://osf.io/b3c56?view_only=84a5ee4af2f24bbb82a65bc99235684b) needs to be downloaded to run the analysis.

#### Setup

Put the necessary Python files and datasets in the same directory.

    (folder name)
    |- main.py
    |- aosc_constants.py
    |- preprocessor.py
    |- construct.py
    |- interactions.py
    |- model.py
    |- plot.py
    |- student_support_logs.csv
    |- student_support_features.csv
    |- student_prior_stats.csv
    |- infered_student_gender_ASSISTments_2021_Aug_15_2022_Jul_15.csv (Obtained from https://osf.io/b3c56?view_only=84a5ee4af2f24bbb82a65bc99235684b)
    |- student_locale_info.csv
    |- star_authors.csv

Navigate to (folder name) within your terminal and run `main.py`.

For Unix Systems (Linux, MacOS):

    python3 ./main.py

For Windows Systems:

    py ./main.py
