# GPU-GWAS

## Description and goals
This project aims to develop a fast GWAS data analysis pipeline incorporating GPU acceleration and Machine learning.

[rapids.ai](https://rapids.ai/) will be the core SDK that powers this high performance compute. RAPIDS provides GPU-accelerated
implementations of dataframe and machine learning algorithms which are tens to hundreds of times faster than CPU implementations.

Our high level goals for the hackathon include - 
* Recreating the Hail GWAS sample using RAPIDS APIs
* Wrapping RAPIDS GWAS functions into high level APIs within gpu-gwas framework
* (if time permits) Scaling to larger VCF dataset (e.g. chr22)

Working doc for our team is [here](https://docs.google.com/document/d/1d_czQ9OE_XqtRw2X67fqCzUvQRriuvWXqTSNLmTAzVE/edit#heading=h.xvl7m2ful8yu)

## Workflow
![Workflow-diagram](images/workflow.png)

## Setup Instructions
```
pip install -e requirements.txt
```

## Package components
The `gpugwas` package is broken up into multiple independent modules that deal with different components
of the GWAS pipeline. The modules are all located under the `gpugwas` folder.

1. `gpugwas.io` - This module contains I/O related functions such as loading a VCF/annotation file into a CUDA dataframe.
2. `gpugwas.algorithms` - This module contains ML algorithm implementations in CUDA typically used in GWAS (e.g. linear regression, logistic regression, etc).
3. `gpugwas.viz` - This module contains functions used in visualizing the GWAS model outputs (manhattan plots, q-q plots, etc)
4. `gpugwas.filters` - This module containts functions to filter out variants and samples and perform QC on the input data.

## Example Use Case

An example use case of the pipeline is available in `workflow.py`.

To execute, please run
```
python workflow.py
```

## Results
