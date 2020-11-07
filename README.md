# GPU-GWAS

## Description and goals
This project aims to develop a fast GWAS data analysis pipeline incorporating GPU acceleration and Machine learning.

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

### Test
To test proper setup, please run the following from the root folder of the repo
```
python test.py
```

## Example Use Case

## Results
