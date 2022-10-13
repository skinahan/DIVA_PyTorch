# TorchDIVA
Implementation of the [DIVA model of speech acquisition and production](https://sites.bu.edu/guentherlab/research-projects/the-diva-model-of-speech-motor-control/) using PyTorch.

# Overview

DIVA is a popular computational model based on neural networks for better understanding the processes of human speech acquisition and production. The publicly available implementation of DIVA was originally done in Matlab. Recently, neural networks have become increasingly more popular, with new open-source frameworks available to researchers for development. As Matlab is rarely used in this context, we propose to extend the usability of DIVA by implementing it in the Python machine learning framework, PyTorch. 

This repository contains the source code for TorchDIVA, a fully functional recreation of the original DIVA model.

# Model Architecture

This simplified schematic illustrates the combination of the feedforward and feedback control loops in the DIVA model.

![DIVA Structural Diagram](https://user-images.githubusercontent.com/14182023/195723655-dcccfca5-20f2-4533-b25d-2e19a3ca942a.png)

# Model Validation

[An example can be viewed here](src/notebooks/motor-mse-demo.ipynb) of the motor command signal difference between TorchDIVA and the original Matlab DIVA model.

# Setup
The following section lists the requirements for working with TorchDIVA.

# Requirements
- [Conda package and environment manager]()
- Python 3.7
- Operating System: Windows 10
- (Recommended) Python IDE

Note: TorchDIVA was developed under Windows 10. Compatibility with other operating systems is likely, but may require some manual configuration.

# Setup

1. Clone the repository

```
git clone https://github.com/skinahan/DIVA_PyTorch.git
```

2. Create the Conda virtual environment using the provided environment.yml file

```
conda env create --file=environment.yml
```

3. Activate the new environment

```
conda activate DIVAProject-clean
```

# Usage

Run main.py using the python command-line or IDE of your choice.

```
cd src/
python ./main.py
```
You will be greeted by a menu like the following:

```
= PyTorch DIVA Implementation Menu =
Active Production: happy
Select from the following options:
l    :   list targets
n    :   new target
p    :   change production
r    :   reset target
s    :   save target
sim    :   start simulation
last    :    playback last output
log     :    save produced articulator movements
q    :   quit
```

Note: If you see any error message on startup, you may be missing a dependency. Verify that you have successfully created and activated the conda virtual environment before running TorchDIVA.

Use the menu options available to interact with the TorchDIVA model:
Select speech targets, run speech simulations, etc.
