# TorchDIVA
Implementation of the [DIVA model of speech acquisition and production](https://sites.bu.edu/guentherlab/research-projects/the-diva-model-of-speech-motor-control/) using PyTorch.

Note: If you use this repo in your own project, please cite the original paper [which can be found here](https://arxiv.org/abs/2210.09334).

# Overview

The DIVA model is a computational model of speech motor control that combines a simulation of the brain regions responsible for speech production with a model of the human vocal tract. The model is currently implemented in Matlab Simulink; however, this is less than ideal as most of the development in speech technology research is done in Python. This means there is a wealth of machine learning tools which are freely available in the Python ecosystem that cannot be integrated with DIVA. 

We present the source code for TorchDIVA, a full rebuild of DIVA in Python using PyTorch tensors. DIVA source code was directly translated from Matlab to Python, and built-in Simulink signal blocks were implemented from scratch. After implementation, the accuracy of each module was evaluated via systematic block-by-block validation. The TorchDIVA model is shown to produce outputs that closely match those of the original DIVA model, with a negligible level of error.

# Demo
[This python notebook](https://colab.research.google.com/drive/1YJPNJzjERirTD3zQw3pdTW0AiZjj4GDd?usp=sharing) can be used to experiment with the TorchDIVA model in the browser using Google Colab. A Google account is required to use Google Colab.

# Model Architecture

This simplified schematic illustrates the combination of the feedforward and feedback control loops in the DIVA model.

![DIVA Structural Diagram](https://user-images.githubusercontent.com/14182023/195723655-dcccfca5-20f2-4533-b25d-2e19a3ca942a.png)

# Model Validation

[An example can be viewed here](src/notebooks/motor-mse-demo.ipynb) of how the motor command signal difference between TorchDIVA and the original Matlab DIVA model was computed. Apart from the motor cortex module, all TorchDIVA modules produce equivalent output to their corresponding Matlab module. The source of this deviation is a minor difference in numerical precision between PyTorch and Matlab.

# Setup
The following section lists the requirements for working with TorchDIVA in a local development environment.

# Requirements
- Conda (Recommended): [Anaconda Navigator](https://docs.anaconda.com/navigator/)
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

1. Run main.py using the python command-line or IDE of your choice.

```
cd src/
python ./main.py
```
2. You will be greeted by a menu like the following:

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

3. Use the menu options available to interact with the TorchDIVA model:
List/select speech targets, run speech simulations, etc.

4. Enter 'q' at the main menu to quit the main program loop.

# Contributing to TorchDIVA

1. Follow the steps above to set up the TorchDIVA repository locally for development.
2. Create a branch for your feature with a descriptive name (e.g. sensory-delay-block).
3. Submit a pull request (PR) to merge your feature branch into the main branch.
4. When your PR is approved:
- Your changes will merge into the main development branch.
- Your feature branch will be deleted.
