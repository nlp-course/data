# Machine Setup

This class is focused on in-class collaborative labs and a sequence of problem sets, so it is important that you setup environments properly before lab classes start (Sep 10). 

Throughout this readme file, we assume that the working directory is `requirements`.

## Operating System

If you want to work locally using your own computer, we recommend to use Unix-based operating systems, such as Linux and MacOS. While on Windows you can still setup virtual environments, we highly recommend using [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) or running a virtual machine with Ubuntu, such as by using [VirtualBox](https://www.virtualbox.org/). 

Alternatively, you can also use [Google Colab](https://colab.research.google.com/) which provides virtual machines on the cloud to run IPython notebooks. Just be aware that everything saved to disk will be lost unpon disconnection (which happens automatically if left idle for too long), unless you connect to your Google Drive and save files explicitly there.


## Prerequisites

If you have anaconda installed (type `which conda` in a terminal and see if you got any output), please skip this part. Throughout this course, we will use `Python 3.8`. To not mess up with your other Python environments, we will create a virtual environment for this course. In our provided Makefile, we use `conda` for managing virtual environments. Alternatively, you can use `pyenv` and `virtualenv` instead by following instructions in `pyenv/`.

```
make requisites
```

After installing `conda` using the above command, we need to create a virtual environment named `cs187`:

```
make env
```

## Activate Virtual Environment

You will need to activate this virtual environment every time you launch a new terminal using

```
conda activate cs187
```

Alternatively, you can add it to your `$HOME/.bashrc` file to automatically activate it.

## Lab/PSet Python Dependencies

```
pip install -r requirements.txt
```

## IPython Notebook

All labs and problem sets will be in IPython notebooks. If you have not used IPython notebooks before, it might be a good idea to get familiar with it before class. A tutorial can be found at [this link](https://realpython.com/jupyter-notebook-introduction/). Upon installing all above dependencies, you can launch Jupyter notebook using

```
jupyter notebook
```
