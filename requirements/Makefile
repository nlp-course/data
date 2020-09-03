SHELL:=/bin/bash

# Requisites: pyenv and pyenv-virtualenv
requisites: $(shell uname -s)


Darwin:
	(\
	    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh; \
	    bash ~/miniconda.sh -b -p $$HOME/miniconda; \
	    $$HOME/miniconda/condabin/conda init; \
	    exec "$$SHELL"; \
    )


Linux:
	(\
		wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh; \
        bash ~/miniconda.sh -b -p $$HOME/miniconda; \
	    $$HOME/miniconda/condabin/conda init; \
	    exec "$$SHELL"; \
    )

env:
	( \
       conda create -y -n otter-latest python=3.8.3;\
	   conda activate otter-latest; \
	   pip install otter-grader==1.0.0; \
	   pip install jupyterlab; \
	   pip install nbconvert; \
    )
