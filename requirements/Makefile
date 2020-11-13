SHELL:=/bin/bash

# Requisites: pyenv and pyenv-virtualenv
requisites: $(shell uname -s)


Darwin:
	(\
		brew install wget; \
	    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh; \
	    bash ~/miniconda.sh -b -p $$HOME/miniconda; \
	    $$HOME/miniconda/condabin/conda init; \
	    source ~/.bash_profile; \
	    source ~/.bashrc; \
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
       conda create -y -n cs187 python=3.8.3;\
    )
