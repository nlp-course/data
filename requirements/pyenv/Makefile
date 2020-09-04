SHELL:=/bin/bash

# Requisites: pyenv and pyenv-virtualenv
requisites: $(shell uname -s)


Darwin:
	brew update
	brew install pyenv pyenv-virtualenv


Linux:
	(\
		brew install wget; \
	    git clone https://github.com/pyenv/pyenv.git ~/.pyenv; \
	    echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> ~/.bashrc; \
	    echo 'export PATH="$$PYENV_ROOT/bin:$$PATH"' >> ~/.bashrc; \
	    git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv; \
	    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$$(pyenv init -)"\n eval "$$(pyenv virtualenv-init -)"\nfi' >> ~/.bashrc; \
	    . ~/.bashrc; \
	    exec "$$SHELL"; \
    )

env:
	( \
	   pyenv install -s 3.8.3; \
	   pyenv virtualenv -f 3.8.3 otter-latest; \
	   eval "$$(pyenv init -)"; \
	   eval "$$(pyenv virtualenv-init -)"; \
	   pyenv activate otter-latest; \
	   pip install otter-grader==1.0.0; \
	   pip install jupyterlab; \
	   pip install nbconvert; \
    )
