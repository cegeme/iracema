.PHONY: clean lint lint-full tests isort docs 

# python version
PYTHON_VERSION     = 3
PYTHON_INTERPRETER = python$(PYTHON_VERSION)
PIP_COMMAND        = pip$(PYTHON_VERSION)

# checking the operating system (mac os or linux)
UNAME :=$(shell uname -s)

# set the open command according to the operating system
ifeq ($(UNAME), Darwin)
OPENCMD := open
else ifeq ($(UNAME), Linux)
OPENCMD := xdg-open
else
$(error Your operating system is not supported)
endif

.PHONY: clean
clean:
	find iracema/. -name '*.pyc' -exec rm --force {} +; \
	find iracema/. -name '*.pyo' -exec rm --force {} +; \
	find iracema/. -name '__pycache__' -exec rm -rf {} +

.PHONY: lint
lint:
	pylint3 -E -f colorized iracema/*

.PHONY: lint-full
lint-full:
	pylint3 -f colorized iracema

.PHONY: tests
tests:
	pytest

.PHONY: isort
isort:
	isort -rc iracema/.

.PHONY: docs
docs: 
	cd docs; make clean; make html; cd ..; \
	$(OPENCMD) docs/_build/html/index.html

# TODO: this fix for matplotlib is not solving the problem
.PHONY: fix-matplotlib-mac
fix-matplotlib-mac:
	$(eval MATPLOTLIB_CONFIG_DIR = ${HOME}/.matplotlib)
	$(eval MATPLOTLIB_CONFIG_FILE = $(MATPLOTLIB_CONFIG_DIR)/matplotlibrc)
	@if [ "$(UNAME)" = "Darwin" ]; then \
		if ! [ -d $(MATPLOTLIB_CONFIG_DIR) ]; then \
			mkdir $(MATPLOTLIB_CONFIG_DIR); \
		elif [ -f $(MATPLOTLIB_CONFIG_FILE) ]; then \
			echo "Change matplotlib backend or the virtualenv won't work."; \
			echo "Add this to the file $(MATPLOTLIB_CONFIG_FILE):"; \
			echo "backend: agg"; \
		else \
			echo "Configuring matplotlib backend."; \
			echo "backend: agg" > $(MATPLOTLIB_CONFIG_FILE); \
		fi; \
	fi
