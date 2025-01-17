autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	flake8

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .
	flake8

test:
	set -e
	coverage run -m pytest tests/

test-cov:
	set -e
	pytest tests/ --cov=./ --cov-report=xml

build-docs:
	set -e
	mkdir -p docs/source/_static
	rm -rf docs/build
	rm -rf docs/source/generated
	cd docs && make html

dev:
	pip install black coverage isort flake8 flake8-bugbear flake8-comprehensions
	pip install sphinx sphinx-rtd-theme recommonmark m2r2
	pip install -r docs/requirements.txt

all: autoformat test build-docs