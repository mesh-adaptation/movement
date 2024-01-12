all: install

.PHONY: test

install:
	@echo "Installing dependencies..."
	@python3 -m pip install -r requirements.txt
	@echo "Done."
	@echo "Installing Movement..."
	@python3 -m pip install -e .
	@echo "Done."

lint:
	@echo "Checking lint..."
	@flake8
	@echo "PASS"

test: lint
	@echo "Running test suite..."
	@pytest -v test
	@echo "PASS"

tree:
	@tree -d .
