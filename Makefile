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
	@flake8 --ignore=E501,E226,E402,E731,E741,F403,F405,F999,N803,N806,W503
	@echo "PASS"

test: lint
	@echo "Running test suite..."
	@pytest -v test
	@echo "PASS"

tree:
	@tree -d .
