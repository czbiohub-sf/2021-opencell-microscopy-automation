init:
	pip install -r requirements.txt

lint:
	flake8 ./dragonfly_automation --count --statistics --exit-zero
	python -m pylint ./dragonfly_automation

test:
	pytest -v
