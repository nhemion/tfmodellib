install:
	python setup.py install

test:
	python -m doctest tfmodels/tfmodel.py
