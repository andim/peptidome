.PHONY: cython

all: netmhc 

download:
	python download.py

netmhc: download
	python netmhcrun.py

clean:
	rm -rf build *.c *.so *.pyc

