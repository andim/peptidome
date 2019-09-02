serve: copycode posts
	jekyll serve -w --config _config.yml,_config_dev.yml

copycode:
	cp ../code . -r
	rm -f code/*/*.md
	rm -f code/*/*.pdf
	rm -f code/*.md
	rm -f code/*/data/ -r
	rm -f code/*/.ipynb_checkpoints/ -r
	rm -f code/.gitignore
	rm -f code/*/.gitignore
	rm -f code/.ipynb_checkpoints/ -r
	rm -f code/lib/ -r
	rm -f code/*.py -r
	rm -f code/*.ipynb -r
	rm -f code/*/*.ipynb -r
	rm -f code/*/__pycache__/ -r
	rm -f code/plots/ -r

posts: nbconvert
	python code_to_posts.py

nbconvert:
	jupyter nbconvert --to markdown ../code/*/*.ipynb --output-dir notebooks
	mv notebooks/*/*png notebook_files/

commitposts:
	git rm -r _posts
	git add _posts/*md

clean:
	rm -rf build *.c *.so *.pyc

