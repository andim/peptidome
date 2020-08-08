serve: copycode posts
	jekyll serve -w --config _config.yml,_config_dev.yml

copycode:
	cp ../code . -r
	rm -f code/*/*.md
	rm -f code/*/*.pdf
	rm -f code/*/submit*.sh
	rm -f code/*.md
	rm -f code/requirements.txt
	rm -f code/*/Makefile
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

commit: copycode nbconvert
	git rm -rf _posts
	mkdir _posts
	python code_to_posts.py
	git add _posts/*md
	git add code/*/*png code/*/*py notebook_files/*png code/*/*/*png

clean:
	rm -rf build *.c *.so *.pyc

