red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
## Original one
data/segthor_part1: data/segthor_part1.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_part1.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

data/SEGTHOR:
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--shape 256 256 --retain 5
	mv $@_tmp $@

data/SEGTHOR_CLEAN:
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1_cleaned --dest_dir $@_tmp \
		--shape 256 256 --retain 5
	mv $@_tmp $@

PROC ?= -1

data/totalseg_part1/.done:
	$(info $(green)python prepare_totalseg.py$(reset))
	python prepare_totalseg.py --source_dir data/Totalsegmentator_dataset_small_v201 \
		--dest_dir $(@D) -p $(PROC)
	touch $@

data/TOTALSEG: data/totalseg_part1/.done
	$(info $(green)python -O slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python -O slice_segthor.py --source_dir $(<D) --dest_dir $@_tmp \
		--shape 256 256 --retain 10
	mv $@_tmp $@
