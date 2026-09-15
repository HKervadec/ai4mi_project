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

data/SEGTHOR: data/segthor_part1
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--shape 256 256 --retain 5
	mv $@_tmp $@

## Corrected GT (see fix_gt.py) -- shared source for the measurement experiments.
## No prerequisite on data/segthor_part1 on purpose: the source dirs are expected
## to be already extracted, and depending on data/segthor_part1 would drag in its
## .zip rule (which fails when the zip is absent). slice_segthor.py / fix_gt.py
## assert their source exists and error clearly if it is missing.
data/segthor_fixed:
	$(info $(yellow)python fix_gt.py$(reset))
	python fix_gt.py --source_dir data/segthor_part1/train --dest data/segthor_fixed/train

## Per-experiment sliced datasets (options come from experiments.py).
## All grouped under data/experiments/<name> to keep data/ uncluttered; add a
## new one by registering an experiment and copying a rule. The dataset dir is
## the switch: train on it with `python main.py --experiment <name>`.
## Prerequisite: data/segthor_part1 (original) or data/segthor_fixed (baseline,
## hu_window) must exist first -- run `make data/segthor_fixed` if needed.
data/experiments/original:
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment original --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/baseline:
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment baseline --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/hu_window:
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment hu_window --dest_dir $@_tmp
	mv $@_tmp $@

.PHONY: data-experiments
data-experiments: data/experiments/original data/experiments/baseline data/experiments/hu_window
