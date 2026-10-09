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

## Legacy single-dataset slice for `main.py --dataset SEGTHOR`. Slices from the
## corrected GT (data/gt/watershed_refined, built as an order-only prereq), which
## holds both the CT and the repaired GT. The modular pipeline does NOT use this
## target -- it slices once via `make data-cache` (see below). For per-technique
## experiments use run.py with a config (PIPELINE_PLAN.md), not main.py.
data/SEGTHOR: | data/gt/watershed_refined
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir data/gt/watershed_refined --dest_dir $@_tmp \
		--shape 256 256 --retain 5
	mv $@_tmp $@

## Corrected-GT variants, one folder per fix method, grouped under data/gt/.
## No prerequisite on data/segthor_part1 on purpose: it is expected to be already
## extracted, and depending on it would drag in its .zip rule (which fails when
## the zip is absent). The fixers assert their source exists and error clearly.
## The two fixers take slightly different CLIs -- this is where that is absorbed.
data/gt/watershed:
	$(info $(yellow)python fix_gt.py -> $@$(reset))
	rm -rf $@_tmp $@
	python fix_gt.py --source_dir data/segthor_part1/train --dest $@_tmp/train
	mv $@_tmp $@

data/gt/watershed_refined: 
	$(info $(yellow)python fix_segthor_gt.py -> $@$(reset))
	rm -rf $@_tmp $@
	python fix_segthor_gt.py --source_dir data/segthor_part1 --dest_dir $@_tmp
	mv $@_tmp $@

## Per-experiment sliced datasets (options come from experiments.py), named
## <gtfix>_<intensity> and grouped under data/experiments/<name>. The dataset dir
## is the switch: train on it with `python main.py --experiment <name>`. GT-fix
## sources are order-only prereqs (|), so a missing one is built and an existing
## one is used as-is. `original` slices from data/segthor_part1 (must exist).
data/experiments/original:
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment original --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/watershed_minmax: | data/gt/watershed
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment watershed_minmax --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/watershed_window: | data/gt/watershed
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment watershed_window --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/watershed_window_resampled: | data/gt/watershed
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment watershed_window_resampled --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/refined_window: | data/gt/watershed_refined
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment refined_window --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/refined_ce_dice: | data/gt/watershed_refined
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment refined_ce_dice --dest_dir $@_tmp
	mv $@_tmp $@

data/experiments/refined_window_resampled: | data/gt/watershed_refined
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --experiment refined_window_resampled --dest_dir $@_tmp
	mv $@_tmp $@

.PHONY: data-experiments
data-experiments: data/experiments/original data/experiments/watershed_minmax \
                  data/experiments/watershed_window data/experiments/watershed_window_resampled \
                  data/experiments/refined_window data/experiments/refined_window_resampled

## Modular pipeline (see PIPELINE_PLAN.md): all patients sliced once for run.py.
## run.py builds it automatically when missing; this target only does it ahead of time.
## Resampling (target_spacing) gets its own cache folder, so switching it on/off
## never reuses the wrong slices -- run.py rebuilds the cache automatically.
.PHONY: data-cache
data-cache: | data/gt/watershed_refined
	python -m segpipe.data --config configs/current.yaml
