CC = python3.12
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all plot train pack view metrics report

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/segthor

# CFLAGS = -O
# DEBUG = --debug
EPC = 50
# EPC = 5
BS = 24

K = 5

G_RGX = (Patient_\d+_\d+)_\d+
P_RGX = (Patient_\d+)_\d+_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NET = ENet
# NET = Dummy

TRN = $(RD)/ce \
	$(RD)/residual# \
# 	$(RD)/residual_equalized

# 	$(RD)/box_prior_neg_size_residual

GRAPH = $(RD)/tra_loss.png $(RD)/val_loss.png \
		$(RD)/tra_dice.png $(RD)/val_dice.png \
		$(RD)/val_3d_dsc.png \
		$(RD)/val_3d_hausdorff.png \
		$(RD)/val_3d_hd95.png

# 		$(RD)/val_hausdorff.png \
# HIST =  $(RD)/val_dice_hist.png
HIST =
BOXPLOT = $(RD)/val_dice_boxplot.png \
		$(RD)/val_3d_dsc_boxplot.png
# 		$(RD)/val_hausdorff_boxplot.png \

PLT = $(GRAPH) $(HIST) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-segthor.tar.gz
LIGHTPACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-segthor_light.tar.gz

all: pack

plot: $(PLT)
# plot: $(RD)/val_3d_dsc.png

train: $(TRN)

pack: report $(PACK) $(LIGHTPACK)

$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available
$(LIGHTPACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	$(eval PLTS:=$(filter %.png, $^))
	$(eval FF:=$(filter-out %.png, $^))
	$(eval TGT:=$(addsuffix /best_epoch, $(FF)) $(addsuffix /*.npy, $(FF)) $(addsuffix /best_epoch.txt, $(FF)) $(addsuffix /metrics.csv, $(FF)))
	tar cf - $(PLTS) $(TGT) | pigz > $@
	chmod -w $@


# Dataset
data/segthor: data/segthor.b3sum data/train.zip data/test.zip
	$(info $(yellow)unzip $@$(reset))
	b3sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	mv $@_tmp $@

# 	mv $@_tmp/SEGTHOR_R1.1/* $@_tmp
# 	rmdir $@_tmp/SEGTHOR_R1.1
# 	ls $@_tmp | grep Site
# 	for f in `ls $@_tmp | grep Site` ; do \
# 		ls -1 $@_tmp/$$f >> $@_tmp/all_ids ; \
# 		mv $@_tmp/$$f/* $@_tmp ; \
# 	done
# 	rmdir $@_tmp/Site*
# 	echo `wc -l $@_tmp/all_ids` patients total
# 	sort $@_tmp/all_ids > $@_tmp/sorted_ids
# 	uniq $@_tmp/sorted_ids > $@_tmp/uniq_ids
# 	echo `wc -l $@_tmp/uniq_ids` unique patients


data/SEGTHOR/train/gt data/SEGTHOR/val/gt: | data/SEGTHOR
data/SEGTHOR/train data/SEGTHOR/val: | data/SEGTHOR
data/SEGTHOR: data/segthor
	$(info $(yellow)$(CC) $(CFLAGS) preprocess/slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	$(PP) $(CC) $(CFLAGS) preprocess/slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
		--n_augment 4 --shape 256 256 --retain 10
	mv $@_tmp $@


# Weak labels generation
weaks = data/SEGTHOR/train/centroid data/SEGTHOR/val/centroid \
		data/SEGTHOR/train/erosion data/SEGTHOR/val/erosion \
		data/SEGTHOR/train/random data/SEGTHOR/val/random \
		data/SEGTHOR/train/box data/SEGTHOR/val/box \
		data/SEGTHOR/train/thickbox data/SEGTHOR/val/thickbox
# weak: $(weaks)
weak:

data/SEGTHOR/train/centroid data/SEGTHOR/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/SEGTHOR/train/erosion data/SEGTHOR/val/erosion: OPT = --seed=0 --strategy=erosion_strat --max_iter 9
data/SEGTHOR/train/random data/SEGTHOR/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat
data/SEGTHOR/train/box data/SEGTHOR/val/box: OPT = --seed=0 --margin 0 --strategy=box_strat --allow_overflow --allow_bigger
data/SEGTHOR/train/thickbox data/SEGTHOR/val/thickbox: OPT = --seed=0 --margin=5 --strategy=box_strat --allow_bigger --allow_overflow

$(weaks): | data/SEGTHOR/train/gt data/SEGTHOR/val/gt
	$(info $(yellow)$(CC) $(CFLAGS) gen_weak.py --base_folder=$(@D) --save_subfolder=$(@F)$(reset))
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp \
		--quiet --per_connected_components $(OPT)
	mv $@_tmp $@


# Statistics for high-order constraints
STATS = data/SEGTHOR/stats/size/size.npz data/SEGTHOR/stats/length/length.npz data/SEGTHOR/stats/compactness/compactness.npz
STATS_FOLDERS = $(dir $(STATS))
STATS_PLOTS = $(addsuffix scatter.png, $(STATS_FOLDERS)) \
			  $(addsuffix histogram.png, $(STATS_FOLDERS)) \
			  $(addsuffix kde.png, $(STATS_FOLDERS))
stats: $(STATS) $(STATS_PLOTS)
$(STATS): | data/SEGTHOR/train/gt data/SEGTHOR/train/box
	$(CC) $(CFLAGS) stats_labels.py --ref_folder data/SEGTHOR/train/gt --new_folders data/SEGTHOR/train/box \
		--num_classes $(K) --metric $(notdir $(@D)) --save_dest $@ --class_column 1 $(DEBUG)

$(STATS_PLOTS): plot_stats.py $(STATS)
	$(eval type:=$(subst .png,,$(@F)))
	$(eval metric:=$(notdir $(@D)))
	$(eval source:=$(addsuffix /$(metric).npz,$(@D)))
	$(CC) $(CFLAGS) $< --plot_type $(type) --ref_folder data/SEGTHOR/train/gt --new_folder data/SEGTHOR/train/box \
		--save_dest $@ --metric $(metric) --num_classes $(K) --source $(source)



# Trainings
$(RD)/ce: OPT = --losses="[('CrossEntropy', {'idc': [0, 1, 2, 3, 4]}, None, None, None, 1)]"
$(RD)/ce: data/SEGTHOR/train/gt data/SEGTHOR/val/gt
$(RD)/ce: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

$(RD)/residual: OPT = --losses="[('CrossEntropy', {'idc': [0, 1, 2, 3, 4]}, None, None, None, 1)]"
$(RD)/residual: NET = ResidualUNet
# $(RD)/residual: BS = 44
$(RD)/residual: data/SEGTHOR/train/gt data/SEGTHOR/val/gt
$(RD)/residual: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

$(RD)/residual_equalized: OPT = --losses="[('CrossEntropy', {'idc': [0, 1, 2, 3, 4]}, None, None, None, 1)]"
$(RD)/residual_equalized: NET = ResidualUNet
# $(RD)/residual_equalized: BS = 44
$(RD)/residual_equalized: data/SEGTHOR/train/gt data/SEGTHOR/val/gt
$(RD)/residual_equalized: DATA = --folders="[('img', equalized_png, False), ('gt', gt_transform, True), ('gt', gt_transform, True)]"


# Template
$(RD)/%:
	$(info $(green)$(CC) $(CFLAGS) main.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	printenv > $@_tmp/env.txt
	git diff > $@_tmp/repo.diff
	git rev-parse --short HEAD > $@_tmp/commit_hash
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=$(K) --metric_axis 1 2 3 4 \
		--network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@

# Inference
$(RD)/%/test: data/SEGTHOR/test/img
	$(info $(magenta)$(CC) $(CFLAGS) inference.py $@$(reset))
	rm -rf $@_tmp $@
	$(CC) $(CFLAGS) inference.py --data_folder $< --save_folder $@_tmp \
		--model_weights $(@D)/best.pkl --num_classes $(K)
	mv $@_tmp $@
# 	mv $@_tmp/* $@_tmp/
# 	rmdir $@_tmp/iter000

$(RD)/%/result: $(RD)/%/test
	$(info $(magenta)$(CC) $(CFLAGS) merge_slices.py $@$(reset))
	rm -rf $@_tmp $@
	mkdir $@_tmp
	$(CC) $(CFLAGS) merge_slices.py --data_folder $< --dest_folder $@_tmp \
		--grp_regex "(Patient_\d+)_\d+_\d+" --num_classes $(K)
	mv $@_tmp $@

$(RD)/%/result.zip: $(RD)/%/result
	rm -f $@
	zip -j $@ $</*.nii


# Metrics
metrics: $(TRN) \
	$(addsuffix /val_3d_dsc.npy, $(TRN)) \
	$(addsuffix /val_3d_hausdorff.npy, $(TRN)) \
	$(addsuffix /val_3d_assd.npy, $(TRN)) \
	$(addsuffix /val_3d_hd95.npy, $(TRN))

$(RD)/%/val_3d_dsc.npy $(RD)/%/val_3d_hausdorff.npy $(RD)/%/val_3d_assd.npy $(RD)/%/val_3d_hd95.npy: data/SEGTHOR/val/gt $(RD)/%/
	$(info $(cyan)$(CC) $(CFLAGS) metrics.py $@$(reset))
	$(CC) $(CFLAGS) metrics_overtime.py --basefolder $(@D) --metrics 3d_dsc 3d_hausdorff 3d_hd95 \
		--grp_regex "$(G_RGX)" --resolution_regex "$(P_RGX)" \
		--spacing $(<D)/../spacing.pkl \
		--num_classes $(K) --n_epoch $(EPC) \
		--gt_folder $<


# Plotting
$(RD)/tra_dice.png $(RD)/val_dice.png $(RD)/val_3d_dsc.png: COLS = 1 2 3 4
$(RD)/tra_loss.png $(RD)/val_loss.png: COLS = 0
$(RD)/tra_loss.png $(RD)/val_loss.png: OPT = --dynamic_third_axis
$(RD)/val_3d_dsc.png: | $(addsuffix /val_3d_dsc.npy, $(TRN))

$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: COLS = 1 2 3 4
$(RD)/val_3d_hausdorff.png $(RD)/val_3d_hd95.png: OPT = --ylim 0 50 --min
$(RD)/val_3d_hausdorff.png: | $(addsuffix /val_3d_hausdorff.npy, $(TRN))
$(RD)/val_3d_hd95.png: | $(addsuffix /val_3d_hd95.npy, $(TRN))
$(GRAPH): plot.py $(TRN)

$(RD)/val_dice_boxplot.png $(RD)/val_3d_hausdorff_boxplot.png: COLS = 1 2 3 4
$(RD)/val_3d_dsc_boxplot.png: COLS = 1 2 3 4
$(RD)/val_3d_hausdorff_boxplot.png: OPT = --epc 99 --ylim 0 50
$(RD)/val_dice_boxplot.png: OPT = --epc 99
$(RD)/val_3d_dsc_boxplot.png: OPT = --epc 99
$(RD)/val_3d_dsc_boxplot.png: | $(addsuffix /val_3d_dsc.npy, $(TRN))
$(RD)/val_3d_hausdorff_boxplot.png: | $(addsuffix /val_3d_hausdorff.npy, $(TRN))
$(BOXPLOT): moustache.py $(TRN)

$(RD)/%.png:
	$(info $(blue)$(CC) $(CFLAGS) $< $@$(reset))
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless $(OPT) $(DEBUG)


# Viewing
view: $(TRN) | data/SEGTHOR weak
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 2 --img_source data/SEGTHOR/val/img data/SEGTHOR/val/gt \
		$(addsuffix /best_epoch/val, $^) \
		--display_names gt $(notdir $^) \
		-C $(K) --cmap Set1 --legend \
		--class_names background  esophagus heart trachea aorta \
		--no_contour --show_img --alpha .8


view_train: | data/SEGTHOR weak
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 1 --img_source data/SEGTHOR/train/img data/SEGTHOR/train/gt \
		$(addsuffix /best_epoch/train, $^) \
		--display_names gt $(notdir $^) \
		-C $(K) --cmap Set1 --legend \
		--class_names background  esophagus heart trachea aorta \
		--no_contour

view_test: $(addsuffix /test, $(TRN)) | data/SEGTHOR weak
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) $(CFLAGS) viewer/viewer.py -n 1 --img_source data/SEGTHOR/test/img data/SEGTHOR/test/gt \
		$^ \
		--display_names gt $(notdir $^) \
		-C $(K) --cmap Set1 --legend \
		--class_names background  esophagus heart trachea aorta \
		--no_contour

report: $(TRN) | metrics
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_dice val_3d_dsc --axises 1 2 3 4 \
		--precision 3 --detail_axises
