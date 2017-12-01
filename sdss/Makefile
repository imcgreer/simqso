#!/usr/bin/make

ifndef DIR
	DIR := simoutput
endif

ifdef FULLRES
	FORESTFN := "forest_dr9qlf_3000deg2_5000los.fits.gz"
	FOREST := "--forest $(DIR)/$(FORESTFN) --foresttype fullres"
endif

MODELS := bossdr9 dr9expdust ebossdr14 ebossdr14ghwdust dr14expdust

doall: dosims docolorz dofits

dosims:
	for model in $(MODELS) ; do \
		python ebosscore.py $$model -o $(DIR) $(FOREST) $(XARGS) ; \
	done

docolorz:
	for model in $(MODELS) ; do \
		python ebosscolorz.py $(DIR)/$$model".fits" $(XARGS) ; \
	done

dofits:
	for model in $(MODELS) ; do \
		python ebossfit.py $(DIR)/$$model".fits" $(XARGS) ; \
	done

