#!/usr/bin/env bash

shopt -s expand_aliases

alias steinbock="docker run -v /mnt/data/git/spellmatch/data/datasets/burger_windhager_2022:/data -u $(id -u):$(id -g) ghcr.io/bodenmillergroup/steinbock:0.13.5"

# # panel.csv has been manually annotated!
# steinbock preprocess imc panel

# # images.csv has been manually annotated!
# steinbock preprocess imc images --hpf 50

# steinbock segment deepcell --minmax --type nuclear
