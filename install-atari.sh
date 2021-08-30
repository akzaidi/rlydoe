#!/usr/bin/env bash

mkdir atari
wget http://www.atarimania.com/roms/Roms.rar -P atari && \
	unrar x -r atari/Roms.rar atari && \
	unzip atari/ROMS.zip -d atari && \
	/opt/conda/envs/rlex/bin/python -m atari_py.import_roms atari/ROMS