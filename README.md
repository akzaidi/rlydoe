# Really Doe - Reinforcement Learning Diary of Experiments

## Running Atari Experiments

### Install Atari Roms

In order to play the Atari games, you need to install the emulators for Atari first:

```bash
wget http://www.atarimania.com/roms/Roms.rar -P atari && \
	unrar x -r atari/Roms.rar atari && \
	unzip atari/ROMS.zip -d atari && \
	/opt/conda/envs/rlex/bin/python -m atari_py.import_roms atari/ROMS
```

A provided `atari.yaml` environment file exists for running atari experiments with `stable-baselines3`:

```bash
python trainer-sb3.py environment=atari environment.name=Pong-v0 callbacks.wandb=False
```