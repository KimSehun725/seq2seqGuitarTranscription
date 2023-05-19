# seq2seqGuitarTranscription
## File structure

```
seq2seqGuitarTranscription
├── configs
│   ├── callbacks
│   ├── data_preprocess
│   ├── datamodule
│   ├── extras
│   ├── hydra
│   ├── local_env
│   ├── logger
│   ├── model
│   ├── paths
│   ├── tokenizer
│   ├── trainer
│   └── train.py
├── data
│   └── classic_guitar_midi
├── src
│   ├── data_preprocess
│   ├── datamodules
│   ├── models
│   ├── tokenizer
│   ├── utils
│   └── train.py
└── requiments.txt
```

## Details
### configs/
This directory contains all the config files managed by hydra.

### data/
This directory is for storing the MIDI data from classic guitar MIDI archives, the GuitatSet and preprocessed data of the datasets.

### src/
This directory contains the source codes.

### [requirements.txt](https://github.com/KimSehun725/seq2seqGuitarTranscription/blob/main/requirements.txt)
This file contains the informations on python packages required to run this project.

## Setup
First, make `data/guitarset/` directory and subdirectories, and download GuitarSet with following codes:
```
mkdir -p data/guitarset/annotation data/guitarset/audio_mono-mic
wget -qO- https://zenodo.org/record/3371780/files/annotation.zip?download=1 | busybox unzip - -d data/guitarset/annotation
wget -qO- https://zenodo.org/record/3371780/files/audio_mono-mic.zip?download=1 | busybox unzip - -d data/guitarset/audio_mono-mic
```

Next, create virtual environment and install all the necessary packages using `pip`. Run the following codes to do so.
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the code

In all of the following instructions, we assume that you are in `seq2seqGuitarTranscription/` directory.

### Basic usage
Running [src/train.py](https://github.com/KimSehun725/seq2seqGuitarTranscription/blob/main/src/train.py) with command line arguments for overriding default configs can do most of the thing that you need.

We use Wandb for logging loss, metrics, VRAM usage, etc. so we recommend making an account and pass your API key when it ask for it.

### Pre-training with synthetic audio-MIDI pair data
To train with default settings using gpu, run the following command.
```
python3 src/train.py trainer=gpu trainer.max_epochs=32 datamodule=classic_guitar logger.wandb.group="NAME_OF_THIS_RUN"
```
Running this command will do:
1. Preprocess data (MIDI files from classic guitar MIDI archive) from raw MIDI files to 4 bar long MIDI files, synthesized .wav file and corresponding CQT.
2. Training the model with default network settings for 32 epochs using GPU. During the training process, model will be saved in `logs/train_hybrid_ctc/runs/DATETIME/checkpoints` every 4 epochs.
3. Testing the trained model with test set. During testing, estimated MIDI and figures will be saved to `logs/train_hybrid_ctc/runs/DATETIME/`. 

Note that all the metrics and losses will be logged with logger in real-time, and printed to standard output after training or testing process is over.

### Finetuning with GuitarSet
To train with default settings using gpu, run the following command.
```
python3 src/train.py trainer=gpu trainer.max_epochs=128 group="NAME_OF_THIS_RUN" datamodule=guitarset ckpt_path="logs/train_hybrid_ctc/runs/DATETIME/checkpoints/epoch_031.ckpt" datamodule.test_player_n=0
```
Running this command will do:
1. Convert annotation of GuitarSet from jams files to MIDI files and preprocess the data to 4 bar long CQT and corresponding MIDI.
2. Load the pretrained model.
3. Training the pretrained model using training data (player number 1 to 5) and default network settings for 96 epochs (32 + 96 = 128) using GPU. During the training process, model will be saved in `logs/train_hybrid_ctc/runs/DATETIME/checkpoints` every 4 epochs.
4. Testing the trained model with test set (using player number 0). During testing, estimated MIDI and figures will be saved to `logs/train_hybrid_ctc/runs/DATETIME/`. 