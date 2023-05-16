import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import glob
import hydra
import jams
import librosa
import numpy as np
import os
import pretty_midi
from miditoolkit.midi import parser as mid_parser
import scipy
import soundfile as sf
from math import ceil, floor
from multiprocessing import Pool
from itertools import repeat
from omegaconf import DictConfig, OmegaConf
from src import utils


def process_cqt(
    audio,
    tempo,
    original_sr,
    track_name,
    split_unit_in_bars,
    split_hop_bar_len,
    hparams,
    dataset_name,
    output_dir,
):

    len_in_sec = librosa.get_duration(y=audio, sr=original_sr)
    len_in_bars = int(round((tempo * len_in_sec) / (4 * 60)))
    n_units = int(
        round((tempo * len_in_sec) / (split_hop_bar_len * split_unit_in_bars * 60))
        - ((split_unit_in_bars - split_hop_bar_len) // split_hop_bar_len)
    )
    audio = audio.astype(float)
    audio = librosa.resample(
        audio, orig_sr=original_sr, target_sr=hparams["down_sampling_rate"]
    )
    if hparams["normalize_wave"]:
        audio = librosa.util.normalize(audio)
    cqt = librosa.cqt(
        audio,
        hop_length=hparams["hop_length"],
        sr=hparams["down_sampling_rate"],
        n_bins=hparams["total_n_bins"],
        bins_per_octave=hparams["bins_per_octave"],
    )
    if hparams["db_scale"]:
        cqt = librosa.amplitude_to_db(np.abs(cqt))
    if hparams["normalize_cqt"]:
        cqt = utils.z_score_normalize(cqt)
    cqt = np.abs(cqt).T

    for unit_n in range(n_units):
        cqt_filename = os.path.join(
            output_dir, dataset_name, "cqt", f"{track_name}_0{unit_n}.npy"
        )
        st = int(round(len(cqt) * unit_n * split_hop_bar_len / len_in_bars))
        end = (
            int(
                round(len(cqt))
                * ((unit_n * split_hop_bar_len + split_unit_in_bars) / len_in_bars)
            )
            if int(
                round(len(cqt))
                * ((unit_n * split_hop_bar_len + split_unit_in_bars) / len_in_bars)
            )
            <= len(cqt)
            else len(cqt)
        )
        np.save(cqt_filename, cqt[st:end])

def jams_to_midi(jam, track_name, split_unit_in_bars, split_hop_bar_len, output_dir):
    tempo = float(track_name.split("-")[1])
    second_per_unit = float(60.0 * 4 * split_unit_in_bars) / tempo
    second_per_hop = float(60.0 * 4 * split_hop_bar_len) / tempo
    split_midi_list = []
    midi_ch_list = []
    block_range_list = []
    n_units = int(
        round(
            (tempo * jam.file_metadata.duration)
            / (split_hop_bar_len * split_unit_in_bars * 60)
        )
        - ((split_unit_in_bars - split_hop_bar_len) // split_hop_bar_len)
    )

    for block_n in range(n_units):
        split_midi_list.append(pretty_midi.PrettyMIDI(initial_tempo=tempo))
        midi_ch_list.append(pretty_midi.Instrument(program=25, name=track_name))
        start = second_per_hop * block_n
        end = start + second_per_unit
        block_range_list.append((start, end))

    annos = jam.search(namespace="note_midi")
    for string, anno in enumerate(annos):
        for note in anno:
            pitch = int(round(note.value))
            bend_amount = int(round((note.value - pitch) * 4096))

            st = note.time
            dur = note.duration

            for i, (block_st, block_ed) in enumerate(block_range_list):
                if block_st <= st and st < block_ed:
                    rel_st = st - block_st

                    # rel_st = st % second_per_unit

                    n = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=rel_st,
                        end=min(rel_st + dur, second_per_unit),
                    )
                    midi_ch_list[i].notes.append(n)

                elif block_st <= st + dur and st + dur < block_ed:
                    n = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=0,
                        end=st + dur - block_st,
                    )
                    midi_ch_list[i].notes.append(n)

    for unit_n in range(n_units):
        split_midi_list[unit_n].instruments.append(midi_ch_list[unit_n])
        midi_filename = os.path.join(
            output_dir, "guitarset", "midi", f"{track_name}_0{unit_n}.mid"
        )
        split_midi_list[unit_n].write(midi_filename)

def split_midi(
    midi,
    track_name,
    split_unit_in_bars,
    split_hop_bar_len,
    dataset_name,
    output_dir,
):
    tempo = midi.get_tempo_changes()[1][0]
    len_in_sec = midi.get_end_time()
    second_per_unit = float(60.0 * 4 * split_unit_in_bars) / tempo
    second_per_hop = float(60.0 * 4 * split_hop_bar_len) / tempo
    n_units = int(
        round((tempo * len_in_sec) / (split_hop_bar_len * split_unit_in_bars * 60))
        - ((split_unit_in_bars - split_hop_bar_len) // split_hop_bar_len)
    )
    split_midi_list = []
    split_midi_ch_list = []
    block_range_list = []
    for block_n in range(n_units):
        split_midi_list.append(pretty_midi.PrettyMIDI(initial_tempo=tempo))
        split_midi_ch_list.append(pretty_midi.Instrument(program=25, name=track_name))
        start = second_per_hop * block_n
        end = start + second_per_unit
        block_range_list.append((start, end))

    notes = midi.instruments[0].notes
    for note in notes:
        for i, (block_st, block_ed) in enumerate(block_range_list):
            if block_st <= note.start < block_ed:
                rel_st = note.start - block_st
                rel_ed = min(note.end - block_st, block_ed)
                new_note = pretty_midi.Note(
                    velocity=100,
                    pitch=note.pitch,
                    start=rel_st,
                    end=rel_ed,
                )
                split_midi_ch_list[i].notes.append(new_note)

            elif block_st <= note.end < block_ed:
                rel_st = 0
                rel_ed = min(note.end - block_st, block_ed)
                new_note = pretty_midi.Note(
                    velocity=100,
                    pitch=note.pitch,
                    start=rel_st,
                    end=rel_ed,
                )
                split_midi_ch_list[i].notes.append(new_note)

    for unit_n in range(n_units):
        split_midi_list[unit_n].instruments.append(split_midi_ch_list[unit_n])
        midi_filename = os.path.join(
            output_dir, dataset_name, "midi", f"{track_name}_0{unit_n}.mid"
        )
        split_midi_list[unit_n].write(midi_filename)


def midi_to_wav(midi, track_name, output_dir):
    sr = 22050
    wave = midi.synthesize(fs=sr, wave=scipy.signal.square)
    audio_path = os.path.join(output_dir, "classic_guitar", "wav", track_name + ".wav")
    len_in_sec = len(wave) / sr - 1
    sf.write(audio_path, wave[: int(len_in_sec * sr)], sr)
    return audio_path, sr


def check_midi_validity(midi_path):
    midi = mid_parser.MidiFile(midi_path)
    if (
        len(midi.tempo_changes) == 1
        and len(midi.time_signature_changes) == 1
        and midi.time_signature_changes[0].numerator
        == midi.time_signature_changes[0].denominator
        == 4
        and len(midi.instruments) == 1
        and (midi.instruments[0].program == 24 or midi.instruments[0].program == 25)
    ):
        validity = True
        time = midi.max_tick / midi.ticks_per_beat / midi.tempo_changes[0].tempo
    else:
        validity = False
        time = 0
    return midi_path, validity, time


def guitarset_preprocess(args) -> None:
    audio_path, cfg = args
    track_name = os.path.split(audio_path)[1][:-8]
    tempo = float(track_name.split("-")[1])
    jams_data = jams.load(
        os.path.join(
            cfg.data_dir, "guitarset", "annotation", "jams", f"{track_name}.jams"
        )
    )

    jams_to_midi(
        jams_data,
        track_name,
        cfg.split_unit_in_bars,
        cfg.split_hop_bar_len,
        cfg.output_dir,
    )

    audio, original_sr = librosa.load(audio_path)
    process_cqt(
        audio,
        tempo,
        original_sr,
        track_name,
        cfg.split_unit_in_bars,
        cfg.split_hop_bar_len,
        cfg.cqt_hparams,
        "guitarset",
        cfg.output_dir,
    )


def classic_guitar_preprocess(args) -> None:
    midi_path, cfg = args
    track_name = os.path.split(midi_path)[1][:-4]
    midi = pretty_midi.PrettyMIDI(midi_path)
    tempo = midi.get_tempo_changes()[1][0]
    audio_path, sr = midi_to_wav(
        midi,
        track_name,
        cfg.output_dir,
    )
    audio, original_sr = librosa.load(audio_path)
    process_cqt(
        audio,
        tempo,
        original_sr,
        track_name,
        cfg.split_unit_in_bars,
        cfg.split_hop_bar_len,
        cfg.cqt_hparams,
        "classic_guitar",
        cfg.output_dir,
    )

    split_midi(
        midi,
        track_name,
        cfg.split_unit_in_bars,
        cfg.split_hop_bar_len,
        "classic_guitar",
        cfg.output_dir,
    )


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="data_preprocess.yaml"
)
def main(cfg: DictConfig) -> None:
    audio_path_list = sorted(glob.glob(cfg.audio_dir + "*.wav"))
    pool = Pool(processes=cfg.n_workers)
    pool.close()


if __name__ == "__main__":
    main()
