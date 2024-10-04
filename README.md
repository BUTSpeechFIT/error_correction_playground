# Error Correction Playground for multi-talker meeting transcription

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a playground repository to evaluate performance of different models on task of error correction
on [NOTSOFAR-1](https://www.chimechallenge.org/current/task2/index) dataset (more datasets are coming soon).

Predictions in `NSF_predictions_GT_diar` were made using the GT diarization and BUT/JHU CHIME-8 NOTSOFAR-1 system. Please
see references for details.

## Dataset

The predictions are located in `predictions_GT_diar` folder. Each session has corresponding directory that
contains `ref.json` and `(tc_orc_wer|tcp_wer)_hyp.json` files. Each `.json` file contains list of segments in the
following format:

```json
  {
  "session_id": "singlechannel/MTG_32000_meetup_0",
  "start_time": 61.3900000000000005684341886080801486968994140625,
  "end_time": 63.6700000000000017053025658242404460906982421875,
  "words": "we should probably list the",
  "speaker": "Ron"
}
```

## Environment Setup

1. Clone the repository
2. Cd into the repository

```bash
cd error_correction_playground
```

3. (Optional) Install virtualenv

```bash
pip install virtualenv
```

4. Create a virtual environment

```bash
virtualenv venv
```

5. Activate the virtual environment

```bash
source venv/bin/activate
```

6. Install the requirements

```bash
pip install -r requirements.txt
```

## Usage

To evaluate the performance of the models, copy the predictions to separate directory, run your error correction system
and save the predictions in the same format as the original predictions. Then run the following command to evaluate the
performance of the system (optionally you can also compute `tc-orc-wer` by setting `--compute_orc` flag):

```bash

```bash
NEW_PREDICTIONS_DIR= # Path to the new predictions directory
python score.py --predictions_dir $NEW_PREDICTIONS_DIR  --save_visualizations --collar 5 --text_norm chime8
```

You shall see the following output:

```bash
2024-10-04 14:55:22,864 [INFO] [wer]  Metrics: {'cp_wer': 0.20452815011744505, 'cp_errors': 315.63125, 'cp_length': 1473.375, 'cp_insertions': 93.96875, 'cp_deletions': 51.79375, 'cp_substitutions': 169.86875, 'cp_missed_speaker': 0.0, 'cp_falarm_speaker': 0.0, 'cp_scored_speaker': 4.7375, 'tcp_wer': 0.20863855882623233, 'tcp_errors': 322.075, 'tcp_length': 1473.375, 'tcp_insertions': 99.88125, 'tcp_deletions': 57.70625, 'tcp_substitutions': 164.4875, 'tcp_missed_speaker': 0.0, 'tcp_falarm_speaker': 0.0, 'tcp_scored_speaker': 4.7375, 'tcorc_wer': 0.20292262780954032, 'tcorc_errors': 312.3625, 'tcorc_length': 1473.375, 'tcorc_insertions': 92.4875, 'tcorc_deletions': 51.9625, 'tcorc_substitutions': 167.9125}
```
You can also see per single session metrics in `all_session_wer.csv` file.

For each session visualization of the errors will be saved in `viz.htlm` file.

## References

```bibtex
@inproceedings{polok24_interspeech,
  title     = {BUT/JHU System Description for CHiME-8 NOTSOFAR-1 Challenge},
  author    = {Alexander Polok, Dominik Klement, Jiangyu Han, Šimon Sedláček, Bolaji Yusuf, Matthew Maciejewski, Matthew Wiesner, Lukáš Burget},
  year      = {2024},
  booktitle = {Interspeech 2024},
}
```

```bibtex
@misc{polok2024targetspeakerasrwhisper,
      title={Target Speaker ASR with Whisper}, 
      author={Alexander Polok and Dominik Klement and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
      year={2024},
      eprint={2409.09543},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2409.09543}, 
}
```

```bibtex
@inproceedings{vinnikov24_interspeech,
  title     = {NOTSOFAR-1 Challenge: New Datasets, Baseline, and Tasks for Distant Meeting Transcription},
  author    = {Alon Vinnikov and Amir Ivry and Aviv Hurvitz and Igor Abramovski and Sharon Koubi and Ilya Gurvich and Shai Peer and Xiong Xiao and Benjamin Martinez Elizalde and Naoyuki Kanda and Xiaofei Wang and Shalev Shaer and Stav Yagev and Yossi Asher and Sunit Sivasankaran and Yifan Gong and Min Tang and Huaming Wang and Eyal Krupka},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {5003--5007},
  doi       = {10.21437/Interspeech.2024-1788},
  issn      = {2958-1796},
}
```
