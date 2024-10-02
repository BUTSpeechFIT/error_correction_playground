import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import meeteval
import numpy as np
import pandas as pd
from meeteval.io.seglst import SegLstSegment
from meeteval.viz.visualize import AlignmentVisualization
from meeteval.wer.wer.orc import OrcErrorRate

from src.logging_def import get_logger
from src.text_norm_whisper_like import get_txt_norm
from src.utils import normalize_segment, create_vad_mask, find_group_splits, map_utterance_to_split, \
    agregate_errors_across_groups

_LOG = get_logger('wer')


def save_wer_visualization(ref, hyp, out_dir):
    ref = ref.groupby('session_id')
    hyp = hyp.groupby('session_id')
    assert len(ref) == 1 and len(hyp) == 1, 'expecting one session for visualization'
    assert list(ref.keys())[0] == list(hyp.keys())[0]

    meeting_name = list(ref.keys())[0]
    av = AlignmentVisualization(ref[meeting_name], hyp[meeting_name], alignment='tcp')
    # Create standalone HTML file
    av.dump(os.path.join(out_dir, 'viz.html'))


def calc_session_tcp_wer(ref, hyp, collar):
    res = meeteval.wer.tcpwer(reference=ref, hypothesis=hyp, collar=collar)

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
            'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'tcp_' + k for k in keys})
            .rename(columns={'tcp_error_rate': 'tcp_wer'}))


def calc_session_tcorc_wer(ref, hyp, collar):
    res = meeteval.wer.tcorcwer(reference=ref, hypothesis=hyp, collar=collar)

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'tcorc_' + k for k in keys})
            .rename(columns={'tcorc_error_rate': 'tcorc_wer'}))


def calc_session_cp_wer(ref, hyp):
    res = meeteval.wer.cpwer(reference=ref, hypothesis=hyp)

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
            'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'cp_' + k for k in keys})
            .rename(columns={'cp_error_rate': 'cp_wer'}))


def calc_session_approx_orc_wer(ref, hyp, group_duration=15, time_step=0.1):
    ref_vad = create_vad_mask(ref.segments, time_step=time_step)
    hyp_vad = create_vad_mask(hyp.segments, time_step=time_step)
    max_vad_len = max(len(ref_vad), len(hyp_vad))
    ref_vad = np.pad(ref_vad, (0, max_vad_len - len(ref_vad)))
    hyp_vad = np.pad(hyp_vad, (0, max_vad_len - len(hyp_vad)))
    vad = ref_vad | hyp_vad
    splits = np.array(find_group_splits(vad, group_duration=group_duration, time_step=time_step)) * time_step

    ref_grouped = ref.map(
        lambda seg: SegLstSegment(
            **{"session_id": seg['session_id'] + str(map_utterance_to_split(float(seg['start_time']), splits)),
               "start_time": seg['start_time'],
               "end_time": seg['start_time'],
               "speaker": seg['speaker'],
               "words": seg['words']}))
    hyp_grouped = hyp.map(
        lambda seg: SegLstSegment(
            **{"session_id": seg['session_id'] + str(map_utterance_to_split(float(seg['start_time']), splits)),
               "start_time": seg['start_time'],
               "end_time": seg['start_time'],
               "speaker": seg['speaker'],
               "words": seg['words']}))

    res = meeteval.wer.orcwer(reference=ref_grouped, hypothesis=hyp_grouped)
    res = agregate_errors_across_groups(res, ref.segments[0]['session_id'])
    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'approx_orc_' + k for k in keys})
            .rename(columns={'approx_orc_error_rate': 'orc_wer'}))


def calc_wer(out_dir: Path,
             tcp_wer_hyp_json: Path,
             tcorc_wer_hyp_json: Path,
             ref_json: Path,
             collar: int = 5,
             save_visualizations: bool = False,
             compute_orc: bool = False,
             tn: Optional[Union[Callable, str]] = None) -> pd.DataFrame:
    """
    Calculates tcpWER and tcorcWER for each session in hypothesis files using meeteval, and saves the error
    information to .json.
    Text normalization is applied to both hypothesis and reference.

    Args:
        out_dir: the directory to save the ref.json reference transcript to (extracted from gt_utt_df).
        tcp_wer_hyp_json: path to hypothesis .json file for tcpWER, or json structure.
        tcorc_wer_hyp_json: path to hypothesis .json file for tcorcWER, or json structure.
        gt_utt_df: dataframe of ground truth utterances. must include the sessions in the hypothesis files.
            see load_data() function.
        tn: text normalizer
        collar: tolerance of tcpWER to temporal misalignment between hypothesis and reference.
        save_visualizations: if True, save html visualizations of alignment between hyp and ref.
        meeting_id_is_session_id: if True, the session_id in the hypothesis/ref files is the same as the meeting_id.
    Returns:
        wer_df: pd.DataFrame with columns -
            'session_id' - same as in hypothesis files
            'tcp_wer': tcpWER
            'tcorc_wer': tcorcWER
            ... intermediate tcpWER/tcorcWER fields such as insertions/deletions. see in code.
    """
    # json to SegLST structure (Segment-wise Long-form Speech Transcription annotation)
    to_seglst = lambda x: meeteval.io.chime7.json_to_stm(x, None).to_seglst() if isinstance(x, list) \
        else meeteval.io.load(Path(x))
    tcp_hyp_seglst = to_seglst(tcp_wer_hyp_json)
    tcorc_hyp_seglst = to_seglst(tcorc_wer_hyp_json)

    # map session_id to meetind_id and join with gt_utt_df to include GT utterances for each session.
    # since every meeting contributes several sessions, a meeting's GT will be repeated for every session.
    ref_seglst = to_seglst(ref_json)

    if isinstance(tn, str):
        tn = get_txt_norm(tn)
    # normalization should be idempotent so a second normalization will not change the result
    tcp_hyp_seglst = tcp_hyp_seglst.map(partial(normalize_segment, tn=tn))
    tcorc_hyp_seglst = tcorc_hyp_seglst.map(partial(normalize_segment, tn=tn))
    ref_seglst = ref_seglst.map(partial(normalize_segment, tn=tn))

    ref_file_path = Path(out_dir) / 'ref.json'
    ref_file_path.parent.mkdir(parents=True, exist_ok=True)
    ref_seglst.dump(ref_file_path)

    tcorc_wer_res = calc_session_tcorc_wer(ref_seglst, tcorc_hyp_seglst, collar)

    cp_wer_res = calc_session_cp_wer(ref_seglst, tcp_hyp_seglst)
    tcp_wer_res = calc_session_tcp_wer(ref_seglst, tcp_hyp_seglst, collar)
    if save_visualizations:
        save_wer_visualization(ref_seglst, tcp_hyp_seglst, out_dir)

    wers_to_concat = [cp_wer_res,
                      tcp_wer_res.drop(columns='session_id'),
                      tcorc_wer_res.drop(columns='session_id')
                      ]
    if compute_orc:
        orc_wer_res = calc_session_approx_orc_wer(ref_seglst, tcp_hyp_seglst)
        wers_to_concat.append(orc_wer_res.drop(columns='session_id'))

    wer_df = pd.concat(wers_to_concat, axis=1)

    if isinstance(tcp_wer_hyp_json, str) or isinstance(tcp_wer_hyp_json, Path):
        wer_df['tcp_wer_hyp_json'] = tcp_wer_hyp_json
    if isinstance(tcorc_wer_hyp_json, str) or isinstance(tcorc_wer_hyp_json, Path):
        wer_df['tcorc_wer_hyp_json'] = tcorc_wer_hyp_json

    _LOG.debug('Done calculating WER')
    _LOG.debug(f"\n{wer_df[['session_id', 'cp_wer', 'tcorc_wer', 'tcp_wer']]}")

    return wer_df
