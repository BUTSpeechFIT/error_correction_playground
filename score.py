import argparse
import os
from pathlib import Path

import pandas as pd
import tqdm

from src.logging_def import get_logger
from src.wer import calc_wer

_LOG = get_logger('wer')


def main(predictions_dir: str, save_visualizations: bool, compute_orc: bool, collar: int = 5,
         text_norm: str = 'default'):
    wer_dfs = []
    for i, session_id in enumerate(tqdm.tqdm(os.listdir(predictions_dir), desc='Scoring sessions')):
        if i > 5:
            break
        if not os.path.isdir(Path(predictions_dir) / session_id):
            continue
        calc_wer_out = Path(predictions_dir) / session_id
        out_tcp_file = Path(predictions_dir) / session_id / 'tcp_wer_hyp.json'
        out_tc_file = Path(predictions_dir) / session_id / 'tc_orc_wer_hyp.json'
        ref_file = Path(predictions_dir) / session_id / 'ref.json'
        session_wer: pd.DataFrame = calc_wer(
            calc_wer_out,
            out_tcp_file,
            out_tc_file,
            ref_file,
            collar=collar,
            save_visualizations=save_visualizations,
            compute_orc=compute_orc,
            tn=text_norm)
        wer_dfs.append(session_wer)

    all_session_wer_df = pd.concat(wer_dfs, ignore_index=True)
    all_session_wer_df.to_csv(predictions_dir + '/all_session_wer.csv')
    metrics = {key: value for key, value in all_session_wer_df._get_numeric_data().mean().items()}
    _LOG.info(f"Metrics: {metrics}")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', type=str, required=True)
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--compute_orc', action='store_true')
    parser.add_argument('--collar', type=int, default=5)
    parser.add_argument('--text_norm', type=str, default='chime8')
    return parser


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    main(**vars(args))
