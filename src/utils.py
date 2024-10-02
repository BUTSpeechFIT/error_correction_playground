import decimal

import meeteval
import numpy as np
from meeteval.io.seglst import SegLstSegment
from meeteval.wer.wer.orc import OrcErrorRate


def assign_streams(tcorc_hyp_seglst):
    # def assign_streams(df):
    #     df = df.sort_values(by=['start_time']).reset_index(drop=True)
    #     df['map'] = 0  # Initialize all as non-overlapping
    #     current_stream = 0
    #
    #     # Iterate through all intervals
    #     for i in range(1, len(df)):
    #         prev_row = df.iloc[i - 1]
    #         curr_row = df.iloc[i]
    #
    #         # Check if current interval overlaps with the previous interval, regardless of speaker
    #         if curr_row['start_time'] < prev_row['end_time']:
    #
    #             if curr_row['map'] == 0:  # If it's not already assigned to a stream
    #                 current_stream += 1  # Assign a new stream number
    #                 if current_stream > 1:
    #                     for j in reversed(range(1, current_stream + 1)):
    #                         overlapping_row = df.iloc[i - j]
    #                         if curr_row['start_time'] > overlapping_row['end_time']:
    #                             current_stream = overlapping_row['map']
    #                 df.at[i, 'map'] = current_stream
    #         else:
    #             current_stream = 0  # Reset stream if no overlap
    #     df['speaker_id'] = df['map']
    #     del df['map']
    #
    #     return df
    tcorc_hyp_seglst = tcorc_hyp_seglst.groupby(key='speaker')
    per_stream_list = [[] for _ in range(len(tcorc_hyp_seglst))]
    for speaker_id, speaker_seglst in tcorc_hyp_seglst.items():
        speaker_seglst = speaker_seglst.sorted(key='start_time')
        for seg in speaker_seglst:
            # check if current segment does not overlap with any of the segments in per_stream_list
            for i in range(len(per_stream_list)):
                if not any(seg['start_time'] < s['end_time'] and seg['end_time'] > s['start_time'] for s in per_stream_list[i]):
                    seg['speaker'] = i
                    per_stream_list[i].append(seg)
                    break
            else:
                raise ValueError('No stream found for segment')
    tcorc_hyp_seglst = meeteval.io.SegLST([seg for stream in per_stream_list for seg in stream]).sorted(key='start_time')
    return tcorc_hyp_seglst

def normalize_segment(segment: SegLstSegment, tn):
    words = segment["words"]
    words = tn(words)
    segment["words"] = words
    return segment


def df_to_seglst(df):
    return meeteval.io.SegLST([
        SegLstSegment(
            session_id=row.session_id,
            start_time=decimal.Decimal(row.start_time),
            end_time=decimal.Decimal(row.end_time),
            words=row.text,
            speaker=row.speaker_id,
        )
        for row in df.itertuples()
    ])


def create_vad_mask(segments, time_step=0.1, total_duration=None):
    """
    Create a VAD mask for the given segments.

    :param segments: List of segments, each containing 'start_time' and 'end_time'
    :param time_step: The resolution of the VAD mask in seconds (default: 100ms)
    :param total_duration: Optionally specify the total duration to create the mask.
                           If not provided, the mask will be generated based on the maximum end time of the segments.
    :return: VAD mask as a numpy array, where 1 represents voice activity and 0 represents silence.
    """
    # Find the total duration if not provided
    if total_duration is None:
        total_duration = max(seg['end_time'] for seg in segments)

    # Initialize VAD mask as zeros (silence)
    mask_length = int(float(total_duration) / time_step) + 1
    vad_mask = np.zeros(mask_length, dtype=bool)

    # Iterate over segments and mark the corresponding times as active (1)
    for seg in segments:
        start_idx = int(float(seg['start_time']) / time_step)
        end_idx = int(float(seg['end_time']) / time_step)
        vad_mask[start_idx:end_idx] = 1

    return vad_mask


def find_group_splits(vad, group_duration=30, time_step=0.1):
    non_active_indices = np.argwhere(~vad).squeeze()
    splits = []
    group_shift = group_duration / time_step
    next_offset = group_shift
    for i in non_active_indices:
        if i >= next_offset:
            splits.append(i)
            next_offset = i + group_shift
    return splits


def map_utterance_to_split(utterance_start_time, splits):
    for i, split in enumerate(splits):
        if utterance_start_time < split:
            return i
    return len(splits)


def agregate_errors_across_groups(res, session_id):
    overall_error_number = sum([group.errors for group in res.values()])
    overall_length = sum([group.length for group in res.values()])
    overall_errors = {
        'error_rate': overall_error_number / overall_length,
        'errors': overall_error_number,
        'length': overall_length,
        'insertions': sum([group.insertions for group in res.values()]),
        'deletions': sum([group.deletions for group in res.values()]),
        'substitutions': sum([group.substitutions for group in res.values()]),
        'assignment': []
    }
    for group in res.values():
        overall_errors['assignment'].extend(list(group.assignment))
    overall_errors['assignment'] = tuple(overall_errors['assignment'])
    res = {session_id: OrcErrorRate.from_dict(overall_errors)}
    return res
