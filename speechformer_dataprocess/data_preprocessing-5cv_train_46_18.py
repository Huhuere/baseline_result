import os
import pandas as pd
from pydub import AudioSegment

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = SCRIPT_DIR  # 元数据 CSV 输出目录（默认为脚本同目录）

# 修改此处为 DAIC-WOZ 数据根目录
BASE_DATA_DIR = r'D:/depression detection/1/speechformer_dataprocess/OneDrive_1_2025-8-18/DAIC-WOZ/data'


AUDIO_DIR      = os.path.join(BASE_DATA_DIR, 'audio', 'wav_files')
TRANS_DIR      = os.path.join(BASE_DATA_DIR, 'Text_all')

OUT_TRAIN_MODE = os.path.join(BASE_DATA_DIR, 'train_mode_46_18')
OUT_DEV_MODE   = os.path.join(BASE_DATA_DIR, 'dev_mode_20_20')
OUT_TEST_MODE  = os.path.join(BASE_DATA_DIR, 'test_mode_20_20')

CSV_TRAIN = os.path.join(BASE_DATA_DIR, 'label', 'train_split_Depression_AVEC2017.csv')
CSV_DEV   = os.path.join(BASE_DATA_DIR, 'label', 'dev_split_Depression_AVEC2017.csv')
CSV_TEST  = os.path.join(BASE_DATA_DIR, 'label', 'full_test_split.csv')
FIXED_N   = 20          # dev/test：每个 subject 取多少条最长 utterance
MIN_DURATION = 1.0      # 片段最短持续时间（秒）
# ----------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# 读取CSV文件只保留 id, PHQ8_Binary, PHQ8_Score
def load_all_labels(csv_paths):
    """
    把多个 split 的 CSV 合并成一个，并去重 Participant_ID；仅保留
    Participant_ID, PHQ8_Binary, PHQ8_Score 三列
    """
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        dfs.append(df[['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score']])
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.drop_duplicates(subset='Participant_ID')
    return all_df

# 筛选出指定说话人的所有语句，计算每条语句的持续时间，并按持续时间降序排序
def pick_utterances(transcript_csv, speaker='Participant'):
    """
    返回 DataFrame，其中只包含指定说话人（默认 'Participant'）的 utterance，
    并计算 duration = stop_time - start_time，按 duration 降序排序。
    需要 transcript CSV 至少包含列：speaker, start_time, stop_time
    """
    df = pd.read_csv(transcript_csv)
    df = df[df['speaker'] == speaker].copy()
    if df.empty:
        return df
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['stop_time']  = pd.to_numeric(df['stop_time'], errors='coerce')
    df = df.dropna(subset=['start_time', 'stop_time'])
    df['duration'] = df['stop_time'] - df['start_time']
    return df.sort_values('duration', ascending=False)

def process_subject(pid, label, phq_score, out_dir,
                    sample_mode='proportional', fixed_n=20, min_duration=MIN_DURATION):
    """
    对单个 subject 进行处理：
      - 从 TRANS_DIR 中加载该 subject 的转录 CSV；
      - 计算持续时间，过滤掉持续时间小于 min_duration 的 utterance；
      - 根据采样模式选择前 n 条 utterance（已经按持续时间降序）；
      - 用 pydub 截取对应音频片段，保存 wav 和标签文件到 out_dir。
    返回保存的 clip 数。
    """
    audio_path = os.path.join(AUDIO_DIR, f'{pid}_AUDIO.wav')
    trans_path = os.path.join(TRANS_DIR, f'{pid}_TRANSCRIPT.csv')

    if not os.path.exists(audio_path) or not os.path.exists(trans_path):
        print(f'[Skip] audio / transcript missing for {pid}')
        return 0

    utt_df = pick_utterances(trans_path)
    if utt_df.empty:
        print(f'[Skip] no participant utterances in {pid}')
        return 0

    # 过滤掉持续时间小于 min_duration 的 utterance
    utt_df = utt_df[utt_df['duration'] >= min_duration]
    if utt_df.empty:
        print(f'[Skip] no utterances longer than {min_duration}s for {pid}')
        return 0

    # 选择采样数量（按二分类标签比例或固定数）
    if sample_mode == 'proportional':
        n = 18 if label == 0 else 46
    else:
        n = fixed_n
    utt_df = utt_df.head(n)

    ensure_dir(out_dir)
    audio = AudioSegment.from_wav(audio_path)

    cnt = 0
    for i, row in utt_df.iterrows():
        st_ms = int(float(row['start_time']) * 1000)
        ed_ms = int(float(row['stop_time'])  * 1000)
        seg = audio[st_ms:ed_ms]

        # 跳过小于 1 秒的片段
        if len(seg) < int(min_duration * 1000):
            print(f"Skipping {pid} clip {i} because segment length {len(seg)} ms < {int(min_duration*1000)} ms")
            continue

        cnt += 1

        # 目标命名：pid_s{cnt}_AUDIO.*
        base = f'{pid}_s{cnt}_AUDIO'
        wav_out       = os.path.join(out_dir, f'{base}.wav')
        label_out     = os.path.join(out_dir, f'{base}.label')       # 二分类标签
        phq_label_out = os.path.join(out_dir, f'{base}.phq_label')   # PHQ 分数

        seg.export(wav_out, format='wav')
        with open(label_out, 'w', encoding='utf-8') as f:
            f.write(str(label))
        with open(phq_label_out, 'w', encoding='utf-8') as f:
            f.write(str(phq_score))

    return cnt

def run(mode_out_dir, sample_mode, fixed_n=None, csv_paths=[]):
    ensure_dir(mode_out_dir)
    df = load_all_labels(csv_paths)

    total = 0
    for idx, row in df.iterrows():
        if pd.isna(row['Participant_ID']):
            print(f"[Error] NaN Participant_ID at row {idx}: {row}")
            continue
        try:
            pid = int(row['Participant_ID'])
            label = int(row['PHQ8_Binary'])   # 二分类标签
            phq_score = int(row['PHQ8_Score'])  # PHQ 分数
        except Exception as e:
            print(f"[Error] parse row {idx} failed: {row}, err: {e}")
            continue

        n_ok = process_subject(
            pid, label, phq_score, mode_out_dir,
            sample_mode=sample_mode, fixed_n=fixed_n if fixed_n is not None else FIXED_N,
            min_duration=MIN_DURATION
        )
        total += n_ok
        print(f'[{sample_mode}] {pid} (Binary={label}, PHQ={phq_score}): {n_ok} clips')

    print(f'\nDone!  Total clips ({sample_mode}) = {total}\n')

# 生成片段级 metadata（name,label），并按 pid 与 s 序号排序
def _key_for_sorted_name(name: str):
    stem, _ = os.path.splitext(name)
    parts = stem.split('_') 
    if len(parts) == 3 and parts[0].isdigit() and parts[1].startswith('s') and parts[1][1:].isdigit():
        return (int(parts[0]), int(parts[1][1:]), '')
    return (1_000_000_000, 0, name)

def write_clip_metadata(processed_dir: str, out_csv: str):
    import csv as _csv
    files = [f for f in os.listdir(processed_dir) if f.lower().endswith('.wav')]
    files.sort(key=_key_for_sorted_name)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['name', 'label'])
        for wav in files:
            base = os.path.splitext(wav)[0]
            label_path = os.path.join(processed_dir, base + '.label')
            label = ''
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as lf:
                    label = lf.read().strip()
            w.writerow([wav, label])

def build_metadata_with_state(processed_dir: str, state: str) -> pd.DataFrame:
    files = [f for f in os.listdir(processed_dir) if f.lower().endswith('.wav')]
    files.sort(key=_key_for_sorted_name)
    rows = []
    for wav in files:
        base = os.path.splitext(wav)[0]
        label_path = os.path.join(processed_dir, base + '.label')
        label = ''
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as lf:
                label = lf.read().strip()
        rows.append({'name': wav, 'label': label, 'state': state})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    # 1) 生成切分片段与标签
   if __name__ == '__main__':
    # 1) 生成切分片段与标签
    run(OUT_TRAIN_MODE, sample_mode='proportional', csv_paths=[CSV_TRAIN])
    run(OUT_DEV_MODE,   sample_mode='fixed',        csv_paths=[CSV_DEV])
    run(OUT_TEST_MODE,  sample_mode='fixed',        csv_paths=[CSV_TEST])

    # 2) 生成片段级 metadata（name,label）
    write_clip_metadata(OUT_TRAIN_MODE, os.path.join(OUTPUT_ROOT, 'metadata_train_s_46_18.csv'))
    write_clip_metadata(OUT_DEV_MODE,   os.path.join(OUTPUT_ROOT, 'metadata_dev_s_20_20.csv'))
    write_clip_metadata(OUT_TEST_MODE,  os.path.join(OUTPUT_ROOT, 'metadata_test_s_20_20.csv'))

    # 3) 生成完整的 metadata（name,label,state）
    meta_all = pd.concat([
        build_metadata_with_state(OUT_TRAIN_MODE, 'train'),
        build_metadata_with_state(OUT_DEV_MODE,   'dev'),
        build_metadata_with_state(OUT_TEST_MODE,  'test'),
    ], ignore_index=True)
    meta_all.to_csv(os.path.join(OUTPUT_ROOT, 'metadata_authority.csv'), index=False, encoding='utf-8')

    print('\n数据处理完毕（已生成 metadata_authority.csv）')