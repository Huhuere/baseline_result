# 一、数据预处理与元数据生成（DAIC-WOZ，46_18 / 20_20）

脚本完成：
- 读取转写与原始音频，按句段切分为片段并保存为 `pid_s{idx}_AUDIO.wav`
- 为每个片段生成对应的二分类标签 `.label` 与 PHQ 分数 `.phq_label`
- 生成三份片段级元数据 CSV（train/dev/test）
- 生成合并后的 `metadata_authority.csv`，包含列 `name,label,state`

## 快速开始

1) 修改路径
- 打开 `data_preprocessing_train_46_18.py`
- 将如下变量改为你的 DAIC-WOZ 数据根目录：
```
BASE_DATA_DIR = r'D:/path/to/DAIC-WOZ/data'
```
数据根目录需包含以下结构：
```
{BASE_DATA_DIR}/
  audio/wav_files/           # 原始音频：{PID}_AUDIO.wav
  Text_all/                  # 转写CSV：{PID}_TRANSCRIPT.csv
  label/
    train_split_Depression_AVEC2017.csv
    dev_split_Depression_AVEC2017.csv
    full_test_split.csv
```

2) 安装依赖
```
pip install pandas pydub
```
- 请安装 ffmpeg 并将其 bin 目录加入 PATH（pydub 需要）。

3) 运行
```
python .\speechformer_dataprocess\data_preprocessing_train_46_18.py
```

## 输出说明

- 片段文件目录（在 BASE_DATA_DIR 下生成）
  - `train_mode_46_18/`
  - `dev_mode_20_20/`
  - `test_mode_20_20/`
  其中音频命名为 `PID_s{序号}_AUDIO.wav`，并伴随 `*.label` 与 `*.phq_label`。

- 元数据 CSV（在脚本目录下生成）
  - `metadata_train_s_46_18.csv`（name,label）
  - `metadata_dev_s_20_20.csv`（name,label）
  - `metadata_test_s_20_20.csv`（name,label）
  - `metadata_authority.csv`（name,label,state）

CSV 均按数值顺序排序（先 PID，再片段序号）。

## 参数

- FIXED_N：dev/test 每个 subject 采样的 utterance 数，默认 20。
- MIN_DURATION：最短片段时长（秒），默认 1.0。
- 采样策略：
  - train 使用 `proportional`：`label==0 -> 18 条`，`label==1 -> 46 条`
  - dev/test 使用 `fixed`：每个 subject 固定 `FIXED_N` 条


# 二、特征提取

## 快速开始

1）安装依赖
```
cd SpeechFormer-master
python -m pip install -r requirements.txt
```
2）下载预训练模型wav2vec
* The pre-trained wav2vec model is publicly available at [here.](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec)
```
python ./extract_feature/extract_wav2vec.py
```
* 参数：meta_csv为前文数据切片元文件metadata_authority.csv，audio_dir为音频文件夹，将前文数据预处理中train_mode_46_18、dev_mode_20_20、test_mode_20_20中的.wav文件统一放入save_wav_files。
* 运行结果输出save_feature_files_train_46_18文件夹存放所有数据的特征.mat文件

# 三、训练
## Train model
Set the hyper-parameters on `./config/config.py` and `./config/model_config.json`.  
Note: the value of `expand` in `./config/model_config.json` for SpeechFormer-S is `[1, 1, 1, -1]`, while that of SpeechFormer-B is `[1, 1, 2, -1]`.  
Next, run:
```
python train_model.py
```
You can also pass the hyper-parameters from the command line for convenience, more details can be found in `train_model.py`.
