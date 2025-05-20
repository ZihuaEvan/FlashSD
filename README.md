# Eagle eye

## Install

pip install -e .
pip install -r requiements.txt

## Data preparation
for instruction tuning:
```python
cd ge_data/
python get_data_all_llava.py -outdir [path of data]

python get_data_all_qwen2.5vl.py -outdir [path of data]
```

for video captioning:
```python
cd ge_data/
python get_data_video_all_llava.py -outdir [path of data]

python get_data_video_all_qwen2.5vl.py -outdir [path of data]
```


### Training
for multimodal speculative decoding:
```
cd train/
python train_llava.py --tmpdir [path of data]\
--cpdir [path of checkpoints] -- configpath [path of config file]
```

for flash:
```
cd train/
python train_sar.py --tmpdir [path of data]\
--cpdir [path of checkpoints] -- configpath [path of config file]
```

## Evaluation
```
cd evaluation/
python gen_ee_answer_llava.py  --ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\

python gen_ee_answer_qwen2.5vl.py  --ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\
```

Calculation of speed-up ratioL:

```
python  gen_baseline_answer_llava.py -ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\


python  gen_baseline_answer_qwen2.5vl.py -ee-model-path [path of EAGLE-EYE weight]\ --base-model-path [path of the original model]\

python speed.py [path of the base_model json] [path of the your_model json]
```