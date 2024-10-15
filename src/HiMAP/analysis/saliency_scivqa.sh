export CUDA_VISIBLE_DEVICES=0
python ./src/HiMAP/analysis/saliency_scivqa.py \
    --model-path /code/FasterV/models/llava-v1.5-7b \
    --question-file ./data/scienceqa/himap-inference-MCQ.json \
    --result-file ./output_example/scivqa_props-7b.pt \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --conv-mode vicuna_v1