export CUDA_VISIBLE_DEVICES=0
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path /code/FasterV/models/llava-v1.5-13b \
    --question-file ./data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --use-hmap-v \
    --sys-length 35 \
    --img-length 576 \
    --hmap-v-attn-txt-layer 2 \
    --hmap-v-attn-img-layer 8 \
    --hmap-v-attn-txt-rank 288 \
    --hmap-v-attn-img-rank 72

export CUDA_VISIBLE_DEVICES=0
python ./src/HiMAP/inference/eval_scivqa.py \
    --model-path /code/FasterV/models/llava-v1.5-13b \
    --question-file ./data/scienceqa/himap-inference-MCQ.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt