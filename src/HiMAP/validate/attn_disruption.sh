python ./src/HiMAP/validate/attn_disruption.py \
    --model-path /code/FasterV/models/llava-v1.5-13b \
    --question-file ./data/scienceqa/himap-inference-MCQ-VC-13B.json \
    --image-folder ./data/scienceqa/images/test \
    --single-pred-prompt \
    --isolate-modality img2txt \
    --isolate-layer 36