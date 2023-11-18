#!/bin/bash

export CUDA_VISIBLE_DEIVCES=0

python -m llava.eval.model_vqa \
    --model-path /zecheng/svg_model_hub/custom_llava_codellama/checkpoint-2200 \
    --question-file /zecheng/svg/icon-shop/meta_data/svg_to_image/valid_llava_image_to_svg.json \
    --image-folder /zecheng/svg/icon-shop/meta_data/svg_to_image/rendered_eval_image_c \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_beams 1 \
    --top_p 0.9 \
    --temperature 0.7 \
    --answers-file /zecheng/svg/icon-shop/meta_data/svg_to_image/gen_res/llava_codellama.jsonl;

