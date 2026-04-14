CUDA_VISIBLE_DEVICES=5 python inference.py \
    --dataset evaluation/data/LEMON.json \
    --cache ./cache/ \
    --model ./outputs/Qwen3-8B-Char-Adapter-twnlp/checkpoint-11935 \
    --csc \
    # --gpu_memory_utilization 0.9 \
    # --model /share/project/wuhaiming/data/models/Qwen3-8B \
    # --output predictions/ADC_char_layer28.jsonl \