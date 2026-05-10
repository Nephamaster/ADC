CUDA_VISIBLE_DEVICES=4 python inference.py \
    --dataset evaluation/data/SIGHAN.json \
    --model ./outputs/Qwen3.5-9B-Adapter-CSCMIX/checkpoint-3035/ \
    --output ./predictions/qwen35-9B-base-sft-twnlp \
    --csc \
    --cache ./cache/ \
    # --csc \
    # --cache ./cache/ \
    # --gpu_memory_utilization 0.9 \
    # --model ./outputs/Qwen3-8B-Char-Adapter-twnlp/checkpoint-11935 \
    # --model /share/project/wuhaiming/spaces/LlamaFactory/Nepham/saves/Qwen3/SFT/Adapter/Qwen3-8B-Base-SFT-twnlp/ \