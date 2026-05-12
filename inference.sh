CUDA_VISIBLE_DEVICES=2 python inference.py \
    --dataset data/rSIGHAN.json \
    --model ./outputs/Qwen3.5-9B-Base-Char-Adapter-SFT-CSCMIX/test-checkpoint/ \
    --output ./predictions/qwen35-9B-base-char-adapter-sft-34mix \
    --csc \
    --cache ./cache/ \
    # --csc \
    # --cache ./cache/ \
    # --gpu_memory_utilization 0.9 \
    # --model ./outputs/Qwen3-8B-Char-Adapter-twnlp/checkpoint-11935 \
    # --model /share/project/wuhaiming/spaces/LlamaFactory/Nepham/saves/Qwen3/SFT/Adapter/Qwen3-8B-Base-SFT-twnlp/ \