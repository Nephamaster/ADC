CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name /share/project/wuhaiming/data/models/Qwen3-8B-Base \
    --data_file /share/project/wuhaiming/spaces/LlamaFactory/data/twnlp_csc.jsonl \
    --modal_cache_dir /share/project/wuhaiming/spaces/ADC/cache/ \
    --output_dir Qwen3-8B-Adapter