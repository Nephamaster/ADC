datasets=(
    SIGHAN13
    SIGHAN14
    SIGHAN15
    CSCD-NS
    LEMON_CAR
    LEMON_COT
    LEMON_ENC
    LEMON_GAM
    LEMON_MEC
    LEMON_NEW
    LEMON_NOV
)

for dataset in "${datasets[@]}"
do
    echo "Evaluating $dataset..."

    python evaluate.py \
        --gold data/${dataset}.txt \
        --hypo predictions/${dataset}.txt \
        --metric_algorithm levenshtein \

    echo "Done $dataset"
    echo "----------------------"
done