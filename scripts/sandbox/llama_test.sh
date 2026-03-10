#bin/bash

#source .venv/bin/activate

PROJECT="test_script_04"
LR=(0.00005)
BATCH_SIZE=16
EPOCHS=20
R=(16)
FINE_TUNING=("lora") # "freeze")
declare -a FREEZE_LAYERS=(
    "-1 -2"
    "0 1"
    "0"
    "-1"
    null
)

declare -a TIME_POSITIONAL_ENCODING=(None "additive")
# declare -a DATASETS=(BPI12 BPI17 BPI20PrepaidTravelCosts BPI20TravelPermitData BPI20RequestForPayment)
declare -a DATASETS=(BPI12)
declare -a continuous_features CONTINUOUS_FEATURES=(None all)

# python fetch_wandb.py --project $PROJECT

for dataset in "${DATASETS[@]}"
do  
    for time_pos_enc in "${TIME_POSITIONAL_ENCODING[@]}"
    do
        for lr in "${LR[@]}"
        do
            for fine_tuning in "${FINE_TUNING[@]}"
            do
                for continuous_features in "${CONTINUOUS_FEATURES[@]}"
                do
                    # hidden size and embedding size must be the same as the backbone model:
                    # 896 for qwen25-05b, 768 for gpt2, 2048 for llama32-1b
                    cmd="--dataset $dataset \
                        --backbone llama32-1b \
                        --categorical_features activity \
                        --continuous_targets remaining_time \
                        --strategy sum \
                        --lr $lr \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --fine_tuning $fine_tuning \
                        --project_name $PROJECT"
                        # --categorical_targets activity \
                        # --strategy concat: not possible if continues features is None
                        #--continuous_features $continuous_features \

                    # Only add these flags when a value is requested (None is default)
                    if [[ "$time_pos_enc" != "None" && -n "$time_pos_enc" ]]; then
                    cmd="$cmd --time_positional_encoding $time_pos_enc"
                    fi
                    if [[ "$continuous_features" != "None" && -n "$continuous_features" ]]; then
                    cmd="$cmd --continuous_features $continuous_features"
                    fi

                    if [[ $fine_tuning == "lora" ]]; then
                        for r in "${R[@]}"
                        do
                            cmd2="$cmd \
                            --r $r \
                            --lora_alpha $(( r*2 ))"
                            python main.py $cmd2 --wandb
                            echo $cmd2 --wandb
                        done
                    else
                        for freeze_layers in "${FREEZE_LAYERS[@]}"
                        do
                            if [ "$freeze_layers" == "null" ]; then
                                cmd2=$cmd
                            else
                                cmd2="$cmd --freeze_layers $freeze_layers"
                            fi
                            echo $cmd2 --wandb
                            python main_nep.py $cmd2 --wandb
                        done
                    fi
                done
            done
        done
    done
done



