#bin/bash

#source .venv/bin/activate

PROJECT="finetune_gpt2_persist"
LR=(0.00005)
BATCH_SIZE=32
EPOCHS=10
R=(16)
FINE_TUNING=("freeze") # "lora")  #Lora not possible with GPT2 due to model architecture
declare -a FREEZE_LAYERS=(
   # "-1 -2"
   # "0 1"
   # "0"
   # "-1"
    null
)

declare -a TIME_POSITIONAL_ENCODING=(None)
# declare -a DATASETS=(BPI12 BPI17 BPI20PrepaidTravelCosts BPI20TravelPermitData BPI20RequestForPayment)
declare -a DATASETS=(BPI12)
declare -a continuous_features CONTINUOUS_FEATURES=(all)

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
                    # 896 for qwen25-05b, 768 for gpt2, distilgpt2, 2 for gpt2-tiny, 
                    #1024 for gpt2-medium, 2048 for llama32-1b
                    cmd="--dataset $dataset \
                        --backbone gpt2 \
                        
                        --categorical_features activity \
                        --categorical_targets activity \
                        --strategy sum \
                        --lr $lr \
                        --batch_size $BATCH_SIZE \
                        --epochs $EPOCHS \
                        --fine_tuning $fine_tuning \
                        --project_name $PROJECT \
                        --persist_model"
                        # --continuous_targets remaining_time \
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
                            python main.py $cmd2 --wandb
                        done
                    fi
                done
            done
        done
    done
done



