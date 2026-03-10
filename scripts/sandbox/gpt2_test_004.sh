#!/bin/bash

#source .venv/bin/activate

PROJECT="exp_004"
RUNS=3
LR=(0.00005 
0.0005 0.003)
BATCH_SIZE=128 # P100 GPU: Set to 64
EPOCHS=250
R=(16)
FINE_TUNING=("freeze") # "lora")  #Lora not possible with GPT2 due to model architecture
declare -a FREEZE_LAYERS=(
    "-1 -2"
    #"0 1"
    #"0"
    #"-1"
    #null
)

#declare -a TIME_POSITIONAL_ENCODING=(None) # "additive")
declare -a LIFECYCLE_FLAGS=("--lifecycle")
# declare -a DATASETS=(BPI12 BPI15 BPI17 BPI19 BPI20PrepaidTravelCosts BPI20TravelPermitData BPI20RequestForPayment)
declare -a DATASETS=(BPI15)
declare -a continuous_features CONTINUOUS_FEATURES=(all)

# python fetch_wandb.py --project $PROJECT

for dataset in "${DATASETS[@]}"
do  
    for ((run=0; run<RUNS; run++))
    do
        #for time_pos_enc in "${TIME_POSITIONAL_ENCODING[@]}"
        for lifecycle_flag in "${LIFECYCLE_FLAGS[@]}"
        do
            for lr in "${LR[@]}"
            do
                for fine_tuning in "${FINE_TUNING[@]}"
                do
                    for continuous_features in "${CONTINUOUS_FEATURES[@]}"
                    do
                        cmd="--dataset $dataset \
                            --backbone gpt2-large \
                            --categorical_features activity \
                            --categorical_targets activity \
                            --strategy sum \
                            --lr $lr \
                            --batch_size $BATCH_SIZE \
                            --epochs $EPOCHS \
                            --fine_tuning $fine_tuning \
                            --project_name $PROJECT \
                            --val_size .1 \
                            --val_split prefix \
                            --patience 10 \
                            --min_delta .005 \
                            --compile \
                            $lifecycle_flag
                            "
                            ## GPU compatibility
                            # P100/V100: --precision fp16, remove --compile
                            # A100: --precision bf16, --compile

                            #####################

                            # --verbose i
                            # --precision fp32 
                            # --categorical_targets activity \
                            # --strategy concat: not possible if continues features is None 
                            #--continuous_features $continuous_features \


                        #Only add these flags when a value is requested (None is default)
                        #if [[ "$time_pos_enc" != "None" && -n "$time_pos_enc" ]]; then
                        #cmd="$cmd --time_positional_encoding $time_pos_enc"
                        #fi
                        if [[ "$continuous_features" != "None" && -n "$continuous_features" ]]; then
                        cmd="$cmd --continuous_features $continuous_features"
                        fi

                        if [[ $fine_tuning == "lora" ]]; then
                            for r in "${R[@]}"
                            do
                                cmd2="$cmd \
                                --r $r \
                                --lora_alpha $(( r*2 ))"
                                python main_nep.py $cmd2 --wandb
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
done



