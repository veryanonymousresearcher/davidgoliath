# 75 lines
# --lr 0.00005 --backbone ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b")
# run 1
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.00005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.00005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb   

# run 2
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.00005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.00005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb 
# run 3
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.00005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.00005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile 

# run 4
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.00005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.00005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 5
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.00005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.00005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.00005  --batch_size 32 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# --lr 0.0005 --backbone ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b")
# run 1
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.0005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.0005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 2
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.0005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.0005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 3
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.0005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.0005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 4
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.0005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.0005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 5
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.0005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.0005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.0005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  


# --lr 0.005 --backbone ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b")
# run 1
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 2
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 3
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.005  --batch_size 64 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 4
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.005  --batch_size 32 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  

# run 5
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-0.6b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-1.7b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-4b --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
#python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-8b --lr 0.005  --batch_size 32 --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb
python main_nep.py --project_name BPI20_TPD_spark_001 --dataset BPI20TravelPermitData --backbone qwen3-14b --lr 0.005  --batch_size 32 --patience 25 --val_size .1 --val_split prefix  --lifecycle --wandb --compile  