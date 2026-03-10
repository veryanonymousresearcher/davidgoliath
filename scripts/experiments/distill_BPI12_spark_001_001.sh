# 5 lines to get best BPI12 results to be used for distillation


#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 768 --n_layers 12 --n_heads 12 --lr 0.005  --val_size .1 --val_split prefix --patience 3 --lifecycle --wandb 
#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 512 --n_layers 12 --n_heads 8 --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb 
#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 512 --n_layers 6 --n_heads 8 --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb 
#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 256 --n_layers 6 --n_heads 4 --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb 
#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 256 --n_layers 4 --n_heads 4 --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb 
#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 128 --n_layers 4 --n_heads 2 --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb 
#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 128 --n_layers 2 --n_heads 2 --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb 
#python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_qwen3-0.6b_run74onkpmz.pth --hidden_size 64 --n_layers 2 --n_heads 1 --lr 0.005  --val_size .1 --val_split prefix --patience 10 --lifecycle --wandb 
python main_distill.py --project_name Distill_BPI12_spark_001_001_test --dataset BPI12 --t_model_name BPI12_gpt2-mini_run_rp4iwpdn.pth --hidden_size 64 --n_layers 1 --n_heads 1 --lr 0.005  --val_size .1 --val_split prefix --patience 2 --lifecycle --wandb 

