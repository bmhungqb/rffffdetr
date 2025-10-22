export model=n
# CUDA_VISIBLE_DEVICES=0 train.py --config_file configs/detrpose/detrpose_hgnetv2_${model}.py --device cuda --amp --pretrain dfine_${model}_obj365 
python train.py --config_file configs/detrpose/detrpose_hgnetv2_n.py --device cpu --amp --pretrain dfine_n_obj365 