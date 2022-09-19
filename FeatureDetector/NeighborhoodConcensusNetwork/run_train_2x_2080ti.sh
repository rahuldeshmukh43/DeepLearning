#!/bin/bash
#python3 -m pdb train.py --config train_pfpascal.yaml --batch_size 6 --log_img_interval 1
#
# batch size of 14 with 2x gpu and learnable backbone
# batch size of 28 with 2x gpu and fixed backbone
#


# gradual finetuning resenet bottleneck to initial layer

config=train_pfpascal_2x_2080ti.yaml

python3 train.py --config $config --batch_size 28 --log_scalar_interval 5 --log_img_interval 10  --fix_backbone --num_epochs 5 --save_interval 25 --lr 5e-4 --log_name fixed_backbone  --multi_gpu 
python3 train.py --config $config --batch_size 16 --log_scalar_interval 5 --log_img_interval 10  --train_backbone_layer layer3 --num_epochs 2 --save_interval 25 --log_name train_layer3 --multi_gpu  
python3 train.py --config $config --batch_size 14 --log_scalar_interval 5 --log_img_interval 10  --train_backbone_layer layer2 --num_epochs 2 --save_interval 25 --log_name train_layer2  --multi_gpu 
python3 train.py --config $config --batch_size 14 --log_scalar_interval 5 --log_img_interval 10  --train_backbone_layer layer1 --num_epochs 2 --save_interval 25 --log_name train_layer1  --multi_gpu 
python3 train.py --config $config --batch_size 14 --log_scalar_interval 5 --log_img_interval 10  --num_epochs 2 --save_interval 25 --log_name train_all  --multi_gpu 




