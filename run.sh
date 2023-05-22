python train.py \
    --algo=fwd-r \
    --dataset_name=noisy-uniform-cifar20 \
    --model=resnet18 \
    --lr=1e-4 \
    --seed=0 \
    --data_aug=false \
    --valid_ratio 0.01 \
    --eta 0.03 \
    --test