# cll-experiment
The code repo of "CLCIFAR: CIFAR-Derived Benchmark Datasets with Human Annotated Complementary Labels"(https://arxiv.org/abs/2305.08295)

To reproduce the results for FWD algorithm with true empirical transition matrix, please run
```python
python train.py \
    --algo=fwd-r \
    --dataset_name=clcifar10 \
    --model=m-resnet18 \
    --lr=5e-4 \
    --seed=197 \
    --data_aug=true \
    --test
```

or directly run
```bash
bash run.sh
```
