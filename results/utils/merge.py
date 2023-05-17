import pandas as pd

sub_csv = ['clcifar10-aggregate.csv', 'clcifar10-iid.csv', 'clcifar10-noiseless.csv', 'uniform-cifar10.csv',
           'clcifar20-aggregate.csv', 'clcifar20-iid.csv', 'clcifar20-noiseless.csv', 'clcifar20.csv', 'uniform-cifar20.csv']

df1 = pd.read_csv('clcifar10.csv')

for filename in sub_csv:
    df = pd.read_csv(filename)
    df = df.reindex(columns=df1.columns)

    df1 = pd.concat([df1, df], ignore_index=True)

df1.to_csv('wandb_result-0.csv', index=False)
