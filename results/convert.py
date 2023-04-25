import pandas as pd
import argparse
import csv

def main(args):
    df = pd.read_csv(args.s)
    header = ['algo', "selected by URE", "URE", "selected by SCEL", "SCEL", "selected by val_acc", "valid_acc", "best_ure", "best_scel", "best_val_acc"]

    dataset_name = args.dataset
    if dataset_name == "uniform-cifar10" or dataset_name == "uniform-cifar20":
        algo_list = ["fwd-u", "ure-ga-u", "scl-nl", "scl-exp", "l-w", "l-uw"]
    else:
        algo_list = ["fwd-u", "fwd-r", "ure-ga-u", "ure-ga-r", "scl-nl", "scl-exp", "l-w", "l-uw"]

    with open(f"{args.dataset}.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for algo in algo_list:
            table_row = []
            table_row.append(algo)
            # selected by URE
            row = df.loc[(df['algo'] == algo) & (df['dataset_name'] == dataset_name)].sort_values('ure').head(1) # ascending
            table_row.append(row['test_acc'].values[0]*100)
            table_row.append(row['ure'].values[0])
            # selected by SCEL
            row = df.loc[(df['algo'] == algo) & (df['dataset_name'] == dataset_name)].sort_values('scel').head(1) # ascending
            table_row.append(row['test_acc'].values[0]*100)
            table_row.append(row['scel'].values[0])
            # selected by valid_acc
            row = df.loc[(df['algo'] == algo) & (df['dataset_name'] == dataset_name)].sort_values('valid_acc', ascending=False).head(1) # ascending
            table_row.append(row['test_acc'].values[0]*100)
            table_row.append(row['scel'].values[0])

            # Early-stopping
            row = df.loc[(df['algo'] == algo) & (df['dataset_name'] == dataset_name)].sort_values('best_epoch-ure.ure').head(1) # ascending
            table_row.append(row['best_epoch-ure.test_acc'].values[0]*100)
            row = df.loc[(df['algo'] == algo) & (df['dataset_name'] == dataset_name)].sort_values('best_epoch-scel.scel').head(1) # ascending
            table_row.append(row['best_epoch-scel.test_acc'].values[0]*100)
            row = df.loc[(df['algo'] == algo) & (df['dataset_name'] == dataset_name)].sort_values('best_epoch-valid_acc.valid_acc', ascending=False).head(1) # ascending
            table_row.append(row['best_epoch-valid_acc.test_acc'].values[0]*100)
            writer.writerow(table_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args)