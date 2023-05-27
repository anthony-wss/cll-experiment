import pandas as pd 
import wandb

def get_df(project_path):

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(project_path)

    summary_list, config_list, name_list = [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    df = pd.json_normalize([summary_list[i] | config_list[i] for i in range(len(summary_list))], sep='_')
    return df

def get_last_result(sub_df):
    return round(sub_df.sort_values('valid_acc', ascending=False).iloc[0]['test_acc'] * 100, 2)

def get_es_result(sub_df):
    return round(sub_df.sort_values('best_epoch-valid_acc_valid_acc', ascending=False).iloc[0]['best_epoch-valid_acc_test_acc'] * 100, 2)