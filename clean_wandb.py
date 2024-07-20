#%%
import os
from os.path import join
from glob import glob
import shutil
import wandb
from os.path import exists
import yaml


assert exists('.wandb_config/setup.yaml'), "Please configure your wandb settings first."
with open('.wandb_config/setup.yaml', 'r') as stream:
    wandb_config = yaml.safe_load(stream)
wandb_project = f"{wandb_config['entity']}/{wandb_config['project']}"
print(f"==> Project: {wandb_project}")

api = wandb.Api()
runs = api.runs(wandb_project)

log_dir = 'logs'
print('!!! Carefully check: ', runs, '<=>', log_dir, ' !!!')
runs_remote = []
for run in runs:
    runs_remote.append(run.name)

exclude_list = ['figures']
runs_local = glob(join(log_dir, '*'))
for runs_local_folder in runs_local:
    if os.path.basename(runs_local_folder) in exclude_list:
        runs_local.remove(runs_local_folder)

# print("="*20)
# print("The following runs will be removed:")
cnt_removed = 0
for run in runs_local:
    if run.split('/')[-1] not in runs_remote:
        print(run)
        cnt_removed += 1

# print("="*20)
# print("The following runs will NOT be removed:")
cnt_kept = 0
for run in runs_local:
    if run.split('/')[-1] in runs_remote:
        # print(run)
        cnt_kept += 1
# print("="*20)

print(f"Total {len(runs_local)} runs, {cnt_removed} will be removed, {cnt_kept} will be kept.")

confirm = input("Confirm you want to delete them? y/[n]:")
if confirm == 'y':
    for run in runs_local:
        if run.split('/')[-1] not in runs_remote:
            shutil.rmtree(run)
    print('Deleted.')
else:
    print('Aborted.')
