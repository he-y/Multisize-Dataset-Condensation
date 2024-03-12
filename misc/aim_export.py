import os
import time
from aim import Repo

my_repo = Repo(".")

"""
Query date
run.created_at.strftime("%d/%m/%Y") == "03/05/2023"
"""

def aim_log(run, custom_dir, args):
    time.sleep(5)   # wait for 5 seconds to make sure the run is finished
    assert args is not None # store args

    path = os.path.join(custom_dir, f'{run.name}_{args.cur_time}.txt')
    # create if not exists
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        print(f"Writing to file {path}")
        f.write(f"created_at: {str(run.created_at)}")
        f.write(str(args.__dict__))
        f.write('\n')
        for line in run.get_terminal_logs().values.tolist():
            f.write(str(line))
            f.write('\n')