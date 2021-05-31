import os
import re

def parse_ds_log():
    root = "log/fmnist/cnn3/ds"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    for s in [0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        file = os.path.join(root, f'btopk-exp-fbs-0.6-exp-ss-0-cs-0.6-smd-ds-{s}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"ds: total: {len(run_ids)} runs")
    print("runs =", run_ids)


if __name__ == "__main__":
    parse_ds_log()
