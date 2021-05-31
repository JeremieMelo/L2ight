import os
import re

def parse_fbs_cg_log():
    root = "log/fmnist/cnn3/fbs_cg"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    for s in [0.2, 0.4, 0.6, 0.8, 0.9]:
        file = os.path.join(root, f'topk_norm-none_fbs-{s}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"fbs_cg: total: {len(run_ids)} runs")
    print("fbs_cg_runs =", run_ids)


if __name__ == "__main__":
    parse_fbs_cg_log()

