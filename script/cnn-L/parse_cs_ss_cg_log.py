import os
import re

def parse_cs_cg_log():
    root = "log/fmnist/cnn3/cs_cg"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    for s in [0.2, 0.4, 0.6, 0.8, 0.9]:
        file = os.path.join(root, f'norm-none_ss-0_cs-{s}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"cs_cg: total: {len(run_ids)} runs")
    print("cs_cg_runs =", run_ids)

def parse_ss_cg_log():
    root = "log/fmnist/cnn3/ss_cg"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    for s in [0.2, 0.4, 0.6, 0.8, 0.9]:
        file = os.path.join(root, f'norm-none_cs-0_ss-{s}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"ss_cg: total: {len(run_ids)} runs")
    print("ss_cg_runs =", run_ids)


if __name__ == "__main__":
    parse_cs_cg_log()
    parse_ss_cg_log()

