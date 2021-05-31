import os
import re

def parse_uniform_log():
    root = "log/fmnist/cnn3/fbs"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    for s in [0.2, 0.4, 0.6, 0.8, 0.9]:
        file = os.path.join(root, f'uniform_norm-none_fbs-{s}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"uniform: total: {len(run_ids)} runs")
    print("uniform_runs =", run_ids)

def parse_topk_log():
    root = "log/fmnist/cnn3/fbs"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    for s in [0.2, 0.4, 0.6, 0.8, 0.9]:
        file = os.path.join(root, f'btopk_norm-none_fbs-{s}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"topk: total: {len(run_ids)} runs")
    print("topk_runs =", run_ids)

def parse_gtopk_log():
    root = "log/fmnist/cnn3/fbs"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    for s in [0.2, 0.4, 0.6, 0.8, 0.9]:
        file = os.path.join(root, f'gtopk_norm-none_fbs-{s}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"gtopk: total: {len(run_ids)} runs")
    print("gtopk_runs =", run_ids)

if __name__ == "__main__":
    parse_uniform_log()
    parse_topk_log()
    parse_gtopk_log()
