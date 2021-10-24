'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:42:43
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:42:43
'''
import os
import re

def parse_ic_map_log():
    root = "log/cifar10/vgg8/cs"
    p = re.compile(r".*Run ID: \(([0-9a-z]+)\).*PID.*")
    run_ids = []
    accs = [59.62,66.23,69.89,76.06,83.19,85.02,86.35]
    for acc in accs:
        file = os.path.join(root, f'ds-0.5_fbs-0.6_norm-none_first-0_ss-0_cs-0.6_ic-400_mapacc-{acc}.log')
        with open(file, "r") as f:
            lines = f.read()
            res = p.search(lines)
            run_ids.append(res.groups(1)[0])
    print(f"map: total: {len(run_ids)} runs")
    print("accs =", accs)
    print("runs =", run_ids)


if __name__ == "__main__":
    parse_ic_map_log()
