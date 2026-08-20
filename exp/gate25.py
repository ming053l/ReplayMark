"""Reads exp/25's outcome and routes the night: confirm on success, fallback otherwise."""
import json, re, subprocess, sys
log = open("/ssd1/ming/basinmark/logs/25_combo.log", errors="ignore").read()
m = re.search(r"R8k10\s+\| sync ([\d.]+) \| TPR@5% ([\d.]+) @1% ([\d.]+) @0\.1% ([\d.]+)", log)
q = re.search(r"R8k10: valid (\d+) \| dNLL median [+\-\d.]+ ratio ([\d.]+)", log)
P = "/home/ming0531/miniconda3/envs/mmada/bin/python"
if not m:
    print("GATE: 25 produced no R8k10 row; running fallback"); nxt = "exp/26_fallback.py"
else:
    t001, ratio = float(m.group(4)), (float(q.group(2)) if q else 9.9)
    print(f"GATE: R8k10@1024 TPR@0.1%={t001} ratio={ratio}")
    if t001 >= 0.70 and ratio <= 1.05:
        print("GATE: target met -> confirmatory lock-in"); nxt = "exp/27_confirm.py"
    else:
        print("GATE: short of target -> fallback levers"); nxt = "exp/26_fallback.py"
r = subprocess.run([P, nxt], capture_output=True, text=True)
open(f"/ssd1/ming/basinmark/logs/{nxt.split('/')[-1]}.log", "w").write(r.stdout + r.stderr)
print(f"GATE: {nxt} exit {r.returncode}")
