"""Route after the fallback sweep: graduate the better arm to 1024 tokens."""
import re, subprocess
log = open("/ssd1/ming/basinmark/logs/26_fallback.py.log", errors="ignore").read()
best, bt = None, -1
for name in ("R16k05", "R8k10w"):
    m = re.search(rf"{name}\s+\| sync ([\d.]+) \| TPR@5% [\d.]+ @1% ([\d.]+) @0\.1% ([\d.]+) \| ratio ([\d.]+)", log)
    if m and float(m.group(4)) <= 1.10 and float(m.group(2)) > bt:
        best, bt = name, float(m.group(2))
        rate = float(m.group(1))
print(f"GATE26: winner {best} (TPR@1% {bt})" if best else "GATE26: nothing qualified")
if best:
    kw = {"R16k05": "retries=16, p_floor=0.05, s_min=0.5",
          "R8k10w": "retries=8, p_floor=0.10, s_min=0.4"}[best]
    smin_det = "0.4" if best == "R8k10w" else "0.5"
    src = open("/ssd1/ming/basinmark/exp/25_combo.py").read()
    src = src.replace('("R8k10", dict(s_min=0.5, retries=8, p_floor=0.10))',
                      f'("{best}", dict({kw}))')
    src = src.replace('skip=850', 'skip=1250').replace('nonce=f"cb-{i}"', 'nonce=f"gr-{i}"')
    src = src.replace('seed=9900', 'seed=12000').replace('25_combo.json', '28_graduate.json')
    src = src.replace('s_min=0.5,\n                         retries=1).detect',
                      f's_min={smin_det},\n                         retries=1).detect')
    src = src.replace('"R8k10"', f'"{best}"')
    open("/ssd1/ming/basinmark/exp/28_graduate.py", "w").write(src)
    P = "/home/ming0531/miniconda3/envs/mmada/bin/python"
    r = subprocess.run([P, "/ssd1/ming/basinmark/exp/28_graduate.py"],
                      capture_output=True, text=True)
    open("/ssd1/ming/basinmark/logs/28_graduate.log", "w").write(r.stdout + r.stderr)
    print(f"GATE26: graduate exit {r.returncode}")
