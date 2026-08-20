"""Confirmatory lock-in of the 1024 combo on fresh offsets, n=20."""
import sys, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
exec(open("/ssd1/ming/basinmark/exp/25_combo.py").read()
     .replace('skip=850', 'skip=1100').replace('nonce=f"cb-{i}"', 'nonce=f"cf-{i}"')
     .replace('seed=9900', 'seed=11000').replace('NS = b"retrace-key-A", 1024, 32, 16',
                                                 'NS = b"retrace-key-A", 1024, 32, 20')
     .replace('25_combo.json', '27_confirm.json'))
