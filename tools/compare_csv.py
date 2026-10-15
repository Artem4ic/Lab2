#!/usr/bin/env python3
"""Compare two trajectory CSV files: atol=1e-10, rtol=1e-10."""
import argparse
from pathlib import Path
from benchmark import compare

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('serial', type=Path)
p.add_argument('parallel', type=Path)
a = p.parse_args()
error, scaled, cells = compare(a.serial, a.parallel)
print(f'PASS: {cells} values, max_abs={error:.17g}, max_scaled={scaled:.17g}')
