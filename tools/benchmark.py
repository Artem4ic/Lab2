#!/usr/bin/env python3
"""Reproducible comparison and measurements; standard library only."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def capture(args):
    return subprocess.check_output(args, text=True).strip()


def compare(reference, candidate):
    maximum = 0.0
    scaled = 0.0
    count = 0
    with reference.open() as a, candidate.open() as b:
        ra, rb = csv.reader(a), csv.reader(b)
        if next(ra) != next(rb):
            raise AssertionError('Different CSV headers')
        from itertools import zip_longest
        for x, y in zip_longest(ra, rb):
            if x is None or y is None or len(x) != len(y):
                raise AssertionError('Different CSV sizes')
            for u, v in zip(map(float, x), map(float, y)):
                if not math.isfinite(u) or not math.isfinite(v):
                    raise AssertionError('Non-finite value')
                diff = abs(u-v)
                maximum = max(maximum, diff)
                scaled = max(scaled, diff/(1+abs(u)))
                if diff > 1e-10 + 1e-10*abs(u):
                    raise AssertionError(f'Mismatch: {u} vs {v}')
                count += 1
    return maximum, scaled, count


def generate(n, path):
    rng = random.Random(20261015+n)
    with path.open('w') as f:
        f.write(f'{n}\n')
        for i in range(n):
            # Separated points, nonzero 3D motion; bounded masses.
            b = [rng.uniform(1e10, 1e12), 1000.0*(i % 32),
                 1000.0*((i//32) % 32), 1000.0*(i//1024)]
            b += [rng.uniform(-1, 1) for _ in range(3)]
            f.write(' '.join(format(x, '.17g') for x in b)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='measurements_local')
    parser.add_argument('--sizes', nargs='+', type=int, default=[128, 2048])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 2 or min(args.sizes) < 1 or min(args.threads) < 1:
        parser.error('Use positive sizes/threads and at least two repetitions')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'date_utc': capture(['date', '-u', '+%FT%TZ']),
                'platform': platform.platform(), 'lscpu': capture(['lscpu']),
                'gcc': capture(['gcc', '--version']),
                'affinity': sorted(os.sched_getaffinity(0)),
                'arguments': vars(args),
                'method': '1 warm-up per configuration, shuffled configurations in each of 5 rounds (or --repeats); arithmetic mean; dt=0.0625, tend=1, 16 steps, xyz CSV',
                'compiler_flags': '-O2 -std=c11 -Wall -Wextra -Wpedantic; -pthread for parallel; -lm',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(ROOT.glob('*.c'))}}
    for name in ('cpu.max', 'cpuset.cpus.effective', 'memory.max'):
        p = Path('/sys/fs/cgroup') / name
        metadata[name] = p.read_text().strip() if p.exists() else 'not available'
    metadata['os_release'] = Path('/etc/os-release').read_text()
    metadata['method'] = metadata['method'].replace('each of 5 rounds (or --repeats)',
                                                    f'each of {args.repeats} rounds')
    (out/'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    raw, comparisons = [], []
    rng = random.Random(71)
    configurations = [('serial', 1)] + [('pthreads', p) for p in args.threads]
    for n in args.sizes:
        inp = out / f'input_{n}.txt'
        generate(n, inp)
        reference = out / f'trajectory_{n}_serial.csv'
        for round_no in range(args.repeats+1):
            order = configurations[:]
            if round_no > 0:
                rng.shuffle(order)
            for mode, threads in order:
                dest = out/f'trajectory_{n}_{mode}_{threads}.csv'
                command = [str(ROOT/f'nbody_{mode}'), '1', str(inp), '0.0625', str(dest)]
                if mode == 'pthreads':
                    command.append(str(threads))
                command.append('xyz')
                # The serial executable has no numerical suffix in its filename.
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=300, check=True)
                compute = float(re.search(r'Compute time[^=]*=\s*(\S+)', result.stdout)[1])
                total = float(re.search(r'Total time[^=]*=\s*(\S+)', result.stdout)[1])
                if mode == 'serial':
                    reference.write_bytes(dest.read_bytes())
                if round_no > 0:
                    error, scaled, cells = compare(reference, dest)
                    comparisons.append([n, mode, threads, round_no, error, scaled, cells, 'PASS'])
                    raw.append([n, mode, threads, round_no, compute, total])
                print(f'N={n} {mode} P={threads} round={round_no} compute={compute:.6f}s', flush=True)
    with (out/'raw_times.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['N','mode','P','repeat','compute_seconds','total_seconds']);w.writerows(raw)
    with (out/'comparisons.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['N','mode','P','repeat','max_abs','max_scaled','cells','status']);w.writerows(comparisons)
    summary = []
    for n in args.sizes:
        base = statistics.mean(r[4] for r in raw if r[:2]==[n,'serial'])
        for mode, p in configurations:
            records = [r for r in raw if r[:3]==[n,mode,p]]
            comp = [r[4] for r in records]
            total = [r[5] for r in records]
            avg = statistics.mean(comp)
            summary.append([n,mode,p,avg,statistics.stdev(comp),min(comp),max(comp),
                            statistics.mean(total),base/avg,base/avg/p])
    with (out/'summary.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['N','mode','P','compute_mean','compute_stdev','compute_min',
                                  'compute_max','total_mean','speedup','efficiency']);w.writerows(summary)
    print(f'All comparisons PASS; results: {out}', flush=True)


if __name__ == '__main__':
    main()
