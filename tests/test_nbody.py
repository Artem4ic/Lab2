"""Small independent checks of physics, Euler steps and parallel splitting."""
import csv
import math
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
G = 6.67430e-11


def reference(bodies, end, dt):
    # Independent per-body summation intentionally computes each pair twice.
    bodies = [b[:] for b in bodies]
    time = 0.0
    rows = [[time] + [v for b in bodies for v in b[1:4]]]
    while time < end:
        next_time = end if end - time <= dt else time + dt
        h = next_time - time
        new = []
        for i, b in enumerate(bodies):
            acceleration = [0.0] * 3
            for j, other in enumerate(bodies):
                if i == j:
                    continue
                delta = [other[k] - b[k] for k in range(1, 4)]
                distance = math.sqrt(sum(x*x for x in delta))
                for k in range(3):
                    acceleration[k] += G * other[0] * delta[k] / distance**3
            new.append([b[0]] + [b[k+1] + h*b[k+4] for k in range(3)]
                       + [b[k+4] + h*acceleration[k] for k in range(3)])
        bodies = new
        time = next_time
        rows.append([time] + [v for b in bodies for v in b[1:4]])
    return rows


class NbodyTests(unittest.TestCase):
    def run_program(self, bodies, end=0.07, dt=0.01, threads=None, fmt='xyz'):
        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / 'input.txt'
            out = Path(directory) / 'output.csv'
            inp.write_text(str(len(bodies)) + '\n' + '\n'.join(
                ' '.join(map(str, body)) for body in bodies) + '\n')
            exe = 'nbody_serial' if threads is None else 'nbody_pthreads'
            args = [str(ROOT / exe), str(end), str(inp), str(dt), str(out)]
            if threads is not None:
                args.append(str(threads))
            result = subprocess.run(args + [fmt], capture_output=True, text=True,
                                    timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            with out.open() as file:
                reader = csv.reader(file)
                header = next(reader)
                rows = [list(map(float, row)) for row in reader]
            return header, rows

    def assert_rows(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertTrue(math.isclose(x, y, rel_tol=2e-12, abs_tol=1e-12),
                                (x, y))

    def test_one_body_linear_3d(self):
        bodies = [[2, 1, -2, 3, 4, -5, 6]]
        for threads in (None, 1, 2, 4):
            _, rows = self.run_program(bodies, end=0.025, threads=threads)
            self.assert_rows(rows, reference(bodies, 0.025, 0.01))
            self.assertEqual(rows[-1][0], 0.025)

    def test_two_body_old_velocity_and_force_sign(self):
        bodies = [[1e12, -10, 0, 0, 0, 0, 0],
                  [2e12, 10, 0, 0, 0, 0, 0]]
        for threads in (None, 1, 2, 4):
            _, rows = self.run_program(bodies, end=2, dt=1, threads=threads)
            self.assertEqual(rows[1][1:], rows[0][1:])
            self.assertAlmostEqual(rows[2][1], -10 + G*2e12/400)
            self.assertAlmostEqual(rows[2][4], 10 - G*1e12/400)
            # Newton's third law: the centre of mass stays stationary.
            for row in rows:
                self.assertAlmostEqual((row[1] + 2*row[4])/3, 10/3)

    def test_random_3d_and_nondivisible_work(self):
        rng = random.Random(19)
        for n in (2, 5, 11):
            bodies = [[rng.uniform(1e10, 1e12)]
                      + [rng.uniform(-100, 100) for _ in range(3)]
                      + [rng.uniform(-1, 1) for _ in range(3)] for _ in range(n)]
            expected = reference(bodies, 0.07, 0.01)
            for threads in (None, 1, 2, 3, 8):
                with self.subTest(n=n, threads=threads):
                    _, rows = self.run_program(bodies, threads=threads)
                    self.assert_rows(rows, expected)

    def test_zero_time_and_xy_format(self):
        for threads in (None, 3):
            header, rows = self.run_program([[1, 1, 2, 3, 4, 5, 6]],
                                            end=0, threads=threads, fmt='xy')
            self.assertEqual(header, ['t', 'x1', 'y1'])
            self.assertEqual(rows, [[0, 1, 2]])

    def test_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / 'input.txt'
            out = Path(directory) / 'output.csv'
            for exe in ('nbody_serial', 'nbody_pthreads'):
                cases = [('0\n', '1', '0.01'),
                         ('-1\n', '1', '0.01'),
                         ('1\n-1 0 0 0 0 0 0\n', '1', '0.01'),
                         ('1\n1 nan 0 0 0 0 0\n', '1', '0.01'),
                         ('1\n1 0 0\n', '1', '0.01'),
                         ('1\n1 0 0 0 0 0 0\nextra', '1', '0.01'),
                         ('1\n1 0 0 0 0 0 0\n', '-1', '0.01'),
                         ('1\n1 0 0 0 0 0 0\n', '1', '0'),
                         ('2\n1 0 0 0 0 0 0\n1 0 0 0 0 0 0\n', '1', '0.01')]
                for data, end, dt in cases:
                    inp.write_text(data)
                    result = subprocess.run([str(ROOT/exe), end, str(inp), dt,
                                             str(out)], capture_output=True, timeout=15)
                    self.assertNotEqual(result.returncode, 0)
            inp.write_text('1\n1 0 0 0 0 0 0\n')
            for threads in ('0', '-1', 'abc', '1025'):
                result = subprocess.run([str(ROOT/'nbody_pthreads'), '1', str(inp),
                                         '0.01', str(out), threads],
                                        capture_output=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
