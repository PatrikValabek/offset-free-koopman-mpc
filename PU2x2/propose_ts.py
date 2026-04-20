"""Propose a sampling time Ts for 1 s ident data.

Heuristic: find step changes in any input channel, measure the time each
output channel needs to reach 90 % of its steady-state change (t90).
Thermal plants usually have TWO very different modes (e.g. direct u->y
coupling ~seconds and slow heat transfer ~minutes). A single average
hides that, so we split the t90 population into fast/slow clusters in
log-space and report Ts for each:

  * Ts_fast   = t90_fast / N_TARGET   -> resolves the fast mode (no aliasing)
  * Ts_slow   = t90_slow / N_TARGET   -> only fine for the slow mode
  * horizon   = t90_slow_p90          -> min MPC horizon to cover slow mode

Minimal, general. Edit DATA_DIR / OUT_VARS / IN_VARS / EXP_RANGE as needed.
"""

import os
import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.join(os.path.dirname(__file__), 'ident_data')
OUT_VARS = ['T2', 'T4']           # output channels to load
IN_VARS = ['u2', 'u3']            # input channels to load
EXP_RANGE = range(0, 11)          # which experiment indices to include
DT = 1.0                          # raw sampling time [s]
N_TARGET = 12                     # desired samples within t90 (aim 10-15)
STEP_REL_THRESH = 0.05            # min |du| / range(u) to count as a step
MIN_STEP_GAP = 5                  # min samples between detected steps
MAX_HORIZON_S = 600               # max seconds to search for 90 % response


def load_exp(idx):
    """Return (Y, U) arrays shaped (N, ny), (N, nu) or None if missing."""
    try:
        Y = np.hstack([
            loadmat(f'{DATA_DIR}/{v}_ident_{idx}.mat')[v].reshape(-1, 1)
            for v in OUT_VARS
        ])
        U = np.hstack([
            loadmat(f'{DATA_DIR}/{v}_ident_{idx}.mat')[v].reshape(-1, 1)
            for v in IN_VARS
        ])
    except FileNotFoundError:
        return None
    n = min(len(Y), len(U))
    return Y[:n], U[:n]


def find_steps(u, rel_thresh=STEP_REL_THRESH, min_gap=MIN_STEP_GAP):
    """Return indices (in the shifted array) where a notable step starts."""
    du = np.diff(u, axis=0)
    rng = np.ptp(u, axis=0)
    rng[rng == 0] = 1.0
    mag = np.max(np.abs(du) / rng, axis=1)
    idxs = np.where(mag > rel_thresh)[0]
    if len(idxs) == 0:
        return idxs
    keep = [idxs[0]]
    for i in idxs[1:]:
        if i - keep[-1] >= min_gap:
            keep.append(i)
    return np.array(keep)


def t90_for_step(y_seg, frac=0.9):
    """Time (in samples) for y_seg to reach `frac` of its steady-state change.

    Steady-state is estimated from the last 10 % of the window.
    Returns None if the step is too small or never crosses the threshold.
    """
    y0 = y_seg[0]
    tail = max(1, len(y_seg) // 10)
    yss = y_seg[-tail:].mean()
    delta = yss - y0
    if abs(delta) < 1e-6:
        return None
    target = y0 + frac * delta
    crossed = (y_seg >= target) if delta > 0 else (y_seg <= target)
    hits = np.where(crossed)[0]
    return int(hits[0]) if len(hits) else None


def split_modes(arr):
    """Split 1D positive samples into (fast, slow) using the largest gap in
    sorted log-space. Returns (fast, slow) arrays; slow may be empty if the
    distribution is unimodal."""
    if len(arr) < 4:
        return arr, np.array([])
    s = np.sort(arr)
    gaps = np.diff(np.log(s))
    k = int(np.argmax(gaps))
    # Require a clear bimodal gap: largest gap >> median gap, and split
    # must leave at least ~10 % of points on each side.
    if gaps[k] < 3 * np.median(gaps) or k < max(1, len(s) // 10) or k > len(s) - max(1, len(s) // 10):
        return s, np.array([])
    return s[:k + 1], s[k + 1:]


def snap(ts):
    grid = np.array([1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300])
    return int(grid[np.argmin(np.abs(grid - ts))])


def summary(arr, label):
    q = np.percentile(arr, [10, 50, 90])
    print(f'  {label:4s}  n={len(arr):3d}  p10={q[0]:6.1f} s  p50={q[1]:6.1f} s  p90={q[2]:6.1f} s')
    return q


def analyse():
    t90_samples = []
    n_steps_total = 0

    for i in EXP_RANGE:
        data = load_exp(i)
        if data is None:
            continue
        Y, U = data
        steps = find_steps(U)
        if len(steps) == 0:
            continue

        horizon = int(MAX_HORIZON_S / DT)
        next_step = np.append(steps[1:], len(Y) - 1)

        for s, s_next in zip(steps, next_step):
            start = s + 1
            end = min(start + horizon, s_next)
            if end - start < 10:
                continue
            n_steps_total += 1
            for col in range(Y.shape[1]):
                t = t90_for_step(Y[start:end, col])
                if t is not None and t > 0:
                    t90_samples.append(t)

    if not t90_samples:
        raise RuntimeError('No usable step responses detected; relax thresholds.')

    arr = np.array(t90_samples, dtype=float) * DT
    return arr, n_steps_total


def main():
    arr, n_steps = analyse()
    fast, slow = split_modes(arr)
    bimodal = len(slow) > 0

    print(f'Analysed {n_steps} step changes, {len(arr)} t90 measurements')
    print('Distribution of t90 (percentiles):')
    summary(arr, 'all')
    if bimodal:
        summary(fast, 'fast')
        summary(slow, 'slow')
        pop_fast, pop_slow = fast, slow
        label_fast, label_slow = 'fast-mode median', 'slow-mode median'
    else:
        print('  (no clear bimodal gap; treating tails as fast/slow populations)')
        pop_fast = arr[arr <= np.percentile(arr, 25)]
        pop_slow = arr[arr >= np.percentile(arr, 75)]
        label_fast, label_slow = 'lower-quartile (fast tail)', 'upper-quartile (slow tail)'

    t90_fast = float(np.median(pop_fast))
    t90_slow = float(np.median(pop_slow))
    horizon_s = float(np.percentile(arr, 90))

    print()
    print('Sampling time candidates (aim N={} samples per t90):'.format(N_TARGET))
    for pct in (10, 25, 50, 75):
        t = float(np.percentile(arr, pct))
        ts = t / N_TARGET
        print(f'  Ts @ t90 p{pct:<2d} = {t:6.1f} s  ->  Ts = {ts:6.2f} s  '
              f'(snapped {snap(ts)} s)')

    print()
    print(f'=> Resolve {label_fast} ({t90_fast:.0f} s): '
          f'Ts = {t90_fast/N_TARGET:.2f} s  (snapped {snap(t90_fast/N_TARGET)} s)')
    print(f'=> Resolve {label_slow} ({t90_slow:.0f} s): '
          f'Ts = {t90_slow/N_TARGET:.2f} s  (snapped {snap(t90_slow/N_TARGET)} s)')
    print(f'   Using the slow-mode Ts aliases the faster responses; only OK if '
          f'the fast mode is a near-static gain for your controller.')
    print(f'Min MPC horizon to cover p90 of t90 (~95% settle): >= {horizon_s:.0f} s')
    ts_rec = snap(t90_fast / N_TARGET)
    print(f'   e.g. with Ts={ts_rec} s -> Np >= '
          f'{int(np.ceil(horizon_s / max(ts_rec, 1)))} steps')


if __name__ == '__main__':
    main()
