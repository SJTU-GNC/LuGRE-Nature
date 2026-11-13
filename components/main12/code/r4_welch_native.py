"""Native-bin Hann Welch estimator; no time or frequency interpolation."""
import numpy as np
def welch(y):
    m = min(256, len(y))
    step = m - min(128, m // 2)
    starts = list(range(0, len(y) - m + 1, step))
    w = np.hanning(m)
    energy = np.sum(w * w)
    ps = []
    errors = []
    for start in starts:
        x = y[start:start + m]
        z = (x - x.mean()) * w
        p = np.abs(np.fft.rfft(z)) ** 2 / energy
        p[1:-1 if m % 2 == 0 else None] *= 2
        err = abs(p.sum() / m - np.sum(z * z) / energy)
        errors.append(err)
        assert err < 1e-10
        ps.append(p)
    return (np.fft.rfftfreq(m), np.mean(ps, axis=0), m, starts, max(errors))
