import numpy as np
from DFA import detrended_fluctuation_analysis

class break_point_detect(detrended_fluctuation_analysis):
    def __init__(self, test_path, width0, step):
        self.path = test_path
        self.width0 = width0
        self.step = step
        super().__init__(test_path=test_path, q_list=[2], centers=None, width=None)

    def coarse_h_curve(self):
        centers = np.arange(self.width0 // 2, len(x) - self.width0 // 2, self.step)
        vals = np.empty(len(centers))
        for j, c in enumerate(centers):
            vals[j] = self.local_h2(int(c - self.width0 / 2), int(c + self.width0 / 2))
        return centers, vals

    def median3(self, a):
        out = a.copy()
        for i in range(1, len(a) - 1):
            out[i] = np.median(a[i - 1 : i + 2])
        return out

    def best_in(self, smoothed, lo, hi):
        seg = smoothed[lo : hi + 1]
        s = np.cumsum(seg - seg.mean())
        return int(np.argmax(np.abs(s))) + lo
    
    def det_cusum(self, k = 3, min_gap = 8):
        centers, vals = self.coarse_h_curve()
        smoothed = self.median3(vals)
        n = len(smoothed)
        cuts = []
        pending = [(0, n - 1)]

        while len(cuts) < k and pending:
            lo, hi = pending.pop(0)
            if hi - lo + 1 < 2 * min_gap:
                continue
            i = self.best_in(smoothed, lo, hi)
            if i - lo < min_gap or hi - i < min_gap:
                continue
            cuts.append(i)
            pending.append((lo, i - 1))
            pending.append((i + 1, hi))
        return centers[sorted(cuts)]

    def pelt_cost(self, a, b, pref1, pref2):
        length = b - a
        diff = pref1[b] - pref1[a]
        rss = (pref2[b] - pref2[a]) - diff * diff / length
        return length * np.log(np.maximum(rss, 1e-12) / length)

    def segment_cost(self, s, e, prefix1, prefix2):
        n = e - s
        if n <= 0:
            return np.inf
        S1 = prefix1[e] - prefix1[s]
        S2 = prefix2[e] - prefix2[s]
        return S2 - S1 * S1 / n

    def pelt_segmentation(self, penalty_multiplier = 6, min_segment = 8):
        centers, vals = self.coarse_h_curve()
        smoothed = self.median3(vals)
        y = np.asarray(smoothed, dtype=float)
        m = len(y)
        if m < 2 * min_segment:
            return []
        pref1 = np.concatenate([[0.0], np.cumsum(y)])
        pref2 = np.concatenate([[0.0], np.cumsum(y * y)])
        beta = penalty_multiplier * np.log(m)
        f = np.full(m + 1, np.inf)
        parent = np.zeros(m + 1, dtype=int)
        f[0] = 0

        for e in range(min_segment, m + 1):
            lo = 0
            hi = e - min_segment
            s_arr = np.arange(lo, hi + 1)
            vals = f[s_arr] + beta + self.pelt_cost(s_arr, e, pref1, pref2)
            k = int(np.argmin(vals))
            f[e] = vals[k]
            parent[e] = s_arr[k]
        starts = []
        e = m
        while e > 0:
            s = int(parent[e])
            starts.append(s)
            e = s
        break_indexes = sorted(starts)[1:]
        return [int(centers[i]) for i in break_indexes]