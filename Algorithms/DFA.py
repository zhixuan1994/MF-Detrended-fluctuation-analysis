import numpy as np

class detrended_fluctuation_analysis():
    def __init__(self, test_path, q_list, centers, width, break_t = None):
        self.path = test_path
        self.q_list = np.array(q_list)
        self.centers = centers
        self.width = width
        self.break_t = break_t

    def _poly_projection(self, s, m):
        if s <= m:
            return np.eye(s)
        t = np.arange(s, dtype=float) / max(1, s - 1)
        x = np.vander(t, m + 1, increasing=True)
        return x @ np.linalg.pinv(x.T @ x) @ x.T

    def detrend_poly(self, segs, s, m):
        lev = self._poly_projection(s, m)
        return segs - segs @ lev

    def mfdfa_scaling(self, x, scales, detrend_fun):
        x = np.asarray(x, dtype=float)
        y = np.cumsum(x - x.mean())
        n = len(y)
        EPS = 1e-14
        q_list = np.atleast_1d(self.q_list)
        fq = np.full((len(q_list), len(scales)), np.nan)

        for si, s in enumerate(scales):
            ns = n // s
            if ns < 2 or s < 4:
                continue
            segs = []
            for v in range(ns):
                segs.append(y[v * s : (v + 1) * s])
            for v in range(ns):
                segs.append(y[n - (v + 1) * s : n - v * s])
            segs = np.asarray(segs)
            resid = detrend_fun(segs, int(s))
            f2 = np.mean(resid**2, axis=1)
            f2 = np.clip(f2, EPS, None)
            for qi, q in enumerate(q_list):
                if q == 0:
                    fq[qi, si] = np.exp(0.5 * np.mean(np.log(f2)))
                else:
                    fq[qi, si] = np.mean(f2 ** (q / 2.0)) ** (1.0 / q)
        return fq

    def fit_hq(self, scales, fq, s_min = 24.0):
            valid = ~np.isnan(fq).any(axis=0) & (scales >= s_min)
            log_s = np.log10(scales[valid])
            out = np.empty(len(self.q_list))
            for qi in range(len(self.q_list)):
                log_f = np.log10(fq[qi, valid])
                if len(log_s) < 5:
                    out[qi] = np.nan
                else:
                    out[qi] = np.polyfit(log_s, log_f, 1)[0]
            return out

    def local_h2(self, i0, i1, m = 2):
        seg = self.path[i0:i1]
        if len(seg) < 64:
            return np.nan
        scales = np.unique(
            np.logspace(np.log10(16.0), np.log10(len(seg) // 6), num=12).astype(int)
        )
        fun = lambda segs, s: self.detrend_poly(segs, s, m)
        fq = self.mfdfa_scaling(seg, scales, fun)
        return self.fit_hq(scales, fq, s_min=16.0)

    # Fixed DFA
    def fixed_DFA(self):
        self.Hq_list = self.local_h2(0,len(self.path))

    # Hard Split DFA
    def hard_split_DFA(self, break_t):
        out =  []
        half = self.width // 2
        for c in self.centers:
            lo = max(0, int(c - half))
            hi = min(len(self.path), int(c + half))
            for bt in break_t:
                bt = int(bt)
                if lo < bt < hi:
                    if c < bt:
                        lo = max(0, bt - self.width)
                        hi = bt
                    else:
                        lo = bt
                        hi = min(len(self.path), bt + self.width)
            out.append(self.local_h2(lo, hi))
        self.Hq_list = np.array(out)

    def soft_split_DFA(self, break_t, sigma_min = 16):
        out = []
        half = self.width // 2
        sigma = max(sigma_min, self.width / 6.0)
        h_left_list, h_right_list = [], []
        for bt in break_t:
            h_left_list.append(self.local_h2(max(0, bt - self.width), bt))
            h_right_list.append(self.local_h2(bt, min(len(x), bt + self.width)))
        for c in self.centers:
            lo = max(0, int(c - half))
            hi = min(len(x), int(c + half))
            if len(break_t) == 0:
                out.append(self.local_h2(lo, hi))
            else:
                check_yes = 0
                for i in range(len(break_t)):
                    bt = int(break_t[i])
                    h_left = h_left_list[i]
                    h_right = h_right_list[i]
                    if lo < bt < hi:
                        w_right = 1.0 / (1.0 + np.exp(-(float(c) - bt) / sigma))
                        temp = (1.0 - w_right) * h_left + w_right * h_right
                        check_yes = 1
                        
                if check_yes == 1:
                    out.append(temp)
                else:
                    out.append(self.local_h2(lo, hi))
        self.Hq_list = np.array(out)

    def tau_q(self):
        Hq_array = self.Hq_list
        tau_q_array = Hq_array * self.q_list.reshape(1,-1) - 1
        alpha_list, falpha_list, dalpha_list = [], [], []
        q_list = self.q_list.reshape(-1,)
        for tau_one in tau_q_array:
            tau_one = tau_one.reshape(-1,)
            alpha_one = np.gradient(tau_one, q_list)
            falpha_one = q_list * alpha_one - tau_one
            alpha_list.append(alpha_one)
            falpha_list.append(falpha_one)
            dalpha_list.append(float(np.nanmax(alpha_one) - np.nanmin(alpha_one)))
        self.tau_list = tau_q_array
        self.alpha_list = np.array(alpha_list)
        self.falpha_list = np.array(falpha_list)
        self.dalpha_list = np.array(dalpha_list)
