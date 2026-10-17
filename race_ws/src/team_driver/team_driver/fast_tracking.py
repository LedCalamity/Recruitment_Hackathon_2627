"""Fast raceline tracker (ROS-free). Same code as tools/headless_sim/fastctl.py, validated in the headless sim.

Speed profile from line curvature (optionally per-corner grip scale), pure pursuit on base_link (rear axle),
pose predicted tau s ahead, smooth merge from the grid slot, acceleration-type speed command for the
sim's P speed loop (accelerating kp 4.755, braking kp 19.02)."""
import math, numpy as np
from scipy.interpolate import CubicSpline
WB = 0.33; SMAX = 0.4189; G = 9.81

def spline_closed(xy, ds=0.05):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    keep = seg > 1e-6; xy = xy[keep]; seg = seg[keep]
    s = np.r_[0, np.cumsum(seg)]
    cs = CubicSpline(s, np.vstack([xy, xy[:1]]), bc_type='periodic')
    L = s[-1]; n = int(round(L/ds)); ss = np.linspace(0, L, n, endpoint=False)
    p = cs(ss); d1 = cs(ss, 1); d2 = cs(ss, 2)
    k = (d1[:, 0]*d2[:, 1]-d1[:, 1]*d2[:, 0])/np.linalg.norm(d1, axis=1)**3
    return ss, p, d1/np.linalg.norm(d1, axis=1)[:, None], k, L

def smooth_periodic(a, w):
    if w <= 1: return a
    ker = np.ones(w)/w; pad = np.r_[a[-w:], a, a[:w]]
    return np.convolve(pad, ker, 'same')[w:-w]

def speed_profile(s, k, L, vmax=8.0, alat=8.0, acc=6.0, brk=7.0, ellipse=True):
    n = len(s); ds = L/n
    alat = np.broadcast_to(np.asarray(alat, float), k.shape)
    v = np.minimum(vmax, np.sqrt(alat/np.maximum(np.abs(k), 1e-6)))
    for _ in range(2):
        for i in range(2*n):  # forward
            a, b = i % n, (i+1) % n
            f = max(0., 1-(v[a]**2*abs(k[a])/alat[a])**2) if ellipse else 1.0
            v[b] = min(v[b], math.sqrt(v[a]**2+2*acc*math.sqrt(f)*ds))
        for i in range(2*n, 0, -1):  # backward
            b, a = i % n, (i-1) % n
            f = max(0., 1-(v[b]**2*abs(k[b])/alat[b])**2) if ellipse else 1.0
            v[a] = min(v[a], math.sqrt(v[b]**2+2*brk*math.sqrt(f)*ds))
    return v

def lap_time_est(L, v):
    vv = np.r_[v, v[:1]]; ds = L/len(v)
    return float(np.sum(2*ds/(vv[:-1]+vv[1:])))

class Tracker:
    def __init__(self, xy, vmax=8.0, alat=8.0, acc=6.0, brk=7.0, la0=0.35, la_k=0.18, la_min=0.4, la_max=2.0,
                 mode='pp', v_lead=0.15, k_smooth=5, k_lat=1.5, k_head=1.0, acc_cmd=True, horizon=0.12, tau=0.0, a_lim=9.5, merge_len=6.0, ff_preview=0.05, v0=1.0, yaw_pred=0.0, ellipse=True, seg_bounds=None, seg_scale=None):
        self.s, self.p, self.t, k, self.L = spline_closed(xy)
        self.k = smooth_periodic(k, k_smooth)
        a_arr = np.full(len(self.s), float(alat))
        if seg_bounds is not None and seg_scale is not None:
            # seg_bounds: sorted segment start stations [m]; segment i runs to bound i+1 (last wraps to first)
            idx = (np.searchsorted(np.asarray(seg_bounds), self.s, side='right')-1) % len(seg_bounds)
            a_arr = a_arr*np.asarray(seg_scale, float)[idx]
        self.v = speed_profile(self.s, self.k, self.L, vmax, a_arr, acc, brk, ellipse)
        self.n = len(self.s); self.ds = self.L/self.n
        self.la0, self.la_k, self.la_min, self.la_max = la0, la_k, la_min, la_max
        self.mode = mode; self.v_lead = v_lead; self.k_lat = k_lat; self.k_head = k_head
        self.acc_cmd, self.horizon, self.tau, self.a_lim = acc_cmd, horizon, tau, a_lim
        self.merge_len = merge_len; self.merge = None; self.ff_preview = ff_preview; self.v0 = v0
        self.nrm = np.c_[-self.t[:, 1], self.t[:, 0]]
        self.last_st = 0.0; self.yaw_pred = yaw_pred
        self.idx = None
        self.est = lap_time_est(self.L, self.v)

    def nearest(self, x, y):
        if self.idx is None:
            d = np.hypot(self.p[:, 0]-x, self.p[:, 1]-y); self.idx = int(np.argmin(d)); return self.idx
        w = np.arange(self.idx-20, self.idx+80) % self.n
        d = np.hypot(self.p[w, 0]-x, self.p[w, 1]-y); self.idx = int(w[np.argmin(d)]); return self.idx

    def __call__(self, x, y, yaw, v, t):
        i = self.nearest(x, y)
        if self.tau > 0:  # predict pose forward (kinematic bicycle, last steering) to cover latency + steering lag
            w = v*math.tan(self.last_st)/WB*self.yaw_pred
            x, y = x+v*self.tau*math.cos(yaw+0.5*w*self.tau), y+v*self.tau*math.sin(yaw+0.5*w*self.tau)
            yaw = yaw+w*self.tau
        if self.merge is None:  # first call: remember lateral offset of the grid slot, fade it out
            e0 = float((x-self.p[i, 0])*self.nrm[i, 0]+(y-self.p[i, 1])*self.nrm[i, 1]); self.merge = (i, e0)
        if self.mode == 'pp':
            la = float(np.clip(self.la0+self.la_k*v, self.la_min, self.la_max))
            if self.merge[1] != 0: la = max(la, 1.2)
            j = (i+int(round(la/self.ds))) % self.n
            tx_, ty_ = self.p[j]
            i0, e0 = self.merge; prog = ((j-i0) % self.n)*self.ds
            if e0 != 0 and prog < self.merge_len:
                f = 0.5*(1+math.cos(math.pi*prog/self.merge_len)); tx_ += e0*f*self.nrm[j, 0]; ty_ += e0*f*self.nrm[j, 1]
            elif prog >= self.merge_len and prog < self.L/2: self.merge = (i0, 0.0)
            dx, dy = tx_-x, ty_-y
            c, s = math.cos(yaw), math.sin(yaw)
            lx, ly = c*dx+s*dy, -s*dx+c*dy
            st = math.atan2(2*WB*ly, lx*lx+ly*ly)
        else:  # feedforward curvature + heading/lateral feedback at the front axle (Stanley-like)
            fx, fy = x+WB*math.cos(yaw), y+WB*math.sin(yaw)
            jf = (i+int(round(WB/self.ds))) % self.n
            w = np.arange(jf-15, jf+15) % self.n
            jf = int(w[np.argmin(np.hypot(self.p[w, 0]-fx, self.p[w, 1]-fy))])
            i0, e0 = self.merge; prog = ((jf-i0) % self.n)*self.ds; off = 0.0
            if e0 != 0 and prog < self.merge_len: off = e0*0.5*(1+math.cos(math.pi*prog/self.merge_len))
            elif prog >= self.merge_len and prog < self.L/2: self.merge = (i0, 0.0)
            tx, ty = self.t[jf]; nx, ny = -ty, tx
            ex = (fx-self.p[jf, 0])*nx+(fy-self.p[jf, 1])*ny - off   # +left of path
            he = math.atan2(math.sin(math.atan2(ty, tx)-yaw), math.cos(math.atan2(ty, tx)-yaw))
            jp = (jf+int(round(self.ff_preview*v/self.ds))) % self.n
            st = math.atan(WB*self.k[jp]) + self.k_head*he - math.atan2(self.k_lat*ex, self.v0+v)
        st = float(np.clip(st, -SMAX, SMAX)); self.last_st = st
        if not self.acc_cmd:
            jv = (i+int(round(self.v_lead*max(v, 1)/self.ds))) % self.n
            return st, float(self.v[jv])
        if self.acc_cmd == 'ff':
            # Latency-tolerant: feedforward from the profile only (no measured speed in the loop).
            # cmd = v_ref + a_ref/kp so the sim's P speed loop produces the planned acceleration.
            jv = (i+max(1, int(round(self.v_lead*max(v, 1)/self.ds)))) % self.n
            vr = float(self.v[jv]); vn = float(self.v[(jv+1) % self.n])
            a = float(np.clip((vn*vn-vr*vr)/(2*self.ds), -self.a_lim, self.a_lim))
            return st, vr + (a/4.755 if a > 0 else a/19.02)
        # speed P controller in sim: accel kp=4.755 (accelerating) / 19.02 (braking).
        # Command the speed that yields the profile's acceleration over a short horizon.
        d = max(v*self.horizon, 0.1); jv = (i+max(1, int(round(d/self.ds)))) % self.n
        vt = float(self.v[jv]); a = (vt*vt-v*v)/(2*d)
        a = float(np.clip(a, -self.a_lim, self.a_lim))
        cmd = v + (a/4.755 if a > 0 else a/19.02)
        if v < 0.5: cmd = max(cmd, vt)
        return st, float(cmd)
