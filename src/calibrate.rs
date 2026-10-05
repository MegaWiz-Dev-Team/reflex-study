//! Online confidence calibration: Platt scaling on `logit(p)` with Laplace-smoothed targets
//! (Platt 1999), refit from a rolling window of `(reported confidence, was correct)` pairs.
//!
//! Identity until `min_obs` observations; a fit that loses to identity on the window, or that
//! reverses the ranking (w ≤ 0), falls back to identity. Upstream adds an ULP resolution floor
//! and an AUC check on top of the same shape; this is the plain textbook form.

use crate::sigmoid;

const P_CLIP: f64 = 1e-4;

fn logit(p: f32) -> f64 {
    let p = (p as f64).clamp(P_CLIP, 1.0 - P_CLIP);
    (p / (1.0 - p)).ln()
}

fn sig64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Smoothed log-loss of `sigmoid(w·z + c)` against targets `t`.
fn loss(t: &[f64], z: &[f64], w: f64, c: f64) -> f64 {
    t.iter()
        .zip(z)
        .map(|(&ti, &zi)| {
            let p = sig64(w * zi + c).clamp(1e-12, 1.0 - 1e-12);
            -(ti * p.ln() + (1.0 - ti) * (1.0 - p).ln())
        })
        .sum()
}

/// Newton–Raphson with backtracking on the 2-parameter logistic loss.
fn fit(t: &[f64], z: &[f64]) -> (f64, f64) {
    let (mut w, mut c) = (1.0f64, 0.0f64);
    for _ in 0..64 {
        let (mut g0, mut g1, mut h00, mut h01, mut h11) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&ti, &zi) in t.iter().zip(z) {
            let p = sig64(w * zi + c);
            let r = p - ti;
            let s = (p * (1.0 - p)).max(1e-12);
            g0 += r * zi;
            g1 += r;
            h00 += s * zi * zi;
            h01 += s * zi;
            h11 += s;
        }
        let det = h00 * h11 - h01 * h01;
        if det.abs() < 1e-12 {
            break;
        }
        let dw = -(h11 * g0 - h01 * g1) / det;
        let dc = -(h00 * g1 - h01 * g0) / det;
        let l0 = loss(t, z, w, c);
        let mut step = 1.0;
        while step > 1e-6 && loss(t, z, w + step * dw, c + step * dc) > l0 {
            step *= 0.5;
        }
        w += step * dw;
        c += step * dc;
        if (step * dw).abs() + (step * dc).abs() < 1e-9 {
            break;
        }
    }
    (w, c)
}

pub struct Calibrator {
    w: f32,
    c: f32,
    window: Vec<(f32, bool)>,
    head: usize,
    capacity: usize,
    min_obs: usize,
}

impl Calibrator {
    pub fn new(capacity: usize, min_obs: usize) -> Self {
        assert!(capacity >= 1);
        Self {
            w: 1.0,
            c: 0.0,
            window: Vec::with_capacity(capacity),
            head: 0,
            capacity,
            min_obs: min_obs.min(capacity),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.w == 1.0 && self.c == 0.0
    }

    pub fn apply(&self, p: f32) -> f32 {
        if self.is_identity() {
            return p;
        }
        sigmoid(self.w * logit(p) as f32 + self.c)
    }

    /// The reported temperature (1/w), upstream's `calibration.temperature`.
    pub fn temperature(&self) -> f32 {
        1.0 / self.w
    }

    pub fn n(&self) -> usize {
        self.window.len()
    }

    /// Record one outcome, refit once the window holds `min_obs`. Returns whether a refit ran.
    pub fn observe(&mut self, p: f32, correct: bool) -> bool {
        if self.window.len() < self.capacity {
            self.window.push((p, correct));
        } else {
            self.window[self.head] = (p, correct);
        }
        self.head = (self.head + 1) % self.capacity;
        if self.window.len() < self.min_obs.max(2) {
            return false;
        }
        let n_pos = self.window.iter().filter(|(_, y)| *y).count() as f64;
        let n_neg = self.window.len() as f64 - n_pos;
        let (t_pos, t_neg) = ((n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0));
        let t: Vec<f64> = self
            .window
            .iter()
            .map(|(_, y)| if *y { t_pos } else { t_neg })
            .collect();
        let z: Vec<f64> = self.window.iter().map(|(p, _)| logit(*p)).collect();
        let (w, c) = fit(&t, &z);
        let better = w.is_finite()
            && c.is_finite()
            && w > 0.0
            && loss(&t, &z, w, c) < loss(&t, &z, 1.0, 0.0);
        (self.w, self.c) = if better { (w as f32, c as f32) } else { (1.0, 0.0) };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_until_min_obs_then_fixes_an_overconfident_readout() {
        let mut cal = Calibrator::new(512, 64);
        assert_eq!(cal.apply(0.9), 0.9);
        // The readout says 0.9 but is right only half the time.
        for i in 0..200 {
            cal.observe(0.9, i % 2 == 0);
            cal.observe(0.2, i % 5 == 0);
        }
        let p = cal.apply(0.9);
        assert!((p - 0.5).abs() < 0.05, "calibrated 0.9 → {p}");
        let q = cal.apply(0.2);
        assert!((q - 0.2).abs() < 0.05, "calibrated 0.2 → {q}");
    }
}
