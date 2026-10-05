//! Fit YOUR confidence threshold from a labeled slice instead of guessing one.
//!
//! Two postures, both with upstream's thin-support rule (fewer than 16 observations → no
//! recommendation) and the support disclosed with every number:
//!
//! - percentile: the ρ-quantile of the confidences (sorted ascending, index `floor(n·ρ)`), so
//!   roughly ρ of in-distribution questions abstain;
//! - target accuracy: the LOWEST cutoff whose at-or-above accuracy meets the target; when none
//!   does, the best-achievable cutoff, reported as unmet.

use serde::{Deserialize, Serialize};

pub const THIN_SUPPORT_FLOOR: usize = 16;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    pub threshold: f32,
    /// Observations the fit saw.
    pub n: usize,
    /// Observations at or above the threshold (answered).
    pub n_pass: usize,
    /// Observations below it (abstained).
    pub n_abstain: usize,
    /// Accuracy on the answered set.
    pub answered_accuracy: f64,
    /// Target posture only: whether the target accuracy was reached.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub met: Option<bool>,
}

fn summarize(obs: &[(f32, bool)], threshold: f32, met: Option<bool>) -> Recommendation {
    let pass: Vec<_> = obs.iter().filter(|(c, _)| *c >= threshold).collect();
    let right = pass.iter().filter(|(_, y)| *y).count();
    Recommendation {
        threshold,
        n: obs.len(),
        n_pass: pass.len(),
        n_abstain: obs.len() - pass.len(),
        answered_accuracy: if pass.is_empty() { 0.0 } else { right as f64 / pass.len() as f64 },
        met,
    }
}

/// `obs` = `(confidence, was_the_top_answer_correct)`.
pub fn percentile(obs: &[(f32, bool)], rho: f64) -> Option<Recommendation> {
    assert!((0.0..=1.0).contains(&rho));
    if obs.len() < THIN_SUPPORT_FLOOR {
        return None;
    }
    let mut c: Vec<f32> = obs.iter().map(|(c, _)| *c).collect();
    c.sort_by(f32::total_cmp);
    let idx = ((c.len() as f64 * rho) as usize).min(c.len() - 1);
    Some(summarize(obs, c[idx], None))
}

pub fn target_accuracy(obs: &[(f32, bool)], target: f64) -> Option<Recommendation> {
    if obs.len() < THIN_SUPPORT_FLOOR {
        return None;
    }
    let mut cuts: Vec<f32> = obs.iter().map(|(c, _)| *c).collect();
    cuts.sort_by(f32::total_cmp);
    cuts.dedup();
    let mut best: Option<Recommendation> = None;
    for &t in &cuts {
        let r = summarize(obs, t, Some(false));
        if r.answered_accuracy >= target {
            return Some(Recommendation { met: Some(true), ..r });
        }
        if best.as_ref().is_none_or(|b| r.answered_accuracy > b.answered_accuracy) {
            best = Some(r);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_support_gives_no_recommendation() {
        let obs = vec![(0.5, true); 15];
        assert!(percentile(&obs, 0.3).is_none());
        assert!(target_accuracy(&obs, 0.9).is_none());
    }

    #[test]
    fn percentile_and_target_pick_the_expected_cutoffs() {
        // Confidences 0.00..0.19; correct exactly when confidence ≥ 0.10.
        let obs: Vec<(f32, bool)> = (0..20).map(|i| (i as f32 / 100.0, i >= 10)).collect();
        let p = percentile(&obs, 0.3).unwrap();
        assert_eq!((p.threshold, p.n_pass, p.n_abstain), (0.06, 14, 6));
        let t = target_accuracy(&obs, 0.9).unwrap();
        assert_eq!(t.met, Some(true));
        assert_eq!(t.threshold, 0.09); // 10/11 right at ≥ 0.09
        let unreachable = target_accuracy(&obs.iter().map(|(c, _)| (*c, false)).collect::<Vec<_>>(), 0.5);
        assert_eq!(unreachable.unwrap().met, Some(false));
    }
}
