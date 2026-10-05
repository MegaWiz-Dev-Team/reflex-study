//! The modelless decision engine: corpus → domains → per-question verdicts.

use crate::calibrate::Calibrator;
use crate::corpus::DomainDocs;
use crate::drafter::Drafter;
use crate::embed::{Embedder, dot, unit_centroid};
use crate::gate::{DistanceGate, GATE_MID, GATE_SCALE};
use crate::sigmoid;
use crate::tokenize::Tokenizer;
use crate::wire::{
    Answer, Calibration, DecisionRequest, DecisionResponse, Kind, Outcome, Routing, WireError,
};

/// Up to this many options the confidence is `1 − H/ln K`; above it, the top probability.
pub const NARROW_MAX_OPTIONS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct EngineConfig {
    /// Sigmoid temperature on the drafter's byte delta.
    pub score_temperature: f32,
    /// Abstain when the (calibrated) confidence is below this.
    pub score_threshold: f32,
    /// Abstain when the corpus-distance confidence is below this.
    pub distance_threshold: f32,
    /// Slope of the state→option-centroid cosine term.
    pub route_scale: f32,
    /// Map options to domains by exact name (else only by index when #options == #domains).
    pub option_name_route: bool,
    pub gate_mid: f32,
    pub gate_scale: f32,
    pub cal_capacity: usize,
    pub cal_min_obs: usize,
    pub tokenizer: Tokenizer,
    pub dim: usize,
    /// Weight features by IDF fitted from the corpus documents (not upstream).
    pub idf: bool,
    /// Distance gate reads the state alone instead of state + prompt + criteria (not upstream).
    pub gate_on_state: bool,
}

impl Default for EngineConfig {
    /// Upstream's shipped defaults (the demo engine).
    fn default() -> Self {
        Self {
            score_temperature: 24.0,
            score_threshold: 0.35,
            distance_threshold: 0.5,
            route_scale: 8.0,
            option_name_route: true,
            gate_mid: GATE_MID,
            gate_scale: GATE_SCALE,
            cal_capacity: 512,
            cal_min_obs: 64,
            tokenizer: Tokenizer::Ascii,
            dim: crate::embed::EMBED_DIM,
            idf: false,
            gate_on_state: false,
        }
    }
}

impl EngineConfig {
    /// Upstream's posture when booting on a user corpus: the score axis is open (a first
    /// corpus is too thin to calibrate the readout), only the distance gate abstains.
    pub fn corpus_boot() -> Self {
        Self {
            score_threshold: 0.0,
            ..Self::default()
        }
    }
}

struct Domain {
    name: String,
    drafter: Drafter,
    gate: DistanceGate,
    direction: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstainCause {
    Answered,
    /// Confidence under the score threshold.
    ScoreGate,
    /// Confident enough, but far from the routed domain's documents.
    DistanceGate,
}

/// One question's full internal result (the wire keeps a subset).
#[derive(Clone, Debug)]
pub struct Verdict {
    pub pick: usize,
    /// Internal distribution (noul: `[yes, no]`).
    pub probs: Vec<f32>,
    pub raw_confidence: f32,
    pub confidence: f32,
    pub distance_confidence: f32,
    /// Max cosine to the routed domain's nearest document (what the distance gate read).
    pub max_similarity: f32,
    pub cause: AbstainCause,
    pub domain: usize,
    /// No centroid term reached the options: the drafter delta alone ranked them.
    pub drafter_only: bool,
}

impl Verdict {
    pub fn abstained(&self) -> bool {
        self.cause != AbstainCause::Answered
    }
}

pub struct Engine {
    cfg: EngineConfig,
    embedder: Embedder,
    domains: Vec<Domain>,
    calibrator: Calibrator,
    calibrated: bool,
}

fn normalized_entropy(p: &[f32]) -> f32 {
    let k = p.len();
    if k <= 1 {
        return 0.0;
    }
    let h: f32 = p.iter().filter(|x| **x > 0.0).map(|x| -x * x.ln()).sum();
    (h / (k as f32).ln()).clamp(0.0, 1.0)
}

/// The raw confidence readout of one distribution.
pub fn readout(p: &[f32]) -> f32 {
    if p.len() <= NARROW_MAX_OPTIONS {
        1.0 - normalized_entropy(p)
    } else {
        p.iter().copied().fold(0.0, f32::max)
    }
}

impl Engine {
    pub fn build(corpus: Vec<DomainDocs>, cfg: EngineConfig) -> Result<Self, String> {
        if corpus.is_empty() {
            return Err("an engine needs at least one domain".into());
        }
        let mut embedder = Embedder::new(cfg.tokenizer, cfg.dim);
        if cfg.idf {
            let all: Vec<&str> = corpus.iter().flat_map(|d| d.docs.iter().map(String::as_str)).collect();
            embedder = embedder.with_idf(&all);
        }
        let mut domains = Vec::with_capacity(corpus.len());
        for d in corpus {
            if d.docs.is_empty() {
                return Err(format!("domain `{}` has no documents", d.name));
            }
            let rows: Vec<Vec<f32>> = d.docs.iter().map(|doc| embedder.embed(doc)).collect();
            domains.push(Domain {
                drafter: Drafter::new(d.docs.join("\n").into_bytes()),
                gate: DistanceGate::new(&rows, cfg.gate_mid, cfg.gate_scale),
                direction: unit_centroid(&rows, cfg.dim),
                name: d.name,
            });
        }
        Ok(Self {
            calibrator: Calibrator::new(cfg.cal_capacity, cfg.cal_min_obs),
            calibrated: false,
            cfg,
            embedder,
            domains,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn domain_names(&self) -> Vec<&str> {
        self.domains.iter().map(|d| d.name.as_str()).collect()
    }

    /// Answer every question of the request (or fail before any work on an invalid one).
    pub fn solve(&mut self, req: &DecisionRequest) -> Result<Vec<Verdict>, WireError> {
        req.validate()?;
        let n = self.domains.len();
        // The option-ranking cosine reads the STATE alone: prompt and criteria repeat on every
        // case and would tilt every cosine toward whichever centroid shares their words.
        let state_q = self.embedder.embed(&req.state);
        let route_terms: Vec<f32> = self
            .domains
            .iter()
            .map(|d| sigmoid(dot(&state_q, &d.direction) * self.cfg.route_scale))
            .collect();
        let mut out = Vec::with_capacity(req.questions.len());
        for q in &req.questions {
            let mut ctx = req.state.clone().into_bytes();
            ctx.push(b'\n');
            ctx.extend_from_slice(q.prompt.as_bytes());
            if let Some(c) = &q.criteria {
                ctx.push(b'\n');
                ctx.extend_from_slice(c.as_bytes());
            }
            let ctx_q = self.embedder.embed(std::str::from_utf8(&ctx).expect("utf-8 in, utf-8 out"));

            // Route: argmax over unit centroids, ties to the lowest index.
            let mut di = 0;
            let mut best_dot = f32::NEG_INFINITY;
            for (i, d) in self.domains.iter().enumerate() {
                let s = dot(&ctx_q, &d.direction);
                if s > best_dot {
                    best_dot = s;
                    di = i;
                }
            }

            let noul = q.kind == Kind::Noul;
            let k = if noul { 2 } else { q.options.len() };
            // Option → domain: by name when EVERY option names a domain, else by index when
            // there are exactly as many options as domains. Never for noul ([yes, no] is not a
            // label list).
            let by_name: Option<Vec<usize>> = (self.cfg.option_name_route && !noul && k <= n)
                .then(|| {
                    q.options
                        .iter()
                        .map(|o| self.domains.iter().position(|d| d.name == *o))
                        .collect::<Option<Vec<_>>>()
                })
                .flatten();
            let opt_dom: Option<Vec<usize>> = if noul {
                None
            } else {
                by_name.or_else(|| (k == n).then(|| (0..n).collect()))
            };

            let dom = &mut self.domains[di];
            let base = dom.drafter.baseline(&ctx);
            let mut scores = Vec::with_capacity(k);
            for i in 0..k {
                let cand: &[u8] = if noul {
                    if i == 0 { b"yes" } else { b"no" }
                } else {
                    q.options[i].as_bytes()
                };
                let s = dom.drafter.delta(base, &ctx, cand);
                let mut score = sigmoid(s as f32 / self.cfg.score_temperature);
                if let Some(od) = &opt_dom {
                    score += route_terms[od[i]];
                }
                scores.push(score);
            }
            let sum: f32 = scores.iter().sum();
            let probs: Vec<f32> = if sum > 0.0 {
                scores.iter().map(|s| s / sum).collect()
            } else {
                vec![1.0 / k as f32; k]
            };
            let mut pick = 0;
            for i in 1..k {
                if probs[i] > probs[pick] {
                    pick = i;
                }
            }
            let raw_confidence = readout(&probs);
            let confidence = self.calibrator.apply(raw_confidence);
            let gate_q = if self.cfg.gate_on_state { &state_q } else { &ctx_q };
            let gate = &self.domains[di].gate;
            let (max_similarity, distance_confidence) = (gate.max_similarity(gate_q), gate.confidence(gate_q));
            let cause = if confidence < self.cfg.score_threshold {
                AbstainCause::ScoreGate
            } else if distance_confidence < self.cfg.distance_threshold {
                AbstainCause::DistanceGate
            } else {
                AbstainCause::Answered
            };
            out.push(Verdict {
                pick,
                probs,
                raw_confidence,
                confidence,
                distance_confidence,
                max_similarity,
                cause,
                domain: di,
                drafter_only: opt_dom.is_none() && !noul,
            });
        }
        Ok(out)
    }

    pub fn calibration(&self) -> Calibration {
        if self.calibrated {
            Calibration {
                method: "sigmoid-gate".into(),
                temperature: self.calibrator.temperature(),
            }
        } else {
            Calibration {
                method: "none".into(),
                temperature: 1.0,
            }
        }
    }

    pub fn to_response(&self, req: &DecisionRequest, verdicts: &[Verdict]) -> DecisionResponse {
        let answers = req
            .questions
            .iter()
            .zip(verdicts)
            .map(|(q, v)| Answer {
                question_id: q.id.clone(),
                outcome: (!v.abstained()).then_some(match q.kind {
                    Kind::Choice => Outcome::Choice { index: v.pick as u32 },
                    Kind::Score => Outcome::Score { level: v.pick as u32 },
                    Kind::Noul => Outcome::Noul { yes: v.pick == 0 },
                }),
                probabilities: if q.kind == Kind::Noul {
                    vec![v.probs[0]]
                } else {
                    v.probs.clone()
                },
                confidence: v.confidence,
            })
            .collect();
        let mut counts = vec![0usize; self.domains.len()];
        for v in verdicts {
            counts[v.domain] += 1;
        }
        let mut reason = String::from("modelless corpus routing:");
        for (d, c) in self.domains.iter().zip(&counts) {
            reason.push_str(&format!(" {}={c}", d.name));
        }
        reason.push_str("; fused abstain (score+distance) armed");
        let drafter_only = verdicts.iter().filter(|v| v.drafter_only).count();
        if drafter_only > 0 {
            reason.push_str(&format!("; drafter-only={drafter_only}"));
        }
        DecisionResponse {
            answers,
            routing: Routing {
                lane: "modelless".into(),
                reason,
            },
            calibration: self.calibration(),
        }
    }

    pub fn decide(&mut self, req: &DecisionRequest) -> Result<DecisionResponse, WireError> {
        let v = self.solve(req)?;
        Ok(self.to_response(req, &v))
    }

    /// Outcome feedback: the engine's own confidence for a past case and whether it was right.
    /// Returns whether a refit ran (≥ `cal_min_obs` observations).
    pub fn observe(&mut self, p: f32, correct: bool) -> bool {
        let refit = self.calibrator.observe(p, correct);
        self.calibrated |= refit && !self.calibrator.is_identity();
        refit
    }
}

/// Upstream's built-in two-domain demo corpus, verbatim.
pub fn demo_corpus() -> Vec<DomainDocs> {
    const OPS: &str = "Deploy the server to staging and verify the rollout before promoting \
to production. The staging cluster mirrors production capacity and runs the \
same release candidate. Rollback is one command when a deploy regresses the \
error budget. Verify the health endpoints after every rollout step.";
    const SUPPORT: &str = "The customer asked for a refund of the last invoice because the \
billing account was charged twice. Check the account balance and the payment \
history, then refund the duplicate charge to the original payment method. \
Escalate to the billing team when the invoice does not match the account \
records.";
    vec![
        DomainDocs {
            name: "ops".into(),
            docs: vec![OPS.into()],
        },
        DomainDocs {
            name: "support".into(),
            docs: vec![SUPPORT.into()],
        },
    ]
}
