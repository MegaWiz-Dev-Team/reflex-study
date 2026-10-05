//! The HTTP wire, field for field with upstream's `/decide` contract.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Multiple choice over `options` (≥ 2).
    Choice,
    /// Ordinal rubric; `options` are the levels, lowest first.
    Score,
    /// Yes/no; no `options`.
    Noul,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    pub kind: Kind,
    pub prompt: String,
    #[serde(default)]
    pub options: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub state: String,
    pub questions: Vec<Question>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Choice { index: u32 },
    Score { level: u32 },
    Noul { yes: bool },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    pub question_id: String,
    /// `None` = abstained (serialized as `null`).
    pub outcome: Option<Outcome>,
    /// One per option in option order; a `noul` question carries exactly one, p(yes).
    pub probabilities: Vec<f32>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Routing {
    pub lane: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub method: String,
    pub temperature: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub answers: Vec<Answer>,
    pub routing: Routing,
    pub calibration: Calibration,
}

#[derive(Clone, Debug, PartialEq)]
pub enum WireError {
    EmptyPrompt { at: usize },
    NoulCarriesOptions { at: usize },
    TooFewOptions { at: usize, n: usize },
    DuplicateQuestionId { at: usize },
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt { at } => write!(f, "question {at} has an empty prompt"),
            Self::NoulCarriesOptions { at } => write!(f, "noul question {at} must not carry options"),
            Self::TooFewOptions { at, n } => {
                write!(f, "choice/score question {at} needs ≥2 options, got {n}")
            }
            Self::DuplicateQuestionId { at } => write!(f, "duplicate question id at position {at}"),
        }
    }
}

impl DecisionRequest {
    /// Per-question rules, then id uniqueness. An empty `questions` list is legal.
    pub fn validate(&self) -> Result<(), WireError> {
        for (at, q) in self.questions.iter().enumerate() {
            if q.prompt.trim().is_empty() {
                return Err(WireError::EmptyPrompt { at });
            }
            match q.kind {
                Kind::Noul if !q.options.is_empty() => {
                    return Err(WireError::NoulCarriesOptions { at });
                }
                Kind::Choice | Kind::Score if q.options.len() < 2 => {
                    return Err(WireError::TooFewOptions { at, n: q.options.len() });
                }
                _ => {}
            }
        }
        for (at, q) in self.questions.iter().enumerate() {
            if self.questions[..at].iter().any(|o| o.id == q.id) {
                return Err(WireError::DuplicateQuestionId { at });
            }
        }
        Ok(())
    }
}
