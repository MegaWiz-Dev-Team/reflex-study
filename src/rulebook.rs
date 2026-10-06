//! A labelling **rulebook** kept as data: one JSON file that every party reads — the people who
//! label, the generator that writes synthetic rows, and the llm rung's prompt (`ladder train
//! --guide`). The shape follows katopz's tetris rulebook in katgpt-rs
//! (`crates/katgpt-tetris/src/rulebook.rs`, MIT): each rule is a row with an id, the clause it came
//! from, an optional precondition and an optional KG triple, and the whole book has a BLAKE3 id
//! that changes whenever anything in it changes.
//!
//! ```json
//! {
//!   "name": "frontdesk", "version": "1.0.0",
//!   "labels": [{"id": "billing", "th": "การเงิน", "definition": "…",
//!               "positive": ["…"], "near_miss": [{"text": "…", "is": "records", "why": "…"}]}],
//!   "precedence": {"order": ["records", "billing", "appointment"], "why": "…"},
//!   "rules": [{"id": "R2", "rule": "…", "when": ["…"], "triple": ["…", "is", "…"], "source": "…"}],
//!   "decisions": [{"id": "Q1", "ask": "…", "decision": "…", "by": "…", "why": "…"}]
//! }
//! ```
//!
//! The id is BLAKE3 over the canonical JSON (keys sorted, no whitespace, non-ASCII kept), the same
//! bytes Python's `json.dumps(rb, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`
//! gives, so tools in either language agree on it.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize)]
pub struct NearMiss {
    pub text: String,
    /// The label this text actually gets.
    pub is: String,
    pub why: String,
}

#[derive(Debug, Deserialize)]
pub struct Label {
    pub id: String,
    /// A display name (any language); shown next to the id in the guide.
    #[serde(default)]
    pub th: Option<String>,
    pub definition: String,
    #[serde(default)]
    pub positive: Vec<String>,
    #[serde(default)]
    pub near_miss: Vec<NearMiss>,
}

#[derive(Debug, Deserialize)]
pub struct Precedence {
    pub order: Vec<String>,
    #[serde(default)]
    pub why: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Rule {
    pub id: String,
    pub rule: String,
    /// Preconditions under which the rule applies (katgpt-rs: the physics a rule needs).
    /// `null` or absent = applies everywhere.
    #[serde(default, alias = "phase", deserialize_with = "null_as_empty")]
    pub when: Vec<String>,
}

fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    Ok(Option::<Vec<String>>::deserialize(d)?.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
struct Book {
    #[serde(default)]
    name: Option<String>,
    version: String,
    labels: Vec<Label>,
    #[serde(default)]
    precedence: Option<Precedence>,
    #[serde(default)]
    rules: Vec<Rule>,
}

#[derive(Debug)]
pub struct Rulebook {
    pub name: String,
    pub version: String,
    pub labels: Vec<Label>,
    pub precedence: Option<Precedence>,
    pub rules: Vec<Rule>,
    id: String,
}

impl Rulebook {
    /// Parse a rulebook and compute its id (the full JSON is hashed, typed fields or not).
    pub fn parse(json: &str) -> Result<Self, String> {
        let raw: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("rulebook: {e}"))?;
        let canonical = serde_json::to_string(&raw).map_err(|e| format!("rulebook: {e}"))?;
        let book: Book = serde_json::from_value(raw).map_err(|e| format!("rulebook: {e}"))?;
        Ok(Rulebook {
            name: book.name.unwrap_or_else(|| "rulebook".into()),
            version: book.version,
            labels: book.labels,
            precedence: book.precedence,
            rules: book.rules,
            id: blake3::hash(canonical.as_bytes()).to_hex().to_string(),
        })
    }

    pub fn load(path: &str) -> Result<Self, String> {
        Self::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)
    }

    /// BLAKE3 hex of the canonical JSON.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Label ids, sorted — the order the ladder lists options in.
    pub fn label_ids(&self) -> Vec<String> {
        self.labels.iter().map(|l| l.id.clone()).collect::<BTreeSet<_>>().into_iter().collect()
    }

    /// Everything that would make the book ambiguous or unusable; empty when it is fine.
    pub fn problems(&self) -> Vec<String> {
        let mut p = Vec::new();
        let mut seen = BTreeSet::new();
        if self.labels.len() < 2 {
            p.push("needs at least two labels".into());
        }
        for l in &self.labels {
            if !seen.insert(l.id.as_str()) {
                p.push(format!("label {} is defined twice", l.id));
            }
            if l.definition.trim().is_empty() {
                p.push(format!("label {} has no definition", l.id));
            }
        }
        for l in &self.labels {
            for n in &l.near_miss {
                if !seen.contains(n.is.as_str()) {
                    p.push(format!("label {}: near miss points to unknown label {}", l.id, n.is));
                } else if n.is == l.id {
                    p.push(format!("label {}: a near miss must belong to another label", l.id));
                }
            }
        }
        if let Some(pr) = &self.precedence {
            let mut once = BTreeSet::new();
            for id in &pr.order {
                if !seen.contains(id.as_str()) {
                    p.push(format!("precedence names unknown label {id}"));
                }
                if !once.insert(id.as_str()) {
                    p.push(format!("precedence lists {id} twice"));
                }
            }
            for l in &self.labels {
                if !once.contains(l.id.as_str()) {
                    p.push(format!("precedence leaves out label {}", l.id));
                }
            }
        }
        let mut rules = BTreeSet::new();
        for r in &self.rules {
            if !rules.insert(r.id.as_str()) {
                p.push(format!("rule {} is defined twice", r.id));
            }
        }
        p
    }

    /// The guide text for the llm rung (`ladder train --guide`): every label with its first
    /// example and near miss, the precedence, and every rule with its precondition.
    pub fn guide(&self) -> String {
        let by: BTreeMap<&str, &Label> = self.labels.iter().map(|l| (l.id.as_str(), l)).collect();
        let mut out = vec![
            format!("Labelling rulebook {} {} (id {}).", self.name, self.version, &self.id[..16]),
            "Labels (exactly one per text):".to_string(),
        ];
        for id in self.label_ids() {
            let l = by[id.as_str()];
            match &l.th {
                Some(th) => out.push(format!("- {} ({th}): {}", l.id, l.definition)),
                None => out.push(format!("- {}: {}", l.id, l.definition)),
            }
            if let Some(e) = l.positive.first() {
                out.push(format!("    e.g. {e}"));
            }
            if let Some(n) = l.near_miss.first() {
                out.push(format!("    NOT this label: \"{}\" → {} ({})", n.text, n.is, n.why));
            }
        }
        if let Some(pr) = &self.precedence {
            out.push(format!("Precedence — if several labels apply, choose the earliest: {}", pr.order.join(" > ")));
        }
        if !self.rules.is_empty() {
            out.push("Rules:".into());
            for r in &self.rules {
                if r.when.is_empty() {
                    out.push(format!("- {}: {}", r.id, r.rule));
                } else {
                    out.push(format!("- {} [when: {}]: {}", r.id, r.when.join(", "), r.rule));
                }
            }
        }
        out.join("\n") + "\n"
    }

    /// One-line descriptions for the ladder's options (`ladder train --descriptions`): the
    /// definition, plus the label's place in the precedence when there is one.
    pub fn descriptions(&self) -> BTreeMap<String, String> {
        let order = self.precedence.as_ref().map(|p| &p.order);
        self.labels
            .iter()
            .map(|l| {
                let d = match order.and_then(|o| o.iter().position(|x| *x == l.id).map(|k| (k, o.len()))) {
                    Some((k, n)) => format!("{} (priority {} of {n} when several apply)", l.definition, k + 1),
                    None => l.definition.clone(),
                };
                (l.id.clone(), d)
            })
            .collect()
    }
}
