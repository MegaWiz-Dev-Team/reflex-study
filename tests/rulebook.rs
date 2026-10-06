//! The rulebook as data: its id must match the Python canonical-JSON hash, `problems` must catch an
//! ambiguous book, and the guide must carry every label, the precedence and every rule.

use reflex_study::rulebook::Rulebook;

const EXAMPLE: &str = include_str!("../examples/frontdesk-rulebook/rulebook.json");

#[test]
fn example_id_matches_python_canonical_json() {
    // python: blake3(json.dumps(rb, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    let rb = Rulebook::parse(EXAMPLE).unwrap();
    assert_eq!(rb.id(), "3e1e4dab49bfe225d0ccd238c01fcb13fa9244a207fe7342237b71f438763097");
    assert!(rb.problems().is_empty(), "{:?}", rb.problems());
}

#[test]
fn id_changes_with_any_change_and_ignores_layout() {
    let rb = Rulebook::parse(EXAMPLE).unwrap();
    let reflowed: serde_json::Value = serde_json::from_str(EXAMPLE).unwrap();
    let compact = serde_json::to_string(&reflowed).unwrap();
    assert_eq!(Rulebook::parse(&compact).unwrap().id(), rb.id(), "whitespace and key order are not content");
    let edited = EXAMPLE.replace("records > billing > appointment\"", "records > appointment > billing\"");
    assert_ne!(edited, EXAMPLE);
    assert_ne!(Rulebook::parse(&edited).unwrap().id(), rb.id(), "a changed rule must change the id");
}

#[test]
fn guide_carries_labels_precedence_and_rules() {
    let rb = Rulebook::parse(EXAMPLE).unwrap();
    let g = rb.guide();
    assert!(g.starts_with(&format!("Labelling rulebook frontdesk 1.0.0 (id {}).", &rb.id()[..16])));
    for l in ["appointment", "billing", "records"] {
        assert!(g.contains(&format!("- {l} (")), "label {l} missing");
    }
    assert!(g.contains("choose the earliest: records > billing > appointment"));
    for r in ["R1", "R2", "R3", "R4"] {
        assert!(g.contains(&format!("- {r}: ")), "rule {r} missing");
    }
    assert!(g.contains("- R5 [when: "), "a precondition is shown with its rule");
    assert_eq!(g, rb.guide(), "rendering is deterministic");
    let d = rb.descriptions();
    assert!(d["records"].ends_with("(priority 1 of 3 when several apply)"));
    assert!(d["appointment"].ends_with("(priority 3 of 3 when several apply)"));
}

#[test]
fn problems_catch_an_ambiguous_book() {
    let bad = r#"{"version": "0.1", "labels": [
        {"id": "a", "definition": "x", "near_miss": [{"text": "t", "is": "a", "why": "w"}]},
        {"id": "b", "definition": "y", "near_miss": [{"text": "t", "is": "zzz", "why": "w"}]},
        {"id": "b", "definition": "z"}],
      "precedence": {"order": ["a", "c", "a"]},
      "rules": [{"id": "R1", "rule": "r"}, {"id": "R1", "rule": "s", "phase": null}]}"#;
    let p = Rulebook::parse(bad).unwrap().problems();
    let has = |s: &str| p.iter().any(|x| x.contains(s));
    assert!(has("label b is defined twice"), "{p:?}");
    assert!(has("must belong to another label"), "{p:?}");
    assert!(has("unknown label zzz"), "{p:?}");
    assert!(has("precedence names unknown label c"), "{p:?}");
    assert!(has("precedence lists a twice"), "{p:?}");
    assert!(has("precedence leaves out label b"), "{p:?}");
    assert!(has("rule R1 is defined twice"), "{p:?}");
}
