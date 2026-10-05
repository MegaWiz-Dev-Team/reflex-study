"""Render metrics.json (from rubric.py) as a markdown report.

usage: python3 scripts/rubric_report.py MANIFEST.json METRICS.json OUT.md [NOTES.md]
NOTES.md, when given, is appended verbatim (the verdict and caveats a person writes).
"""
import json, sys

RUBRIC = [
    ("ความถูกต้อง", "accuracy", "pct ↑ · **primary**", "สัดส่วนที่ถูก (ไม่ตอบ = ผิด) · Wilson 95% CI", "evx_metric `accuracy`"),
    ("ความถูกต้อง", "macro_f1", "ratio ↑", "F1 เฉลี่ยทุก class เท่ากัน · bootstrap CI", "Reflex bench `macro_f1`"),
    ("ความถูกต้อง", "cohen_kappa", "ratio ↑", "ความตรงกับ gold หลังหัก chance", "Mimir HITL agreement"),
    ("ความถูกต้อง", "recall / precision (slice class)", "pct ↑", "ราย class + n ของ class", "evx `slice_dim=class`"),
    ("การ abstain", "coverage · selective_accuracy", "pct ↑", "สัดส่วนที่ตอบ · ความแม่นเฉพาะที่ตอบ", "Reflex bench `calibrated_abstain`"),
    ("Calibration", "ece_10bin · brier_top_label", "ratio ↓", "เฉพาะคำตอบที่มีค่า confidence (LLM/keyword ไม่มี)", "Reflex G1 / `readout_ece`"),
    ("ความปลอดภัย", "harmful_recall", "pct ↑", "recall ของกลุ่ม label ที่พลาดแล้วเสียหายที่สุด (manifest `harmful`)", "HealthBench-style safety"),
    ("ความปลอดภัย", "harmful_missed_as_benign", "pct ↓", "ข้อที่อยู่ในกลุ่ม harmful แต่ถูกจัดเป็นกลุ่มปกติ", "HealthBench-style safety"),
    ("ความเป็นธรรม", "benign_flagged_harmful", "pct ↓", "ข้อปกติที่ถูกจัดเข้ากลุ่ม harmful — กล่าวหาผิด", "—"),
    ("ผลต่อระบบปลายทาง", "stimulus_error_mean", "milli ↓", "L1 ระหว่าง vector ผลกระทบของ label ที่ทายกับ label จริง (manifest `stimulus`) · bootstrap CI", "manifest `stimulus`"),
    ("ความปลอดภัยของ ladder", "silent_error_rate", "pct ↓", "ตอบผิดโดยชั้นที่ไม่ใช่ชั้นบนสุด — ไม่มีใครเห็นอีก", "การทดลอง loop"),
    ("ต้นทุน", "llm_calls_per_100 · answer_share (slice rung)", "count ↓", "เรียก LLM กี่ครั้งต่อ 100 · สัดส่วนที่แต่ละชั้นตอบ", "Rethink rung model"),
    ("ต้นทุน", "expected_latency_ms · latency_p50_ms (slice rung)", "ms ↓", "lexical/memory วัดใน process; ชั้น encoder/LLM ใช้ค่าที่วัดแยก (manifest `latency_ms`)", "evx `p95_latency_ms`"),
    ("ความเสถียร", "deterministic", "bool", "รันซ้ำได้ผลเดิมทุกไบต์ (verified) หรือโดยโครงสร้าง", "Reflex G-determinism"),
    ("สุขภาพของ loop", "lexical_confidence_auc", "ratio ↑", "confidence ของชั้น µs ยังแยกข้อถูก/ผิดได้ไหม", "การทดลอง loop"),
    ("เทียบ baseline", "delta_accuracy_vs_baseline · mcnemar_p_vs_baseline", "pct ↑ · ratio ↓", "paired bootstrap CI · McNemar exact", "Mimir A/B / KATGPT proposal"),
]


def main():
    man = json.load(open(sys.argv[1]))
    rep = json.load(open(sys.argv[2]))
    notes = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""

    def g(s, name, dim="", val=""):
        return next((x for x in s["metrics"] if x["name"] == name and x["slice_dim"] == dim and x["slice_val"] == val), None)

    def fmt(x, digits=1, ci=True):
        if not x:
            return "–"
        v = f"{x['value']:.{digits}f}"
        if ci and x.get("ci_low") is not None:
            v += f" [{x['ci_low']:.{digits}f}–{x['ci_high']:.{digits}f}]"
        return v

    ds = man["dataset"]
    L = []
    L.append(f"# Evaluation — {man['experiment_id']}\n")
    L.append(f"family `{man['family']}` · dataset `{ds['name']}` v{ds['version']} · n = {rep[0]['n']} · baseline `{man['baseline']}` · labels {len(man['labels'])}\n")
    L.append("> " + " · ".join(f"**{k}**: {v}" for k, v in ds.get("spec", {}).items()) + "\n")
    L.append("## 1. Rubric\n")
    L.append("| หมวด | metric | หน่วย / ทิศทาง | วัดอะไร | ที่มา |\n|---|---|---|---|---|")
    for r in RUBRIC:
        L.append("| " + " | ".join(r) + " |")
    L.append("\nทุก metric บันทึกตามโครง `evx_metric` ของ Mimir: name · slice · value · unit · higher_is_better · is_primary · ci_low/ci_high · n\n")

    L.append("## 2. ผลรวม (95% CI ในวงเล็บ)\n")
    L.append("| ระบบ | training | accuracy % | macro-F1 | κ | harmful recall % | harmful→benign % | benign→harmful % | stimulus err | Δ vs baseline (pp) | McNemar p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(rep, key=lambda s: -g(s, "accuracy")["value"]):
        L.append(f"| `{s['id']}` | {s['training']} | {fmt(g(s,'accuracy'))} | {fmt(g(s,'macro_f1'),3)} | {fmt(g(s,'cohen_kappa'),3,False)} | {fmt(g(s,'harmful_recall'),1,False)} | {fmt(g(s,'harmful_missed_as_benign'),1,False)} | {fmt(g(s,'benign_flagged_harmful'),1,False)} | {fmt(g(s,'stimulus_error_mean'),1)} | {fmt(g(s,'delta_accuracy_vs_baseline'),1)} | {fmt(g(s,'mcnemar_p_vs_baseline'),4,False)} |")

    L.append("\n## 3. ต้นทุน · calibration · ความปลอดภัยของ ladder\n")
    L.append("| ระบบ | expected latency ms | gemma ต่อ 100 | silent error % | ตอบที่ lexical / memory / encoder / llm % | lexical conf AUC | ECE | Brier | deterministic |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(rep, key=lambda s: g(s, "expected_latency_ms")["value"] if g(s, "expected_latency_ms") else 1e9):
        shares = " / ".join(fmt(g(s, "answer_share", "rung", r), 0, False) for r in ["lexical", "memory", "encoder", "llm"]) if g(s, "silent_error_rate") else "–"
        L.append(f"| `{s['id']}` | {fmt(g(s,'expected_latency_ms'),1,False)} | {fmt(g(s,'llm_calls_per_100'),1,False)} | {fmt(g(s,'silent_error_rate'),1)} | {shares} | {fmt(g(s,'lexical_confidence_auc'),2,False)} | {fmt(g(s,'ece_10bin'),3,False)} | {fmt(g(s,'brier_top_label'),3,False)} | {s.get('deterministic')} |")

    L.append("\n## 4. Recall ราย class (%)\n")
    L.append("| ระบบ | " + " | ".join(man["labels"]) + " |")
    L.append("|---|" + "---|" * len(man["labels"]))
    ns = {l: g(rep[0], "recall", "class", l)["n"] for l in man["labels"]}
    L.append("| *n* | " + " | ".join(str(ns[l]) for l in man["labels"]) + " |")
    for s in sorted(rep, key=lambda s: -g(s, "accuracy")["value"]):
        L.append(f"| `{s['id']}` | " + " | ".join(fmt(g(s, "recall", "class", l), 0, False) for l in man["labels"]) + " |")

    if notes:
        L.append("\n" + notes)
    open(sys.argv[3], "w").write("\n".join(L) + "\n")
    print("wrote", sys.argv[3])


if __name__ == "__main__":
    main()
