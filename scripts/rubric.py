"""Score classifier / decision-ladder runs on one rubric, in Mimir's evx shape.

usage: python3 scripts/rubric.py MANIFEST.json OUT_DIR

MANIFEST: {"experiment_id", "family", "dataset": {...evx dataset...}, "labels": [...],
           "harmful": [...labels whose miss matters most...], "stimulus": {label: [ints]} | null,
           "rung_latency_ms": {"encoder": x, "llm": y} (measured outside; lexical/memory come
           from the answer files), "baseline": "<system id>", "systems": [
             {"id", "name", "kind", "runtime", "model_id", "training", "file", "latency_ms" | null,
              "deterministic": "verified" | "by construction" | "not verified", "group"}]}
Answer files: JSONL in one shared row order, {"gold", "label", "confidence"?, "rung"?, "steps"?}
(`steps` = [[rung, passed, micros], ...] as written by examples/dump_answers.rs).

Writes OUT_DIR/evx_runs.json (one POST body per system for /api/v1/eval/evx/runs) and
OUT_DIR/metrics.json (everything, for the report). Every proportion carries a Wilson 95% CI
and its n; F1, κ-free differences and costs carry a paired bootstrap CI (2000 resamples, seed 7).
"""
import json, math, os, random, sys
from collections import Counter

Z = 1.959964


def wilson(k, n):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h))


def macro_f1(rows, labels):
    f = []
    for l in labels:
        tp = sum(r["label"] == l and r["gold"] == l for r in rows)
        fp = sum(r["label"] == l and r["gold"] != l for r in rows)
        fn = sum(r["label"] != l and r["gold"] == l for r in rows)
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f) / len(f)


def kappa(rows, labels):
    n = len(rows)
    po = sum(r["label"] == r["gold"] for r in rows) / n
    a, b = Counter(r["gold"] for r in rows), Counter(r["label"] for r in rows)
    pe = sum(a[l] * b[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def ece(pairs, bins=10):
    if not pairs:
        return None
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if (lo < c <= hi) or (b == 0 and c == 0)]
        if sel:
            tot += len(sel) / len(pairs) * abs(sum(c for c, _ in sel) / len(sel) - sum(ok for _, ok in sel) / len(sel))
    return tot


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def bootstrap(n, stat, b=2000, seed=7):
    rng = random.Random(seed)
    vals = sorted(stat([rng.randrange(n) for _ in range(n)]) for _ in range(b))
    return vals[int(0.025 * b)], vals[int(0.975 * b) - 1]


def mcnemar(a_ok, b_ok):
    b = sum(x and not y for x, y in zip(a_ok, b_ok))
    c = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def main():
    man = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    base = os.path.dirname(os.path.abspath(sys.argv[1]))
    labels, harmful = man["labels"], set(man["harmful"])
    stim = man.get("stimulus")
    lat = man.get("rung_latency_ms", {})
    load = lambda f: [json.loads(l) for l in open(os.path.join(base, f)) if l.strip()]
    data = {s["id"]: load(s["file"]) for s in man["systems"]}
    golds = [[r["gold"] for r in rows] for rows in data.values()]
    assert all(g == golds[0] for g in golds), "answer files must share one row order"
    gold = golds[0]
    n = len(gold)
    base_ok = [r["label"] == r["gold"] for r in data[man["baseline"]]]

    report, bodies = [], []
    for s in man["systems"]:
        rows = data[s["id"]]
        ok = [r["label"] == r["gold"] for r in rows]
        answered = [r for r in rows if r["label"] is not None]
        m = []

        def add(name, value, unit, hib=True, primary=False, ci=(None, None), nn=n, dim="", val=""):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return
            m.append({"name": name, "slice_dim": dim, "slice_val": val, "value": round(value, 6), "unit": unit,
                      "higher_is_better": hib, "is_primary": primary,
                      "ci_low": None if ci[0] is None else round(ci[0], 6), "ci_high": None if ci[1] is None else round(ci[1], 6), "n": nn})

        # effectiveness
        p, lo, hi = wilson(sum(ok), n)
        add("accuracy", p, "pct", primary=True, ci=(lo, hi))
        add("macro_f1", macro_f1(rows, labels), "ratio", ci=bootstrap(n, lambda ix: macro_f1([rows[i] for i in ix], labels)))
        add("cohen_kappa", kappa(rows, labels), "ratio")
        for l in labels:
            g = [r for r in rows if r["gold"] == l]
            pr = [r for r in rows if r["label"] == l]
            k = sum(r["label"] == l for r in g)
            p, lo, hi = wilson(k, len(g))
            add("recall", p, "pct", ci=(lo, hi), nn=len(g), dim="class", val=l)
            if pr:
                p, lo, hi = wilson(sum(r["gold"] == l for r in pr), len(pr))
                add("precision", p, "pct", ci=(lo, hi), nn=len(pr), dim="class", val=l)
        # selective prediction
        p, lo, hi = wilson(len(answered), n)
        add("coverage", p, "pct", ci=(lo, hi))
        if answered:
            p, lo, hi = wilson(sum(r["label"] == r["gold"] for r in answered), len(answered))
            add("selective_accuracy", p, "pct", ci=(lo, hi), nn=len(answered))
        # calibration (answers that carry a confidence)
        conf = [(r["confidence"], r["label"] == r["gold"]) for r in answered if r.get("confidence") is not None]
        if conf:
            add("ece_10bin", ece(conf), "ratio", hib=False, nn=len(conf))
            add("brier_top_label", sum((c - o) ** 2 for c, o in conf) / len(conf), "ratio", hib=False, nn=len(conf))
        # safety / harm
        hg = [r for r in rows if r["gold"] in harmful]
        if hg:
            p, lo, hi = wilson(sum(r["label"] == r["gold"] for r in hg), len(hg))
            add("harmful_recall", p, "pct", ci=(lo, hi), nn=len(hg))
            p, lo, hi = wilson(sum(r["label"] is not None and r["label"] not in harmful for r in hg), len(hg))
            add("harmful_missed_as_benign", p, "pct", hib=False, ci=(lo, hi), nn=len(hg))
        bg = [r for r in rows if r["gold"] not in harmful]
        if bg:
            p, lo, hi = wilson(sum(r["label"] in harmful for r in bg), len(bg))
            add("benign_flagged_harmful", p, "pct", hib=False, ci=(lo, hi), nn=len(bg))
        if stim:
            cost = lambda r: 0 if r["label"] is None else sum(abs(a - b) for a, b in zip(stim[r["label"]], stim[r["gold"]]))
            add("stimulus_error_mean", sum(cost(r) for r in rows) / n, "milli_stimulus", hib=False,
                ci=bootstrap(n, lambda ix: sum(cost(rows[i]) for i in ix) / n))
        # ladder internals: rung shares, silent errors, llm calls, latency, lexical calibration
        latency = s.get("latency_ms")
        if any("rung" in r for r in rows):
            top = s.get("top_rung", "llm")  # the ladder's highest rung by construction, not the highest one reached
            for rung, c in sorted(Counter(r.get("rung") for r in rows if r.get("rung")).items()):
                add("answer_share", 100 * c / n, "pct", dim="rung", val=rung, nn=n)
            silent = sum(r.get("rung") not in (None, top) and r["label"] != r["gold"] for r in rows)
            p, lo, hi = wilson(silent, n)
            add("silent_error_rate", p, "pct", hib=False, ci=(lo, hi))
            add("llm_calls_per_100", 100 * sum(any(st[0] == "llm" for st in r.get("steps", [])) for r in rows) / n, "count", hib=False)
            lex_us = sorted(st[2] for r in rows for st in r.get("steps", []) if st[0] == "lexical")
            mem_us = [st[2] for r in rows for st in r.get("steps", []) if st[0] == "memory"]

            def per_row(r):
                rungs = {st[0] for st in r.get("steps", [])}
                t = lex_us[len(lex_us) // 2] / 1000 if lex_us else 0.0
                if rungs & {"memory", "encoder"}:
                    t += lat.get("encoder", 0.0)
                if "llm" in rungs:
                    t += lat.get("llm", 0.0)
                return t
            latency = sum(per_row(r) for r in rows) / n
            if lex_us:
                add("latency_p50_ms", lex_us[len(lex_us) // 2] / 1000, "ms", hib=False, dim="rung", val="lexical", nn=len(lex_us))
            if mem_us:
                add("latency_p50_ms", sorted(mem_us)[len(mem_us) // 2] / 1000, "ms", hib=False, dim="rung", val="memory_lookup", nn=len(mem_us))
            lx = [(r["confidence"], r["label"] == r["gold"]) for r in rows if r.get("rung") == "lexical" and r.get("confidence") is not None]
            a = auc([c for c, o in lx if o], [c for c, o in lx if not o])
            add("lexical_confidence_auc", a, "ratio", nn=len(lx))
        add("expected_latency_ms", latency, "ms", hib=False)
        add("deterministic", {"verified": 1.0, "by construction": 1.0}.get(s.get("deterministic"), None), "bool")
        # paired comparison with the baseline
        if s["id"] != man["baseline"]:
            b, c, pv = mcnemar(base_ok, ok)
            d = 100 * (sum(ok) - sum(base_ok)) / n
            add("delta_accuracy_vs_baseline", d, "pct",
                ci=bootstrap(n, lambda ix: 100 * (sum(ok[i] for i in ix) - sum(base_ok[i] for i in ix)) / n))
            add("mcnemar_p_vs_baseline", pv, "ratio", hib=False)
        names = [(x["name"], x["slice_dim"], x["slice_val"]) for x in m]
        assert len(names) == len(set(names)) and sum(x["is_primary"] for x in m) == 1
        report.append({**s, "metrics": m, "n": n})
        bodies.append({
            "family": man["family"], "run_id": f"{man['experiment_id']}:{s['id']}", "experiment_id": man["experiment_id"],
            "status": "COMPLETED", "n_items": n, "git_sha": man.get("git_sha"),
            "target": {"kind": s["kind"], "name": s["name"], "model_id": s.get("model_id"), "runtime": s.get("runtime"),
                       "config": {"training": s.get("training"), "group": s.get("group"), "deterministic": s.get("deterministic")}},
            "dataset": man["dataset"], "metrics": m,
        })
    os.makedirs(out_dir, exist_ok=True)
    json.dump(bodies, open(os.path.join(out_dir, "evx_runs.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(report, open(os.path.join(out_dir, "metrics.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(report)} systems · {sum(len(r['metrics']) for r in report)} metrics → {out_dir}")


if __name__ == "__main__":
    main()
