"""Record the official reflex binary's /decide answers as a parity fixture.

Usage: python3 scripts/record_parity.py DEMO_URL CORPUS_URL VERSION_STRING > tests/fixtures/parity.jsonl
DEMO_URL serves the built-in demo corpus; CORPUS_URL serves corpora/first-corpus.
Each line: {"engine": "demo"|"first-corpus", "request": {...}, "response": {...}}.
"""
import json
import sys
import urllib.request

DEMO, CORPUS, VERSION = sys.argv[1], sys.argv[2], sys.argv[3]

STATES = [
    "The staging deploy of release candidate 4.2 finished but the error budget is down to 12 percent and two health endpoints are flapping after the rollout. Rollback is one command. Decide what happens next.",
    "our deploy regressed the error budget after the rollout — run the rollback and verify the health endpoints",
    "The customer was charged twice on the same invoice and wants the duplicate refunded to the card.",
    "A new user cannot find the workspace name step during signup and never got the verification email.",
    "my sourdough starter stopped rising after i moved it to a colder kitchen",
    "Drain one node at a time during the maintenance window, then bring it back.",
    "INVOICE TOTAL DOES NOT MATCH THE ACCOUNT RECORDS!!! Escalate?",
    "",
    "ลูกค้าโดนตัดบัตรซ้ำสองครั้งในใบแจ้งหนี้เดียวกัน ขอคืนเงินรายการที่ซ้ำ",
    "deploy ล้มเหลว error budget ลดลง ต้อง rollback ทันที",
    "The piece leaves no holes under it on the left edge, sits flat on the surface, and the stack stays low.",
]

def questions(names):
    qs = [
        {"id": "route_by_name", "kind": "choice", "prompt": "Which runbook applies?", "options": list(names)},
        {"id": "route_reversed", "kind": "choice", "prompt": "Which runbook applies?", "options": list(reversed(names))},
        {"id": "promote", "kind": "noul", "prompt": "Should the rollout be promoted to production right now?"},
        {"id": "severity", "kind": "score", "prompt": "How severe is this situation?", "options": ["routine", "needs-attention", "critical"]},
        {"id": "team", "kind": "choice", "prompt": "Route this ticket to the team that owns it.", "options": ["deploy-ops", "billing-support"],
         "criteria": "Ownership is decided by which team's runbook the situation matches."},
        {"id": "action", "kind": "choice", "prompt": "What should happen next?", "options": ["rollback", "refund", "resend email", "wait", "escalate to billing", "promote", "ignore", "page on-call", "close ticket"]},
    ]
    return qs

def post(url, body):
    req = urllib.request.Request(url + "/decide", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

out = []
for engine, url, names in [("demo", DEMO, ["ops", "support"]), ("first-corpus", CORPUS, ["billing", "deploy", "onboarding"])]:
    for s in STATES:
        # one request per question (single-question shape) and one multi-question request
        for q in questions(names):
            body = {"state": s, "questions": [q]}
            out.append({"engine": engine, "request": body, "response": post(url, body)})
        body = {"state": s, "questions": questions(names)}
        out.append({"engine": engine, "request": body, "response": post(url, body)})

print(json.dumps({"recorded_from": VERSION, "n": len(out)}, ensure_ascii=False))
for o in out:
    print(json.dumps(o, ensure_ascii=False))
