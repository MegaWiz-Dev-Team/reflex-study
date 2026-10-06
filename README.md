# reflex-study

> A study re-implementation of [Reflex](https://reflex.gist.rs/), katopz's modelless decision
> engine — bit-identical to the release binary on 264/264 answers — plus a Thai-aware tokenizer
> and a three-rung decision ladder (lexical → local encoder → local LLM) where each rung answers
> only what the rung below was unsure about.

ศึกษา [Reflex](https://reflex.gist.rs/) (modelless decision engine ของ gist-rs) แล้วเขียนใหม่ใน Rust
เพื่อดูว่าข้างในทำงานยังไงจริง และลองแก้จุดที่ใช้กับภาษาไทยไม่ได้

สถานะ: study / prototype — ยังไม่ใช่ส่วนหนึ่งของ Asgard

- เรื่องเต็ม: [แกะ Reflex ของ katopz จนตรงทุกบิต แล้วต่อเป็นบันไดตัดสินใจภาษาไทย](https://asgard.megawiz.co.th/blog/reflex-modelless-decision-ladder-th)
  และ [Ultra Instinct ของบันไดตัดสินใจ](https://asgard.megawiz.co.th/blog/ultra-instinct-decision-ladder-th)
- ชั้น lexical ของบันได (NBSVM, distillation, gate ของ `auto`) แยกเป็น crate:
  [ultra-instinct](https://github.com/MegaWiz-Dev-Team/ultra-instinct)

## Reflex คืออะไร

binary ตัวเดียว รัน HTTP บน `127.0.0.1:7331` รับ `state` (ข้อความสถานการณ์) + คำถามแบบมีชนิด
(`choice` เลือก 1 ตัวเลือก, `score` เลือกระดับ, `noul` ใช่/ไม่ใช่) แล้วตอบพร้อม `confidence`
หรือ **abstain** (`outcome: null`) ไม่มี neural network — "corpus คือ model":
ไฟล์ `.md` ต่อ domain คือความรู้ทั้งหมด

## ข้างในทำงานยังไง (อ่านจาก source: gist-rs/riir-reflex + katopz/katgpt-rs, MIT)

ต่อคำถาม 1 ข้อ:

1. **embed** — แยกคำด้วย ASCII whitespace, ตัดอักขระที่ไม่ใช่ ASCII a-z/0-9 ที่หัวท้ายคำ, hash คำ + คู่คำ (FNV-1a)
   ลง 256 ช่อง (sign trick) แล้ว L2-normalize
2. **route** — เลือก domain ที่ centroid ของเอกสารใกล้ embedding ของ `state + prompt` ที่สุด
3. **score ต่อตัวเลือก** = `sigmoid(Δ/24)` + `sigmoid(8·cos(state, centroid ของ domain ที่ชื่อตรงกับตัวเลือก))`
   - Δ = ขนาด LZ4 ของ `corpus + context` ลบ ขนาดของ `corpus + context + ตัวเลือก` (compression-as-model)
   - เทอม cosine ใช้เฉพาะเมื่อชื่อตัวเลือกตรงกับชื่อ domain ทุกตัว (หรือจำนวนตัวเลือก = จำนวน domain)
4. **normalize** — หารด้วยผลรวม (ไม่ใช้ softmax)
5. **confidence** — `1 − H/ln K` (K ≤ 8) หรือ max prob (K > 8) ผ่าน Platt calibrator (identity จนกว่าจะได้ feedback ≥ 64 ครั้ง)
6. **abstain** — confidence < 0.35 **หรือ** `sigmoid(8·(max cos กับเอกสารที่ใกล้สุด − 0.35)) < 0.5`
   (เมื่อ boot ด้วย corpus ของเราเอง upstream ปิดเกณฑ์ confidence เหลือแค่ distance gate)

## ข้อสังเกตจากการศึกษา

- **เร็วจริง** — วัดใน process ได้ p50 ราว 10 µs (ASCII) ถึง 26 µs (Thai, 1024 ช่อง) ต่อการตัดสินใจ (Mac mini, corpus 9 เอกสาร) และ deterministic จริง
- **LZ4 drafter แทบไม่ได้ช่วยจัดอันดับ** — upstream เขียนไว้เองว่าตัวเลือกสั้นๆ แทบไม่ขยับขนาด compressed
  ของ context ~300 byte ตัวที่จัดอันดับจริงคือ cosine กับ centroid (คือ nearest-centroid classifier นั่นเอง)
  ถ้าตัวเลือกไม่ใช่ชื่อ domain จะเหลือแต่ drafter → แทบเดาสุ่ม (`routing.reason` บอกเป็น `drafter-only=N`)
- **ค่า confidence เล็กมาก** (0.0002–0.02) เพราะความน่าจะเป็นเกือบเท่ากันทุกตัวเลือก
  ค่านี้ใช้ "เรียงลำดับ" ได้ แต่ตัวเลขดิบไม่มีความหมายเชิงความน่าจะเป็นจนกว่าจะ calibrate ด้วย feedback
- **ตัวเลข benchmark บนเว็บ** (เช่น ag_news 0.88) มาจากส่วนเสริมที่ fit จาก labeled data
  (naive-Bayes count tables, ridge, per-label heads) ซึ่งปิดไว้ตอน serve ตัวเลขของ lane ดิบใน skill file
  คือ ag_news 0.51, sst5 0.22 — ต้องวัดกับงานของเราเอง
- **ภาษาไทยใช้ไม่ได้เลย** — tokenizer ตัดไบต์ที่ไม่ใช่ ASCII ทิ้ง ข้อความไทยล้วนจึงกลายเป็น zero vector
  → abstain ทุกครั้ง หรือถ้าบังคับเลือกก็เท่ากับสุ่ม (upstream pin พฤติกรรมนี้ไว้เป็น contract ใน `tests/thai_posture_pins.rs`
  และบันทึก thai_wisesight 0.12 ต่ำกว่า chance 0.25) ผลข้างเคียงที่แปลก: ข้อความไทยที่มีคำอังกฤษคำเดียว
  เช่น "SMS" จะ embed เป็นคำนั้นคำเดียว แล้วได้ cosine 1.0 กับเอกสารที่มี "SMS" เป็นคำอังกฤษคำเดียวเหมือนกัน
- โครงการมีผู้ดูแลคนเดียว (530 commits) เวอร์ชัน 0.2.x build จาก source ต้องมี sibling repos
  (katgpt-rs, riir-infer, riir-reflexer) วางข้างกัน — binary release มี SHA256SUMS ให้ตรวจ

## ตระกูลเดียวกัน: Instinct และ Rethink (ดูเมื่อ 4 ต.ค. 2569)

gist.rs วาง "บันได" ไว้ แต่ละขั้นตอบเฉพาะสิ่งที่ขั้นล่าง abstain:

| ขั้น | ชื่อ | คืออะไร | สถานะ |
|---|---|---|---|
| R0 | Reflex | modelless (repo นี้) | ใช้ได้, MIT |
| R1 | Instinct (`gist-rs/riir-instinct`) | specialist ต่อ domain: one-vs-all logistic / NBSVM ridge บน hashed bag ชุดเดียวกับ Reflex, train จาก labeled rows, i8-quantized, BLAKE3-sealed; hybrid = cascade (Reflex ก่อน แล้ว specialist) หรือ prior fusion | โค้ด + CPU trainer เปิด MIT; README ใน repo ยังเก่า |
| R3 | Rethink (rethink.gist.rs) | encoder head ที่ train แล้ว รันบน GPU ของเขา รับเฉพาะคำถามที่ชั้นล่าง abstain | ยังไม่เปิดให้บริการ ("record-only"); weights ไม่แจก (hosted-only reader); คิดเงินเป็น token KAT/TUNA; v2 จะให้ "เช่า" specialist/encoder ตาม stake |

- Instinct ใช้ tokenizer ASCII ตัวเดียวกับ Reflex ("the ONE tokenizer law" ข้าม repo) — ข้อความไทยจึงไม่มี feature เช่นกัน
- Rethink ส่งข้อความคำถามออกไปเครื่องของผู้ให้บริการ — ใช้กับข้อมูลผู้ป่วยไม่ได้; ทดแทนในบ้านได้ด้วย encoder ที่รันเอง (เช่น bge-m3 ผ่าน Heimdall) + head ที่ train เอง

## บันไดของเราเอง (`ladder`)

แต่ละชั้นตอบเฉพาะสิ่งที่ชั้นล่างไม่มั่นใจ ทุกคำตอบบอกว่าชั้นไหนตอบ พร้อม receipt (BLAKE3 ของ ladder file / input / decision)

| ชั้น | อะไร | ต้นทุน |
|---|---|---|
| `lexical` | one-vs-all logistic บน n-gram ไทย (`features.rs` + crate ultra-instinct) train จาก label — แบบ Instinct แต่เห็นภาษาไทย | µs, CPU, ไม่มีโมเดล |
| `encoder` | head เดียวกันบน bge-m3 ผ่าน Heimdall ในเครื่อง — แทน Rethink โดยข้อมูลไม่ออกนอกเครื่อง | ~16–30 ms |
| `llm` | gemma-4-26b ผ่าน Heimdall เลือก label หรือ UNSURE | ~0.5 s |

- threshold ของแต่ละชั้น **fit จากข้อมูล**: out-of-fold 4 fold บนแถวที่ใช้ train, จุดตัดต่ำสุดที่ความแม่นยำ ≥ เป้า (default 0.9); น้อยกว่า 16 แถว หรือไม่ถึงเป้า = ชั้นนั้นไม่ตอบ; encoder fit บนแถวที่ lexical ส่งต่อ
- **novelty gate** ทุกชั้นที่ train: classifier รู้จักแค่ label ของมัน จะจัดเรื่องร้านกาแฟเข้า "billing" อย่างมั่นใจ — gate วัด cosine กับแถว train ที่ใกล้สุด เทียบกับ quantile ρ=0.05 ของค่าเดียวกันแบบ out-of-fold (ไม่ต้องมีตัวอย่างนอกเรื่อง)
- **ห้ามเรียก LLM ด้วยชื่ออื่นนอกจาก `gemma-4-26b`**: Heimdall เทียบชื่อโมเดลแบบ string ตรงตัว ชื่ออื่น (แม้ weights เดียวกัน) = hot-swap รีสตาร์ต mlx ของทุก service — client ปฏิเสธเองก่อนส่ง
- frontdesk (ไฟล์ใน repo): ชั้น lexical อย่างเดียว ถูก 15/15 ใน holdout และ abstain ข้อความนอกเรื่อง 6/6
- **ชั้น lexical แบบ Instinct** (`--lexical`, ตามคำแนะนำของ อ.katopz): `nbsvm` = presence × NB log-count ratio ราย class (สูตร Wang & Manning 2012) · `nbsvm-distill:<mix>` = soft target `mix·onehot + (1−mix)·teacher` จาก encoder head · `auto` ("Ultra Instinct") = ลองทุกแบบ out-of-fold แล้วเปลี่ยนจาก bag ก็ต่อเมื่อ paired bootstrap LB95 ของ (ตัวท้าชิงที่ดีสุด − bag) > 0 ไม่อย่างนั้นคง bag — ladder แบบ bag ได้ไฟล์ byte-identical เหมือนเดิม
- **teacher ของ distill ต้อง nested ภายใน fold**: ถ้าทำ teacher OOF ครั้งเดียวทั้งชุด student จะเห็น label ของ fold ที่กำลังถูกวัดผ่าน teacher และ OOF สูงเกินจริง (เจอตอน review 5 ต.ค. 2569 — ตอนนี้ teacher ถูกทำใหม่จากแถว train ของแต่ละ fold ใน [ultra-instinct](https://github.com/MegaWiz-Dev-Team/ultra-instinct) ซึ่งมี test ที่พิสูจน์ความต่างไว้)

```sh
B=target/release/ladder
$B train --task frontdesk --train rows.jsonl --out frontdesk.ladder.json [--descriptions labels.json] [--llm gemma-4-26b] [--no-encoder] [--lexical bag|nbsvm|nbsvm-distill[:mix]|auto]
$B eval  --ladder frontdesk.ladder.json --cases holdout.jsonl
$B serve --ladder frontdesk.ladder.json --bind 127.0.0.1:7342
curl -s -X POST http://127.0.0.1:7342/v1/classify -d '{"task":"frontdesk","text":"ขอใบเสร็จค่ารักษาใหม่ ฉบับเดิมหาย"}'
```

## บันไดเลื่อน (escalator, v2)

บันไดรุ่นที่ "เลื่อนเอง": แต่ละชั้นตอบก็ต่อเมื่อชุดคำตอบแบบ **conformal** เหลือ label เดียว ไม่อย่างนั้นส่งขึ้นชั้นถัดไป
และชั้นบนสุดถ้ายังไม่แน่ใจ จะตอบ label ที่น่าจะเป็นที่สุด (โหมดฝึก) หรือส่งชุดคำตอบให้คนตัดสิน (`rung = "review"`, โหมดสอบ)

```sh
$B calibrate --ladder frontdesk.ladder.json --verified verified.jsonl --out frontdesk.escalator.json \
             [--alpha 0.1] [--when-unsure answer|review] [--embed-cache FILE] [--chat-cache FILE]
$B eval  --ladder frontdesk.escalator.json --cases holdout.jsonl
$B serve --ladder frontdesk.escalator.json      # คำตอบมี "rate_guard" เมื่อสัดส่วนที่แต่ละชั้นตอบเบี่ยงจากตอน calibrate
```

- **calibrate บนแถวที่คนตรวจแล้วและไม่ได้ใช้ train** (ชุด train ทั้งชุดถูกปฏิเสธ) — คะแนน `1 − p̂(label จริง)` ของแต่ละชั้นให้จุดตัด `q̂`
  ที่อันดับ ⌈(n+1)(1−α)⌉; ภายใต้ exchangeability ชุดคำตอบของแต่ละชั้นครอบคลุม label จริงด้วยความน่าจะเป็น ≥ 1 − α
- **ชั้น LLM อ่านการแจกแจงจาก log-probabilities ของ token แรก** (ตัวเลือกเป็นตัวอักษร A, B, …) — forward pass เดียว ไม่ generate ข้อความ
  (`Chat::top_logprobs`; mlx_lm.server รับ `top_logprobs` ได้ไม่เกิน 11)
- **rate guard** (ไอเดียจาก Rethink): เทียบสัดส่วนที่แต่ละชั้นตอบใน N ครั้งล่าสุดกับตอน calibrate; ผลลัพธ์ที่ไม่เคยเห็นตอน calibrate นับเป็น 0%
- ข้อจำกัด: ชุดคำตอบเลือกได้แค่ "ระหว่าง label" — ข้อความนอกเรื่องยังเป็นหน้าที่ของ novelty gate; จุดตัดจากแถวไม่กี่สิบแถวยังแกว่ง
- **คู่มือการติด label** (`train --guide FILE`, ตั้งแต่ 0.3.0): ข้อความ เช่น rulebook ที่มีลำดับความสำคัญและกฎเส้นแบ่ง วางไว้ก่อนตัวเลือกใน prompt ของชั้น LLM
  ให้ LLM อ่านนิยามชุดเดียวกับคนติด label และตัว gen ข้อมูล · อยู่ในไฟล์ ladder จึงอยู่ใน digest และ receipt · ตั้ง guide ใหม่แล้ว calibration เดิมถูกล้าง ต้อง calibrate ใหม่
- ไฟล์ ladder รุ่นเดิม (ไม่มี `conformal`) ทำงานและ serialize เหมือนเดิมทุก byte — `tests/escalator.rs`

## Rulebook: นิยามของ label ชุดเดียวที่ทุกฝ่ายอ่าน (`ladder rulebook`, ตั้งแต่ 0.4.0)

rulebook คือไฟล์ JSON ไฟล์เดียวที่เก็บนิยามของทุก label พร้อมตัวอย่าง ตัวอย่างที่ใกล้เคียงแต่ไม่ใช่ ลำดับความสำคัญ (เมื่อข้อความเดียวเข้าได้หลาย label) และกฎเส้นแบ่ง
กฎแต่ละข้อมี id, ที่มา, เงื่อนไขก่อนใช้ (`when`) และ KG triple ได้ ทั้งเล่มมี id แบบ BLAKE3 ซึ่งเปลี่ยนทุกครั้งที่เนื้อหาเปลี่ยน
รูปแบบนี้ยืมมาจาก rulebook ของ Tetris ใน [katgpt-rs](https://github.com/katopz/katgpt-rs/blob/develop/crates/katgpt-tetris/src/rulebook.rs) ของ katopz

```sh
$B rulebook check  --rulebook rulebook.json
$B rulebook render --rulebook rulebook.json --guide-out guide.txt --descriptions-out labels.json
$B train --task T --train train.jsonl --out t.json --descriptions labels.json --guide guide.txt --llm gemma-4-26b
```

- คนติด label, ตัวสร้างข้อมูล และชั้น LLM อ่าน rulebook เล่มเดียวกัน guide อยู่ในไฟล์ ladder จึงอยู่ใน digest ด้วย
- id ตรงกับ `blake3(json.dumps(rb, sort_keys=True, ensure_ascii=False, separators=(",", ":")))` ของ Python เครื่องมือทั้งสองภาษาจึงได้ id เดียวกัน
- **ตัวอย่างที่รันได้ทันที (offline):** [`examples/frontdesk-rulebook/`](examples/frontdesk-rulebook/) ส่งข้อความถึงเคาน์เตอร์โรงพยาบาลไป 3 แผนก
  และเทียบบันไดที่ต่างกันแค่ guide (นิยามสั้น vs rulebook ทั้งเล่ม)

## สิ่งที่ implement

| ไฟล์ | หน้าที่ |
|---|---|
| `src/tokenize.rs` | `Ascii` = เหมือน upstream ทุกไบต์ · `Unicode` = ASCII เหมือนเดิม + Thai character bigram/trigram + normalize เลขไทย/สระอำ/full-width |
| `src/embed.rs` | hashed bag → `dim` ช่อง, L2-normalize, (option) IDF จาก corpus |
| `src/drafter.rs` | LZ4 compressed-length delta (lz4_flex 0.13.1 เวอร์ชันเดียวกับ upstream) |
| `src/gate.rs` | corpus-distance gate |
| `src/calibrate.rs` | Platt scaling บน logit(p), Laplace-smoothed targets, window 512, refit ที่ 64 |
| `src/threshold.rs` | fit threshold จาก labeled slice: ρ-quantile และ target-accuracy, น้อยกว่า 16 เคส = ไม่แนะนำ |
| `src/engine.rs` | pipeline ทั้งหมด + demo corpus ของ upstream |
| `src/wire.rs` | `/decide` request/response + กฎ 422 |
| `src/bin/reflex-study.rs` | `serve` (HTTP `/decide` `/feedback` `/healthz`) และ `eval` (รันไฟล์เคสที่มี label) |

### ตรวจกับ binary ตัวจริง (reflex 0.2.4, aarch64-apple-darwin, ตรวจ SHA256 แล้ว)

- `tests/parity.rs` — 154 requests ที่อัดจาก binary จริง (`scripts/record_parity.py`): **264/264 คำตอบตรงกันทุกบิต**
  (outcome, probabilities, confidence, routing reason)
- HTTP: response ตรงกันทุกไบต์ทั้ง 200 และ error 400/404/413/422
- feedback 80 ครั้งแล้วถามซ้ำ: confidence หลัง calibrate ตรงกัน (0.749459), temperature ต่างที่หลักที่ 5 (f64 vs f32)

### ภาษาไทย (`corpora/th-frontdesk` — งานธุรการหน้าเคาน์เตอร์ 3 แผนก, 9 เอกสาร, ไม่มีเนื้อหาทางคลินิก)

ไฟล์ calibration (33 เคสในขอบเขต + 10 เคสนอกเรื่อง) ใช้เลือก config และ fit gate midpoint
ไฟล์ holdout (15 + 6) เขียนก่อนเลือก config และรันครั้งเดียว

| config | calibration: เลือกถูก (บังคับเลือก) | gate AUC | holdout: เลือกถูก | holdout: นอกเรื่องที่ abstain |
|---|---|---|---|---|
| upstream (`ascii`, 256) | 12/33 (chance 11) | 0.515 | 5/15 (chance 5) | 6/6 แต่ abstain ทุกเคสรวมถึงเคสในขอบเขต (ตอบ 0/15) |
| `unicode`, 256 | 27/33 | 0.861 | — | — |
| `unicode`, 1024, gate_mid 0.1511 | **30/33** | **0.970** | **13/15** | **4/6** |

- จำนวนช่อง (dim) สำคัญที่สุด: Thai n-gram เยอะ 256 ช่องชนกันมาก ที่ 1024 ช่อง IDF ทำให้แย่ลง (AUC 0.933, 29/33)
  และ gate ที่อ่าน state อย่างเดียวได้ AUC ใกล้กัน (0.979) แต่จุดตัดที่ดีที่สุดแบ่งได้แย่กว่า จึงคงแบบของ upstream ไว้
- gate midpoint ต้อง fit ต่อ corpus: cosine ของภาษาไทยต่ำกว่าภาษาอังกฤษ (in-corpus median ~0.2) ค่า 0.35 ของ upstream จึง abstain เกือบหมด
- ผิด 2 เคสใน holdout เป็นเคส records ที่ไปลง billing เพราะคำว่า "สำเนา" / "ใบรับรองแพทย์" อยู่ในเอกสาร billing ด้วย
  — ขีดจำกัดของการจับคำ ไม่มีความเข้าใจความหมาย
- ตัวอย่างน้อย (holdout 21 เคส) ตัวเลขเหล่านี้บอกทิศทาง ไม่ใช่ความแม่นยำที่อ้างได้

## วิธีรัน

```sh
cargo test --release
B=target/release/reflex-study

$B serve
$B serve --corpus corpora/first-corpus
$B serve --corpus corpora/th-frontdesk --tokenizer unicode --dim 1024 --gate-mid 0.1511 --bind 127.0.0.1:7342

$B eval --corpus corpora/th-frontdesk --cases tests/fixtures/th_frontdesk_cases.jsonl --tokenizer unicode --dim 1024
```

```sh
curl -s -X POST http://127.0.0.1:7342/decide -d '{"state":"ขอใบเสร็จค่ารักษาใหม่ ฉบับเดิมหาย","questions":[{"id":"route","kind":"choice","prompt":"ส่งเรื่องนี้ไปแผนกไหน","options":["appointment","billing","records"]}]}'
```

## เหมาะกับอะไร

- เหมาะ: routing/triage ที่ชุดคำตอบชัดและเขียนเอกสารอธิบายแต่ละกลุ่มได้ — เป็นด่านแรกราคาถูก (µs, ไม่มี GPU)
  ก่อนส่งต่อ LLM เฉพาะเคสที่ abstain
- ไม่เหมาะ: งานที่ต้องเข้าใจความหมาย หรือการตัดสินใจทางคลินิก — มันจับคำที่ซ้ำกัน ไม่ได้เข้าใจเนื้อหา

## License

AGPL-3.0-or-later — ดู [LICENSE](LICENSE) และ [COMMERCIAL.md](COMMERCIAL.md)
ส่วนที่มาจาก upstream (MIT) ดู [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

ขอบคุณ [katopz](https://github.com/katopz) สำหรับ Reflex, riir-instinct และ katgpt-rs
ที่เปิดเป็น MIT พร้อมเอกสารละเอียดพอให้คนนอกเขียนใหม่ได้ตรงทุกบิต
