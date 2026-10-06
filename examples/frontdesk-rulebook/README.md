# ตัวอย่าง: rulebook + บันไดเลื่อน (เคาน์เตอร์โรงพยาบาล)

ตัวอย่างนี้ทำวิธีจาก blog
[Rethink กับบันไดเลื่อน](https://asgard.megawiz.co.th/blog/rethink-vs-escalator-decision-ladder-th)
ครบทุกขั้น บนงานที่เปิดได้ คือส่งข้อความที่ผู้ติดต่อพิมพ์ถึงเคาน์เตอร์โรงพยาบาลไป 1 ใน 3 แผนก
`appointment` (นัดหมาย), `billing` (การเงิน) และ `records` (เวชระเบียน)
ทุกอย่างในโฟลเดอร์นี้สมมติขึ้น ไม่ใช่ระเบียบหรือข้อมูลของโรงพยาบาลจริง

## รัน (offline ไม่ต้องมีโมเดล)

```sh
./examples/frontdesk-rulebook/run.sh
```

สคริปต์ทำ 4 ขั้น

1. `ladder rulebook render` อ่าน `rulebook.json` แล้วเขียน `out/guide.txt`
   (rulebook ทั้งเล่มสำหรับ prompt ของชั้น LLM) และ `out/labels.json` (นิยามสั้นของแต่ละแผนก)
2. train บันไดเลื่อน 2 ตัวที่**ต่างกันแค่ guide**
   - `short` ใช้นิยามสั้นอย่างเดียว
   - `rulebook` ใช้นิยามสั้นบวก rulebook ทั้งเล่ม (`--guide`)
3. calibrate แบบ conformal (`--alpha 0.05`) บน `data/calibrate.jsonl`
4. วัดผลบน `data/test.jsonl` และ `data/hard.jsonl`

คำตอบของชั้น LLM สำหรับทุกคำถามที่สคริปต์ถามถูกบันทึกไว้ใน `cache/chat-cache.json` แล้ว
(log-probability ของ token แรก โดย key เป็น sha256 ของชื่อโมเดล prompt และข้อความ)
จึงรันซ้ำได้ผลเดิมทุกข้อโดยไม่ต้องมีโมเดลหรือ key
ต่างกันแค่ตัวเลขเวลา เพราะอ่านจาก cache ใช้ไม่กี่ µs ส่วนเรียกโมเดลจริงใช้ราว 0.35 s

## ผลที่ได้

```
short     test  labeled n=86: answered 86 (100%) · right 84 · accuracy on answered 0.977
short     hard  labeled n=72: answered 72 (100%) · right 63 · accuracy on answered 0.875
rulebook  test  labeled n=86: answered 86 (100%) · right 85 · accuracy on answered 0.988
rulebook  hard  labeled n=72: answered 72 (100%) · right 66 · accuracy on answered 0.917
```

| | test (86 ข้อ) | hard (72 ข้อ) | ชั้น LLM ผิดใน hard | ชั้นล่างผิดเงียบใน hard |
|---|---|---|---|---|
| นิยามสั้น | 84 | 63 | 4 จาก 34 | 5 จาก 38 |
| rulebook ทั้งเล่ม | 85 | **66** | **1 จาก 34** | 5 จาก 38 |

อ่านผลอย่างไร

- **ชุด test ง่ายเกินไป** ทั้งสองแบบได้ราว 98% เพราะ Gemini เป็นทั้งคนแต่งและคนติด label
  โดยอ่าน rulebook เล่มเดียวกัน ข้อความเลยออกมาตรงตามนิยามเกือบหมด
  นี่คือเหตุผลที่ต้องมีชุด hard ซึ่งคนแต่งเป็นคนละคนกับคนติด label
- **rulebook ช่วยชั้น LLM ในข้อยาก** ข้อที่ผิดลดจาก 4 เหลือ 1
  ทั้ง 4 ข้อที่นิยามสั้นพลาดเป็นข้อที่ขอหลายเรื่องในข้อความเดียว (กฎ R4)
  นิยามสั้นบอกแค่ลำดับความสำคัญ แต่ไม่ได้บอกวิธีใช้ ข้อที่ rulebook ยังพลาดก็เป็น R4 เช่นกัน
- **ชั้นล่างยังผิดเงียบ 5 ข้อเท่าเดิม** เพราะชุด calibrate เป็นข้อง่ายแบบเดียวกับ test
  จุดตัดของ conformal จึงหลวมเกินไปสำหรับข้อยาก
  บทเรียนคือ **ชุด calibrate ต้องหน้าตาเหมือนข้อความจริงที่ระบบจะเจอ** ถ้าเอาข้อยากไป calibrate ด้วย ชั้นล่างจะส่งข้อพวกนี้ขึ้นแทน

## ข้างในมีอะไร

| ไฟล์ | |
|---|---|
| `rulebook.json` | 3 แผนก พร้อมนิยาม ตัวอย่าง และตัวอย่างที่ใกล้เคียงแต่ไม่ใช่ · ลำดับความสำคัญ `records > billing > appointment` · กฎ R1–R5 (R5 มีเงื่อนไข `when`) · id `3e1e4dab49bfe225` |
| `data/train.jsonl` | 171 ข้อ |
| `data/calibrate.jsonl` | 87 ข้อ (ไม่ซ้ำกับ train) |
| `data/test.jsonl` | 86 ข้อ |
| `data/hard.jsonl` | 72 ข้อ ตรงเส้นแบ่งของกฎ R1–R5 ข้อละ 12 และอีก 12 ข้อที่ใช้คำหลอก |
| `cache/chat-cache.json` | คำตอบของชั้น LLM (gemma-4-26b) ที่บันทึกไว้ |
| `run.sh` | ทั้งหมดข้างบนในสคริปต์เดียว |

## ข้อมูลมาจากไหน

- **train / calibrate / test:** ให้ Gemini (gemini-3.8-flash) แต่งตาม rulebook
  - แต่งทีละแผนก 225 ข้อ
  - คู่เทียบตามกฎ R1, R2, R3 และ R5 อีก 96 ข้อ คือสถานการณ์เดียวกันที่ต่างกันแค่จุดเดียว
  - ข้อที่ขอหลายเรื่องในข้อความเดียว (R4) 24 ข้อ
  - จากนั้นให้ Gemini ติด label ซ้ำโดยไม่เห็นของเดิม เก็บข้อที่ตรงกัน 344 จาก 345 ข้อ
  - คู่เทียบอยู่ในชุดเดียวกันเสมอ ไม่แยกไปอยู่คนละชุด
- **hard:** Claude แต่ง 72 ข้อแยกจากชุดอื่น โดยอ่านแค่ `rulebook.json` แล้วติด label พร้อมเลขกฎ
  จากนั้นให้ Gemini ติด label แยกอีกรอบ ได้ตรงกันครบ 72/72
- **ไม่ใช้ gemma ทำคำตอบอ้างอิง** เพราะ gemma อยู่ในบันไดเอง ถ้าใช้ก็เหมือนให้ผู้เข้าสอบออกข้อสอบเอง

## ใช้กับงานของคุณ

1. เขียน `rulebook.json` ของคุณตามรูปแบบในไฟล์นี้ ช่องที่ใช้คือ
   - `labels[]`: `id`, `th` (ชื่อที่แสดง), `definition`, `positive[]`, `near_miss[]` (`text`, `is`, `why`)
   - `precedence.order`
   - `rules[]`: `id`, `rule`, และถ้ามี `when[]`, `triple`, `source`
   - ช่องอื่น เช่น `decisions` ใส่ได้ ไม่ถูกตีความ แต่นับรวมใน id
2. `ladder rulebook check --rulebook rulebook.json` จะจับ label ซ้ำ, ลำดับความสำคัญที่ขาดหรือเกิน,
   และตัวอย่างที่ชี้ไปหา label ที่ไม่มีอยู่
3. แบ่งแถวเป็น train, calibrate และ test โดยแถว calibrate ต้องไม่อยู่ใน train
   และควรหน้าตาเหมือนข้อความจริงที่ระบบจะเจอ
4. ตั้ง gateway แบบ OpenAI-compatible ที่ส่ง `logprobs` / `top_logprobs` ได้ (เช่น mlx_lm.server)
   - ใช้ตัวแปร `HEIMDALL_API_URL` และ `HEIMDALL_API_KEY`
   - ถ้าโมเดลไม่ใช่ gemma-4-26b ให้ตั้ง `HEIMDALL_LOCAL_MODEL=<ชื่อโมเดล>` แล้วใช้ `--llm <ชื่อโมเดล>`
   - ถ้าจะใช้ชั้น encoder ด้วย ให้ลบ `--no-encoder` ออก (ต้องมี `/v1/embeddings` ที่เสิร์ฟ `BAAI/bge-m3`)
5. ถ้าอยากให้ข้อที่ไม่แน่ใจไปถึงคนแทนการเดา ใช้ `ladder calibrate --when-unsure review`
   ข้อนั้นจะได้ `rung = "review"` พร้อมชุด label ที่ยังเป็นไปได้

แก้ rulebook เมื่อไร id จะเปลี่ยน ให้ train ใหม่ด้วย guide ใหม่ แล้ว calibrate ใหม่ทุกครั้ง
