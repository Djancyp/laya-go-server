"""Dev-only: dump Laya reference outputs (Python/PyTorch) that the Go port must reproduce.

    .venv-laya/bin/python laya/testdata/gen_ref.py

Writes laya/testdata/tokenizer_ref.json (HF tokenizer ids) and laya/testdata/predict_ref.json
(sequence ids, marker positions, encoder hidden states, head logits, final answers).
Hidden states are float32 little-endian, base64. Python is NOT needed at runtime.
"""
import base64, json, os, sys
import numpy as np, torch
import laya
from laya.common import QTYPES
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "..", "..", "models", "laya", "tokenizer.json")

# ---- tokenizer corpus ----
texts = [
    "", " ", "hello", "Hello world", " leading space", "trailing space ", "two  spaces", "   three",
    "line one\nline two", "\n\n\n\n", "tab\there", "a\r\nb",
    "what date is it", "I'm so angry!!! Nothing works.", "user: hi\nassistant: hello",
    "Which tool does the latest user message need, given the conversation?",
    "get_date: User asks for today's date, the current day, or what day it is",
    "¿Qué hora es? Estoy muy triste hoy.", "今天天气怎么样？我很开心！", "مرحبا كيف حالك", "मुझे बहुत गुस्सा आ रहा है",
    "日本語のテキストです。", "Привет, как дела?", "emoji 😀🔥 and 🇪🇸 flags", "e\u0301 combining and é precomposed",
    "<h1>heading</h1>", "<table><tr><td>x</td></tr></table>", "<mask> literal mask", "<bos> <eos> <pad> <unk>",
    "[@BOS@] and <2mass> and <unused0>", "<start_of_turn>user\nhi<end_of_turn>",
    "https://example.com/a?b=1&c=2#frag", "email me at a.b+c@example.org", "C:\\path\\to\\file.txt",
    "snake_case camelCase kebab-case", "3.14159 and 1,000,000 and -42", "```go\nfmt.Println(1)\n```",
    "\u200bzero width\u200b", "\ufffd replacement", "control \x01 char", "nbsp\u00a0here", "x" * 300, "ab " * 120,
    "Zażółć gęślą jaźń", "안녕하세요 세계", "ไทยภาษา", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙",
]
tk = Tokenizer.from_file(TOK)
json.dump([{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in texts],
          open(os.path.join(HERE, "tokenizer_ref.json"), "w"), ensure_ascii=False)
print("tokenizer cases:", len(texts))

# ---- predict corpus ----
TOOLS = {"none": "No tool needed; ordinary chat", "ask_user": "The latest user message is too vague to act on",
         "get_date": "User asks for today's date or what day it is", "system_info": "User asks about this computer or its specs",
         "web": "Needs the internet: a URL or a factual question about the world"}
EMO = {"neutral": "No strong emotion", "happy": "Happy, pleased or thankful", "excited": "Excited or eager",
       "sad": "Sad, down, lonely or hurt", "angry": "Angry, irritated or rude", "frustrated": "Frustrated or fed up",
       "anxious": "Anxious, worried or stressed", "confused": "Confused or lost"}
Q_TOOL = {"type": "choice", "instructions": "Which tool does the latest user message need, given the conversation?", "criteria": TOOLS}
Q_EMO = {"type": "choice", "instructions": "What emotion does the latest user message express?", "criteria": EMO}
Q_FITS = {"type": "noul", "instructions": "Is `question` a sensible clarifying question for the last user message in `conversation`?"}
Q_SCORE = {"type": "score", "instructions": "How polite is the message?",
           "criteria": ["very rude", "slightly rude", "neutral", "polite", "very polite"]}
cases = [
    ("user: what date is it", {"tool": Q_TOOL, "emotion": Q_EMO}),
    ("user: i feel so alone today", {"tool": Q_TOOL, "emotion": Q_EMO}),
    ("user: THIS IS THE WORST SERVICE EVER, fix it now", {"tool": Q_TOOL, "emotion": Q_EMO}),
    ("user: hi", {"tool": Q_TOOL, "emotion": Q_EMO}),
    ("user: weather", {"tool": Q_TOOL}),
    ("user: ¿qué hora es? estoy muy triste hoy", {"tool": Q_TOOL, "emotion": Q_EMO}),
    ("user: 今天天气怎么样？我很开心！", {"tool": Q_TOOL, "emotion": Q_EMO}),
    ({"conversation": "user: weather", "question": "Which city?", "options": ["Rome", "Oslo"]}, {"fits": Q_FITS}),
    ("user: please could you kindly help me", {"polite": Q_SCORE}),
    ("Hello <mask> world <h1>x</h1>\n\ttabs   and  spaces 😀 " + "long text " * 200, {"tool": Q_TOOL, "fits": Q_FITS}),
    ([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "weather"}], {"tool": Q_TOOL}),
    ("x", {"many": {"type": "choice", "instructions": "pick", "criteria": {f"opt{i}": f"description number {i} " * 6 for i in range(25)}}}),
]

agent = laya.load("convaiinnovations/laya-multilingual", device="cpu")
from laya.common import build_sequence, serialize_state, encode_text
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()
out = []
for state, questions in cases:
    res = agent.predict(state, questions)
    ids = list(questions)
    internal = {q: agent._to_internal(questions[q]) for q in ids}
    items = agent._encode_state(state, ids, internal)
    per_q = {}
    for qid, it in zip(ids, items):
        seq = torch.tensor([it["ids"]])
        mask = torch.ones_like(seq)
        with torch.no_grad():
            h = agent.model.encoder(input_ids=seq, attention_mask=mask).last_hidden_state
            marker = torch.tensor([it["markers"]])
            logits, act = agent.model(seq, mask, marker, torch.ones_like(marker, dtype=torch.bool), torch.tensor([it["qtype"]]))
        per_q[qid] = {"ids": it["ids"], "markers": it["markers"], "qtype": it["qtype"],
                      "hidden": b64(h[0].numpy()), "n": int(h.shape[1]), "d": int(h.shape[2]),
                      "logits": logits[0, :len(it["markers"])].tolist(), "act": torch.softmax(act.float(), -1)[0].tolist()}
    out.append({"state": state, "questions": questions, "per_question": per_q, "answers": res["answers"]})
    print("case ok:", str(state)[:50].replace("\n", " "), {q: (a.get("choice") or a.get("noul") or a.get("score")) for q, a in res["answers"].items()})
json.dump(out, open(os.path.join(HERE, "predict_ref.json"), "w"), ensure_ascii=False)
print("predict cases:", len(out), "size KB:", os.path.getsize(os.path.join(HERE, "predict_ref.json")) // 1024)
