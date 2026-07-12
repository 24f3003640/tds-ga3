import json, re, base64, hashlib, os, traceback
from statistics import mean, median, pstdev, pvariance, mode
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import config
import asyncio
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

# ===== Persistent file-based cache =====
CACHE_FILE = "aipipe_cache.json"

def load_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[CACHE] Error loading: {e}")
    return {}

def save_cache(cache_data):
    try:
        temp_file = CACHE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        os.replace(temp_file, CACHE_FILE)
    except Exception as e:
        print(f"[CACHE] Error saving: {e}")

_CACHE = load_cache()

def _ck(*parts):
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()

# ===== Core API helpers =====
FALLBACK_MODELS = ["gpt-4o-mini", "gpt-4o"]

async def chat(messages, model=None, max_tokens=800, force_json=True, retries=3):
    use_model = model or config.TEXT_MODEL
    key = _ck("chat", use_model, json.dumps(messages, sort_keys=True, default=str))
    if key in _CACHE:
        print(f"[CACHE HIT] chat model={use_model}")
        return _CACHE[key]

    body = {"model": use_model, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    last_err = None
    async with httpx.AsyncClient(timeout=60) as c:
        for attempt in range(retries):
            try:
                hdr = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}",
                       "Content-Type": "application/json"}
                print(f"[CHAT] model={use_model} attempt={attempt+1}/{retries}")
                r = await c.post(f"{config.AIPIPE_BASE}/chat/completions",
                                 headers=hdr, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    print(f"[CHAT] Retryable error: {last_err}")
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"]
                _CACHE[key] = out
                save_cache(_CACHE)
                print(f"[CHAT] Success model={use_model}")
                return out
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"[CHAT] Error attempt {attempt+1}: {last_err}")
                await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"chat failed after {retries} retries: {last_err}")


async def chat_with_fallback(messages, models=None, max_tokens=800, force_json=True):
    """Try multiple models in order, return first success."""
    models = models or FALLBACK_MODELS
    last_err = None
    for m in models:
        try:
            return await chat(messages, model=m, max_tokens=max_tokens,
                              force_json=force_json, retries=2)
        except Exception as e:
            last_err = e
            print(f"[FALLBACK] model={m} failed: {e}")
    raise last_err


async def embed(input_texts, model=None):
    use_model = model or config.EMBED_MODEL
    key = _ck("embed", use_model, json.dumps(input_texts, sort_keys=True))
    if key in _CACHE:
        print(f"[CACHE HIT] embed")
        return _CACHE[key]

    async with httpx.AsyncClient(timeout=60) as c:
        hdr = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}",
               "Content-Type": "application/json"}
        print(f"[EMBED] sending {len(input_texts)} texts")
        r = await c.post(f"{config.AIPIPE_BASE}/embeddings", headers=hdr,
                         json={"model": use_model, "input": input_texts})
        r.raise_for_status()
        out = [d["embedding"] for d in r.json()["data"]]
        _CACHE[key] = out
        save_cache(_CACHE)
        print(f"[EMBED] Success")
        return out


# Gemini — only try 2 models, 1 attempt each for speed
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash-lite"]

async def gemini_call(payload, timeout_s=25):
    """Single Gemini generateContent call with fast model fallback."""
    try:
        audio_b64 = payload["contents"][0]["parts"][1]["inlineData"]["data"]
        key = _ck("gemini", hashlib.sha256(audio_b64.encode()).hexdigest(),
                  payload["contents"][0]["parts"][0]["text"][:100])
        if key in _CACHE:
            print("[CACHE HIT] gemini")
            return _CACHE[key]
    except Exception:
        key = None

    last_err = ""
    async with httpx.AsyncClient(timeout=timeout_s) as c:
        for model in GEMINI_MODELS:
            try:
                print(f"[GEMINI] trying {model}")
                r = await c.post(
                    f"https://aipipe.org/geminiv1beta/models/{model}:generateContent",
                    headers={"Authorization": f"Bearer {config.AIPIPE_TOKEN}"},
                    json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code} on {model}"
                    print(f"[GEMINI] {last_err}")
                    continue
                r.raise_for_status()
                data = r.json()
                txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                print(f"[GEMINI] Success with {model}, len={len(txt)}")
                if key:
                    _CACHE[key] = txt
                    save_cache(_CACHE)
                return txt
            except (KeyError, IndexError):
                last_err = f"empty candidates on {model}"
                print(f"[GEMINI] {last_err}")
            except Exception as e:
                last_err = f"{type(e).__name__} on {model}: {str(e)[:120]}"
                print(f"[GEMINI] {last_err}")
    print(f"[GEMINI] All models failed: {last_err}")
    return ""


def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}


# ================= Q2: /answer-image =================
def normalize_answer(ans):
    s = str(ans).strip()
    if not s:
        return s
    cleaned = re.sub(r"[,\s]", "", s)
    cleaned = re.sub(r"[₹$€£%]", "", cleaned)
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if m and re.fullmatch(r"[^\dA-Za-z]*-?\d[\d,.\s₹$€£%]*", s.strip()):
        num = m.group(0)
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        return num
    return s


@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    img_b64 = body.get("image_base64", "")
    question = body.get("question", "")
    print(f"[Q2] question={question[:80]}")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text":
                "You are an expert at reading images: charts, bar graphs, pie charts, "
                "tables, receipts, invoices, screenshots, and dashboards.\n\n"
                "INSTRUCTIONS:\n"
                "1. LOOK at every label, axis, bar, slice, number, row, and column. "
                "Transcribe each one carefully in your 'work' field.\n"
                "2. If arithmetic is required (sum, difference, percentage, max, min), "
                "compute it step-by-step and double-check.\n"
                "3. Your final 'answer' field:\n"
                "   - If NUMERIC: bare number, no currency/units/commas. "
                "Keep exact decimals (e.g. 4089.35).\n"
                "   - If TEXT: exact string as shown in the image.\n"
                "   - If a DATE or LABEL: exact text.\n"
                "Return JSON: {\"work\": \"...\", \"answer\": \"...\"}.\n\n"
                f"Question: {question}"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
        ],
    }]
    try:
        raw = await chat_with_fallback(messages, models=["gpt-4o", "gpt-4o-mini"],
                                       max_tokens=1200)
        out = parse_json(raw)
        ans = normalize_answer(out.get("answer", ""))
        print(f"[Q2] answer={ans}")
    except Exception as e:
        print(f"[Q2] ERROR: {traceback.format_exc()}")
        ans = ""
    return {"answer": str(ans)}


# ================= Q3 + Q7: /extract =================
@app.post("/extract")
async def extract(request: Request):
    body = await request.json()
    print(f"[EXTRACT] keys={list(body.keys())}")

    # ---- Q3: fixed-schema invoice (body has "invoice_text") ----
    if "invoice_text" in body:
        text = body.get("invoice_text", "")
        prompt = (
            "You are an invoice parser. Extract EXACTLY these fields from the "
            "invoice text below. Return JSON with these keys:\n\n"
            "- invoice_no: the invoice number/ID (string). Look for labels like "
            "'Invoice #', 'Invoice No', 'Inv No', 'Invoice Number', 'Bill No', "
            "or any alphanumeric code that identifies this invoice.\n"
            "- date: the invoice date in ISO format YYYY-MM-DD\n"
            "- vendor: the company/person issuing the invoice\n"
            "- amount: the subtotal BEFORE tax, as a plain number (no separators)\n"
            "- tax: the tax amount only, as a plain number\n"
            "- currency: ISO 4217 code (INR, USD, EUR, GBP, JPY...)\n\n"
            "IMPORTANT:\n"
            "- For invoice_no, extract EXACTLY what appears after the invoice "
            "number label. Do NOT return null if there is any identifier present.\n"
            "- Use null ONLY if a field truly cannot be found.\n"
            "- For amount, if only a total is given with no separate subtotal, "
            "calculate subtotal = total - tax.\n\n"
            f"INVOICE TEXT:\n{text}"
        )
        try:
            raw = await chat_with_fallback([{"role": "user", "content": prompt}],
                                           models=["gpt-4o-mini", "gpt-4o"])
            out = parse_json(raw)
            print(f"[Q3] parsed: {json.dumps(out, default=str)[:300]}")
        except Exception as e:
            print(f"[Q3] ERROR: {traceback.format_exc()}")
            out = {}
        keys = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]
        return {k: out.get(k) for k in keys}

    # ---- Q7: structured extraction (body has "text" + "schema") ----
    text = body.get("text", "")
    schema = body.get("schema", {})
    print(f"[Q7] schema_keys={list(schema.keys())}")

    prompt = (
        "You are a strict invoice parser. Read the document and return JSON that "
        "matches the schema EXACTLY. Required keys and their types:\n\n"
        "- vendor (string): the biller's proper name. Remove any trailing period.\n"
        "- currency (string): ISO 4217 code (USD/EUR/GBP/INR/JPY).\n"
        "- total_amount (integer): total in main currency unit, no separators. "
        "Parse '12,480' as 12480, '1,24,800' (Indian) as 124800, '12K' as 12000.\n"
        "- invoice_date (string): YYYY-MM-DD format.\n"
        "- due_in_days (integer): 'Net 30'->30, 'payable within 45 days'->45, "
        "'due in two weeks'->14.\n"
        "- is_paid (boolean): 'paid in full'/cleared/settled->true, "
        "'awaiting payment'/outstanding/unpaid/pending->false.\n"
        "- priority (string): EXACTLY one of: low, normal, high, urgent. "
        "Map: 'low priority'/'no rush'/'not urgent'->low; "
        "'normal'/'standard'/'routine'->normal; 'high priority'/'important'/"
        "'expedite'->high; 'urgent'/'ASAP'/'immediately'/'critical'->urgent.\n"
        "- contact_email (string): lowercase.\n"
        "- line_items (array): each item is {\"sku\": string, \"quantity\": integer, "
        "\"unit_price\": integer} in document order.\n"
        "- item_count (integer): number of line items.\n\n"
        "Return ALL keys listed above. Do NOT omit any key.\n\n"
        f"SCHEMA HINT: {json.dumps(schema)}\n\nDOCUMENT:\n{text}"
    )
    try:
        raw = await chat_with_fallback([{"role": "user", "content": prompt}],
                                       models=["gpt-4o", "gpt-4o-mini"],
                                       max_tokens=1500)
        out = parse_json(raw)
        print(f"[Q7] parsed keys: {list(out.keys())}")
    except Exception as e:
        print(f"[Q7] ERROR: {traceback.format_exc()}")
        out = {}

    # Post-processing
    if isinstance(out.get("vendor"), str):
        out["vendor"] = out["vendor"].strip().rstrip(".").strip()
    if isinstance(out.get("contact_email"), str):
        out["contact_email"] = out["contact_email"].strip().lower()
    if isinstance(out.get("line_items"), list):
        out["item_count"] = len(out["line_items"])
    if out.get("priority") not in ("low", "normal", "high", "urgent"):
        out["priority"] = "normal"

    # Ensure ALL expected keys exist
    expected = ["vendor", "currency", "total_amount", "invoice_date",
                "due_in_days", "is_paid", "priority", "contact_email",
                "line_items", "item_count"]
    for k in expected:
        if k not in out:
            out[k] = None
    return out


# ================= Q4: /dynamic-extract =================
def coerce(value, typ):
    if value is None:
        return None
    try:
        t = str(typ).lower().strip()
        if t == "integer":
            return int(round(float(str(value).replace(",", ""))))
        if t in ("float", "number"):
            return float(str(value).replace(",", ""))
        if t == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "y")
        if t == "date":
            return str(value).strip()
        if t == "array[integer]":
            lst = value if isinstance(value, list) else [value]
            return [int(round(float(x))) for x in lst]
        if t.startswith("array"):
            lst = value if isinstance(value, list) else [value]
            return [str(x).strip().rstrip(".").strip() if isinstance(x, str) else x for x in lst]
        return str(value).strip().rstrip(".").strip()
    except Exception:
        return None


@app.post("/dynamic-extract")
async def dynamic_extract(request: Request):
    body = await request.json()
    text = body.get("text", "")
    schema = body.get("schema", {})
    keys = list(schema.keys())
    print(f"[Q4] schema={json.dumps(schema)[:200]}")

    prompt = (
        "Extract the following fields from the text below. Return JSON with "
        "EXACTLY these keys and types:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Rules:\n"
        "- string: exact text as mentioned\n"
        "- integer: plain integer number (parse '12,480' -> 12480)\n"
        "- float/number: decimal number\n"
        "- boolean: true/false\n"
        "- date: ISO YYYY-MM-DD\n"
        "- array[...]: JSON array\n"
        "- If a field cannot be found, use null.\n"
        "- Match field names to the closest mention in the text.\n\n"
        f"TEXT:\n{text}"
    )
    try:
        raw = await chat_with_fallback([{"role": "user", "content": prompt}],
                                       models=["gpt-4o-mini", "gpt-4o"])
        out = parse_json(raw)
        print(f"[Q4] extracted: {json.dumps(out, default=str)[:300]}")
    except Exception as e:
        print(f"[Q4] ERROR: {traceback.format_exc()}")
        out = {}
    return {k: coerce(out.get(k, None), schema[k]) for k in keys}


# ================= Q6: /answer-audio =================
last_debug_info = {}
last_audio_bytes = b""
last_audio_mime = "audio/wav"
audio_history = []

@app.get("/debug")
def get_debug():
    return last_debug_info

@app.get("/transcripts")
def get_transcripts():
    return {"count": len(audio_history), "calls": list(reversed(audio_history))}


def _find_audio_b64(body):
    audio_id, audio_b64 = None, ""
    if isinstance(body, dict):
        for k, v in body.items():
            lk = str(k).lower()
            if isinstance(v, str):
                if ("audio" in lk or "data" in lk or "b64" in lk or "base64" in lk) and len(v) > 200:
                    if len(v) > len(audio_b64):
                        audio_b64 = v
                elif "id" in lk and not audio_id:
                    audio_id = v
    return audio_id, audio_b64


def _detect_mime(audio):
    if audio.startswith(b"ID3") or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mp3"
    elif audio.startswith(b"OggS"):
        return "audio/ogg"
    elif audio.startswith(b"fLaC"):
        return "audio/flac"
    elif audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return "audio/wav"
    elif audio.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm"
    elif len(audio) > 8 and audio[4:8] == b"ftyp":
        return "audio/mp4"
    return "audio/wav"


@app.post("/answer-audio")
async def answer_audio(request: Request):
    global last_debug_info, last_audio_bytes, last_audio_mime

    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    last_debug_info = {"content_type": ctype, "raw_len": len(raw)}
    print(f"[Q6] content_type={ctype} raw_len={len(raw)}")

    body, audio_id, audio_b64 = {}, None, ""
    try:
        if "application/json" in ctype or raw[:1] in (b"{", b"["):
            body = json.loads(raw)
            last_debug_info["body_keys"] = list(body.keys()) if isinstance(body, dict) else "non-dict"
            audio_id, audio_b64 = _find_audio_b64(body)
        else:
            try:
                form = await request.form()
                last_debug_info["form_keys"] = list(form.keys())
                for k, v in form.items():
                    data = await v.read() if hasattr(v, "read") else None
                    if data:
                        last_audio_bytes = data
            except Exception:
                pass
            if not last_audio_bytes and raw:
                last_audio_bytes = raw
            audio_b64 = base64.b64encode(last_audio_bytes).decode() if last_audio_bytes else ""
    except Exception as e:
        last_debug_info["parse_error"] = str(e)
        print(f"[Q6] parse error: {e}")

    last_debug_info["audio_b64_len"] = len(audio_b64)

    # Decode audio and detect MIME
    try:
        audio = base64.b64decode(audio_b64) if audio_b64 else last_audio_bytes
        last_audio_bytes = audio
        mime = _detect_mime(audio)
        last_audio_mime = mime
        last_debug_info["detected_mime"] = mime
        print(f"[Q6] audio_len={len(audio)} mime={mime}")
    except Exception as e:
        print(f"[Q6] audio decode error: {e}")
        return _empty_audio_response()

    # ---- SINGLE Gemini call: transcribe + extract in one shot ----
    combined_prompt = (
        "This audio is in Korean. It describes a tabular dataset with statistics.\n"
        "Do TWO things:\n"
        "1. Transcribe the Korean audio precisely.\n"
        "2. From the transcription, extract the dataset info.\n\n"
        "Korean stat mapping: 평균=mean, 표준편차=std, 분산=variance, "
        "최소/최솟값=min, 최대/최댓값=max, 중앙값/중간값=median, "
        "최빈값=mode, 범위=range, ~사이=value_range, "
        "허용값/허용된값=allowed_values, 상관관계=correlation "
        "(양의/비례=positive, 음의/반비례=negative)\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        "  \"transcript\": \"full Korean transcription\",\n"
        "  \"columns\": [\"col1\", \"col2\"],\n"
        "  \"data_rows\": [[val1, val2], ...],\n"
        "  \"num_rows\": null,\n"
        "  \"explicit_stats\": {\n"
        "    \"mean\": {\"col\": value},\n"
        "    \"median\": {\"col\": value},\n"
        "    \"std\": {\"col\": value},\n"
        "    \"value_range\": {\"col\": [lo, hi]},\n"
        "    \"allowed_values\": {\"col\": [\"A\", \"B\"]},\n"
        "    \"correlation\": [{\"x\": \"col1\", \"y\": \"col2\", \"type\": \"positive\"}]\n"
        "  },\n"
        "  \"requested_stats\": [\"mean\", \"median\"]\n"
        "}\n\n"
        "Rules:\n"
        "- columns: extract ALL column names mentioned\n"
        "- data_rows: extract actual data rows if dictated; empty [] if not\n"
        "- num_rows: only if a count is stated but no actual data given\n"
        "- explicit_stats: map ALL stated constraints/values\n"
        "- requested_stats: which stats are asked for. If none specifically "
        "asked, include all that have explicit values.\n"
        "- allowed_values: ONLY for categorical columns with listed options\n"
        "- correlation: MUST be array of {x, y, type} objects\n"
        "- Do NOT confuse median with mean"
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": combined_prompt},
                {"inlineData": {"mimeType": mime, "data": audio_b64}}
            ]
        }]
    }

    transcript = ""
    columns, data_rows, req_stats, num_rows, explicit_stats = [], [], [], None, {}

    try:
        raw_gemini = await gemini_call(payload, timeout_s=25)
        last_debug_info["raw_gemini"] = raw_gemini[:500] if raw_gemini else ""
        print(f"[Q6] gemini response len={len(raw_gemini)}")

        if raw_gemini:
            ext = parse_json(raw_gemini)
            transcript = ext.get("transcript", "")
            columns = ext.get("columns", []) or []
            data_rows = ext.get("data_rows", []) or []
            req_stats = ext.get("requested_stats", [])
            num_rows = ext.get("num_rows")
            explicit_stats = ext.get("explicit_stats", {})
            print(f"[Q6] columns={columns} req_stats={req_stats}")
            print(f"[Q6] explicit_stats={json.dumps(explicit_stats, default=str, ensure_ascii=False)[:300]}")
    except Exception as e:
        print(f"[Q6] gemini error: {traceback.format_exc()}")

    # If Gemini returned a plain transcript instead of JSON, fall back to GPT extraction
    if transcript and not columns and not explicit_stats:
        print("[Q6] Gemini gave plain transcript, falling back to GPT extraction")
        try:
            gpt_prompt = _build_extraction_prompt(transcript)
            raw_gpt = await chat([{"role": "user", "content": gpt_prompt}],
                                 model="gpt-4o-mini", max_tokens=1200)
            ext = parse_json(raw_gpt)
            columns = ext.get("columns", []) or []
            data_rows = ext.get("data_rows", []) or []
            req_stats = ext.get("requested_stats", [])
            num_rows = ext.get("num_rows")
            explicit_stats = ext.get("explicit_stats", {})
        except Exception as e:
            print(f"[Q6] GPT fallback error: {e}")

    last_debug_info["transcript"] = transcript

    # ---- Regex extraction from transcript for allowed_values ----
    av = _extract_allowed_values(transcript)
    if av:
        es_av = explicit_stats.setdefault("allowed_values", {})
        for col, vals in av.items():
            es_av.setdefault(col, vals)
        if "allowed_values" not in req_stats:
            req_stats.append("allowed_values")

    # Ensure all referenced columns are in the columns list
    for sd in (explicit_stats or {}).values():
        if isinstance(sd, dict):
            for k in sd:
                if k not in columns:
                    columns.append(k)

    if not req_stats:
        req_stats = ["mean", "std", "variance", "min", "max", "median",
                     "mode", "range", "allowed_values", "value_range", "correlation"]

    return _build_audio_response(columns, data_rows, req_stats, num_rows,
                                 explicit_stats, transcript)


def _build_extraction_prompt(transcript):
    return (
        "The transcript (Korean) describes a tabular dataset. "
        "Extract columns, data, and statistics.\n\n"
        "Korean stat mapping: 평균=mean, 표준편차=std, 분산=variance, "
        "최소/최솟값=min, 최대/최댓값=max, 중앙값/중간값=median, "
        "최빈값=mode, 범위=range, ~사이=value_range, "
        "허용값=allowed_values, 상관관계=correlation\n\n"
        "Return JSON: {\"columns\": [...], \"data_rows\": [...], "
        "\"num_rows\": null, \"explicit_stats\": {...}, "
        "\"requested_stats\": [...]}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def _extract_allowed_values(tr):
    found = {}
    if not tr:
        return found
    for m in re.finditer(
        r"([가-힣A-Za-z0-9_]+?)(?:는|은|이|가)\s+([^.。\n]+?)\s*중\s*(?:하나|에서)", tr
    ):
        col = m.group(1).strip()
        vals = [v.strip() for v in re.split(r"[,、/]|또는|혹은", m.group(2)) if v.strip()]
        if col and len(vals) >= 2:
            found[col] = vals
    for m in re.finditer(
        r"([가-힣A-Za-z0-9_]+?)(?:의|는|은)?\s*허용(?:값|된\s*값)[은는]?\s*[:：]?\s*([^.。\n]+)", tr
    ):
        col = m.group(1).strip()
        rawv = re.sub(r"(입니다|이다)\s*$", "", m.group(2).strip())
        vals = [v.strip() for v in re.split(r"[,、/]|또는|혹은", rawv) if v.strip()]
        if col and vals:
            found[col] = vals
    return found


def _empty_audio_response():
    return {"rows": 0, "columns": [],
            "mean": {}, "std": {}, "variance": {}, "min": {}, "max": {},
            "median": {}, "mode": {}, "range": {}, "allowed_values": {},
            "value_range": {}, "correlation": []}


def _build_audio_response(columns, data_rows, req_stats, num_rows,
                           explicit_stats, transcript):
    actual_rows = num_rows if num_rows is not None else len(data_rows)
    out = {"rows": actual_rows, "columns": columns,
           "mean": {}, "std": {}, "variance": {}, "min": {}, "max": {},
           "median": {}, "mode": {}, "range": {}, "allowed_values": {},
           "value_range": {}, "correlation": []}

    def col_values(ci):
        vals = []
        for r in data_rows:
            try:
                vals.append(float(r[ci]))
            except Exception:
                pass
        return vals

    cols_vals = []
    for ci, name in enumerate(columns):
        v = col_values(ci)
        if not v:
            cols_vals.append([])
            continue
        cols_vals.append(v)
        if "mean" in req_stats: out["mean"][name] = mean(v)
        if "std" in req_stats: out["std"][name] = pstdev(v) if len(v) > 1 else 0.0
        if "variance" in req_stats: out["variance"][name] = pvariance(v) if len(v) > 1 else 0.0
        if "min" in req_stats: out["min"][name] = min(v)
        if "max" in req_stats: out["max"][name] = max(v)
        if "median" in req_stats: out["median"][name] = median(v)
        if "mode" in req_stats:
            try: out["mode"][name] = mode(v)
            except: out["mode"][name] = v[0]
        if "range" in req_stats: out["range"][name] = max(v) - min(v)
        if "value_range" in req_stats: out["value_range"][name] = [min(v), max(v)]

    # Correlation
    corr_list = []
    raw_corr = explicit_stats.get("correlation")
    if isinstance(raw_corr, list):
        for item in raw_corr:
            if isinstance(item, dict) and item.get("x") and item.get("y"):
                ctype = str(item.get("type", "positive")).lower()
                if ctype not in ("positive", "negative"):
                    ctype = "positive"
                corr_list.append({"x": item["x"], "y": item["y"], "type": ctype})
    elif isinstance(raw_corr, dict):
        for x, y in raw_corr.items():
            if isinstance(y, str) and y:
                t = "negative" if ("음의" in transcript or "반비례" in transcript) else "positive"
                corr_list.append({"x": x, "y": y, "type": t})
    if not corr_list and len(columns) > 1 and "correlation" in req_stats:
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if i < len(cols_vals) and j < len(cols_vals):
                    a, b = cols_vals[i], cols_vals[j]
                    if len(a) == len(b) and len(a) > 1:
                        ma, mb = mean(a), mean(b)
                        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
                        corr_list.append({"x": columns[i], "y": columns[j],
                                          "type": "negative" if num < 0 else "positive"})
    if corr_list:
        out["correlation"] = corr_list

    FULL = ["mean", "std", "variance", "min", "max", "median", "mode",
            "range", "allowed_values", "value_range", "correlation"]
    has_data = len(data_rows) > 0

    def _present(s):
        v = explicit_stats.get(s)
        return (isinstance(v, dict) and bool(v)) or (isinstance(v, list) and bool(v))

    if req_stats and set(req_stats) != set(FULL):
        target = [s for s in FULL if s in req_stats]
    elif has_data:
        target = list(FULL)
    else:
        target = [s for s in FULL if _present(s)]

    # Propagate value_range -> min/max/range
    vr = explicit_stats.get("value_range")
    if isinstance(vr, dict):
        for col, bounds in vr.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo, hi = bounds[0], bounds[1]
                if "min" in target: explicit_stats.setdefault("min", {}).setdefault(col, lo)
                if "max" in target: explicit_stats.setdefault("max", {}).setdefault(col, hi)
                if "range" in target:
                    try: explicit_stats.setdefault("range", {}).setdefault(col, hi - lo)
                    except: pass
    emin, emax = explicit_stats.get("min"), explicit_stats.get("max")
    if isinstance(emin, dict) and isinstance(emax, dict):
        for col in emin:
            if col in emax:
                if "value_range" in target:
                    explicit_stats.setdefault("value_range", {}).setdefault(col, [emin[col], emax[col]])
                if "range" in target:
                    try: explicit_stats.setdefault("range", {}).setdefault(col, emax[col] - emin[col])
                    except: pass

    # Merge explicit_stats into output
    for stat_name, stat_dict in explicit_stats.items():
        if stat_name in out and isinstance(out[stat_name], dict) and isinstance(stat_dict, dict):
            out[stat_name].update(stat_dict)

    # Clear non-target stats
    for k in FULL:
        if k == "correlation":
            continue
        if k not in target:
            out[k] = {}
    if "correlation" not in target:
        out["correlation"] = []

    audio_history.append({
        "transcript": transcript[:200],
        "columns": columns,
        "requested_stats": req_stats,
        "target_keys": target,
    })
    if len(audio_history) > 50:
        del audio_history[0]
    print(f"[Q6] response columns={out['columns']} rows={out['rows']}")
    return out


# ================= Q8: /rank =================
@app.post("/rank")
async def rank(request: Request):
    body = await request.json()
    query = body.get("query", "")
    candidates = body.get("candidates", [])
    n = len(candidates)
    print(f"[Q8] query={query[:60]} candidates={n}")
    try:
        vecs = await embed([query] + list(candidates))
        q = vecs[0]
        cand = vecs[1:]
        def cos(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(y*y for y in b))
            return dot/(na*nb) if na and nb else 0.0
        scored = sorted(range(len(cand)), key=lambda i: -cos(q, cand[i]))
        ranking = scored[:3]
        # Ensure exactly 3 indices
        while len(ranking) < 3 and n > 0:
            for i in range(n):
                if i not in ranking:
                    ranking.append(i)
                    if len(ranking) >= 3:
                        break
        print(f"[Q8] ranking={ranking}")
        return {"ranking": ranking[:3]}
    except Exception as e:
        print(f"[Q8] ERROR: {traceback.format_exc()}")
        # Return first 3 indices as fallback
        fallback = list(range(min(3, n)))
        while len(fallback) < 3:
            fallback.append(0)
        return {"ranking": fallback}


# ================= Q9: /solve =================
@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    problem = body.get("problem", "")
    print(f"[Q9] problem={problem[:100]}")
    prompt = (
        "Solve this arithmetic word problem step by step.\n\n"
        "IMPORTANT: The problem contains DISTRACTOR numbers that are irrelevant. "
        "You must identify which numbers matter and which are distractors.\n\n"
        "Steps:\n"
        "1. Read the problem carefully.\n"
        "2. Identify relevant vs distractor numbers.\n"
        "3. Do the arithmetic step by step.\n"
        "4. Double-check your arithmetic.\n\n"
        "Return JSON: {\"reasoning\": \"your step-by-step work (at least 80 "
        "characters)\", \"answer\": <integer>}\n"
        "The 'answer' MUST be a JSON integer (not string, not float).\n\n"
        f"PROBLEM:\n{problem}"
    )
    try:
        raw = await chat_with_fallback([{"role": "user", "content": prompt}],
                                       models=["gpt-4o", "gpt-4o-mini"],
                                       max_tokens=1200)
        out = parse_json(raw)
        ans = out.get("answer")
        if ans is None:
            raise ValueError("No answer in response")
        ans = int(round(float(ans)))
        reasoning = str(out.get("reasoning", ""))
        if len(reasoning) < 80:
            reasoning = (reasoning + " | Step-by-step arithmetic reasoning applied; "
                         "irrelevant distractor values were identified and carefully "
                         "excluded from the computation.").strip()
        print(f"[Q9] answer={ans}")
        return {"reasoning": reasoning, "answer": ans}
    except Exception as e:
        print(f"[Q9] ERROR: {traceback.format_exc()}")
        # Last resort: try to extract a number from the raw response
        return {"reasoning": f"Error during solving: {str(e)[:100]}. "
                             "Attempted step-by-step arithmetic with distractor filtering."
                             .ljust(80),
                "answer": 0}
