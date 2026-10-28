"""
Modul AI Model Service untuk Klasifikasi Risiko Diabetes (Groq Cloud API).
Dipakai saat evaluasi (Modul 2A) dan penyajian antarmuka web (Modul 2B).
"""
import json
import re
import time
import groq

NAMA_FITUR = {
    "Pregnancies": "Jumlah Kehamilan",
    "Glucose": "Kadar Glukosa Plasma (mg/dL)",
    "BloodPressure": "Tekanan Darah Diastolik (mm Hg)",
    "SkinThickness": "Ketebalan Lipatan Kulit (mm)",
    "Insulin": "Serum Insulin (mu U/ml)",
    "BMI": "Indeks Massa Tubuh (BMI)",
    "DiabetesPedigreeFunction": "Skor Riwayat Genetik (Pedigree)",
    "Age": "Usia Pasien (tahun)"
}

def load_config(path="studi-kasus-diabetes/config_diabetes.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def format_data(data):
    """Konversi dict data medis pasien menjadi format teks terstruktur klinis."""
    items = []
    for col, nama in NAMA_FITUR.items():
        val = data.get(col, "N/A")
        items.append(f"{nama}: {val}")
    return "; ".join(items)

def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs

def parse_label(output, labels):
    """Menguraikan respon model bahasa menjadi salah satu label baku: 'Positif' atau 'Negatif'."""
    if not output:
        return "invalid"
    raw = output.strip().lower()
    for lbl in labels:
        if lbl.lower() in raw:
            return lbl
    # Regex fallback
    words = re.findall(r"\b[a-zA-Z]+\b", output)
    peta = {l.lower(): l for l in labels}
    for w in words:
        if w.lower() in peta:
            return peta[w.lower()]
    return "invalid"

def classify(client, cfg, data, max_retry=5):
    """Melakukan inferensi klasifikasi risiko diabetes via Groq API."""
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg.get("max_tokens", 10),
                messages=build_messages(cfg, data),
            )
            content = resp.choices[0].message.content or ""
            return parse_label(content, cfg["labels"])
        except groq.RateLimitError:
            wait_time = 12 * (percobaan + 1)
            print(f"[RateLimit] Menunggu {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[API Error] {e}, mencoba ulang...")
            time.sleep(3 * (percobaan + 1))
    return "invalid"
