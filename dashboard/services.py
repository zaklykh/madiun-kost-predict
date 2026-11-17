import base64
import json
from functools import lru_cache
from io import BytesIO

import pandas as pd
from django.conf import settings
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from groq import Groq
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# ================= PENGATURAN =================
SKALA_HARGA = 1000   # kolom 'harga' dalam ribu rupiah (600 = Rp 600.000)
FITUR_MODEL = ["jarak_km", "rating", "ac_num"]
MIN_DATA = 6
# ==============================================


def rp(harga):
    """Format nilai kolom 'harga' (ribuan) menjadi Rupiah."""
    return "Rp " + f"{harga * SKALA_HARGA:,.0f}".replace(",", ".")


@lru_cache(maxsize=1)
def _muat_csv():
    try:
        df = pd.read_csv(settings.KOS_CSV_URL)
    except Exception:
        df = pd.read_csv(settings.KOS_CSV_LOCAL)
    df = df.dropna(subset=["nama", "jarak_km", "harga", "ac", "rating"]).copy()
    df["ac"] = df["ac"].astype(str).str.strip().str.lower()
    df["ac_num"] = (df["ac"] == "ya").astype(int)
    return df


def muat_data():
    return _muat_csv().copy()  # restart server jika CSV di GitHub diganti


def _angka(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def baca_filter(get):
    return {
        "ac": get.get("ac", "").strip(),
        "harga_min": _angka(get.get("harga_min")),
        "harga_max": _angka(get.get("harga_max")),
        "jarak_max": _angka(get.get("jarak_max")),
        "rating_min": _angka(get.get("rating_min")),
        "kategori": get.getlist("kategori"),
        "jarak_baru": _angka(get.get("jarak_baru"), 1.0),
        "rating_baru": _angka(get.get("rating_baru"), 4.0),
        "ac_baru": 1 if get.get("ac_baru") == "ya" else 0,
    }


def batas_harga():
    return float(muat_data()["harga"].median())


def prediksi_kategori(df):
    data = muat_data()
    batas = batas_harga()
    kelas = (data["harga"] <= batas).map({True: "MURAH", False: "MAHAL"})
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(data[FITUR_MODEL], kelas)
    return model.predict(df[FITUR_MODEL])


def terapkan_filter(f):
    df = muat_data()
    if f["ac"] in ("ya", "tidak"):
        df = df[df["ac"] == f["ac"]]
    if f["harga_min"] is not None:
        df = df[df["harga"] >= f["harga_min"] / SKALA_HARGA]
    if f["harga_max"] is not None:
        df = df[df["harga"] <= f["harga_max"] / SKALA_HARGA]
    if f["jarak_max"] is not None:
        df = df[df["jarak_km"] <= f["jarak_max"]]
    if f["rating_min"] is not None:
        df = df[df["rating"] >= f["rating_min"]]
    df["kategori"] = prediksi_kategori(df) if not df.empty else pd.Series(dtype=str)
    kategori = [v for v in f["kategori"] if v in ("MURAH", "MAHAL")]
    if kategori:
        df = df[df["kategori"].isin(kategori)]
    return df


def _ke_png(fig):
    buf = BytesIO()
    FigureCanvasAgg(fig)
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _format_harga(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"Rp {v / 1e6:.1f} jt"))


def _plot_harga(df, batas):
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    ax.hist(df["harga"] * SKALA_HARGA, bins=8, color="#6480f5", edgecolor="white")
    ax.axvline(batas * SKALA_HARGA, color="#e85d5d", linestyle="--",
               linewidth=2, label=f"Batas {rp(batas)}")
    ax.set(title="Sebaran harga kos", xlabel="Harga per bulan", ylabel="Jumlah kos")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f} jt"))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    return _ke_png(fig)


def _plot_jarak(df):
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    warna = {"ya": "#4263eb", "tidak": "#e79932"}
    for ac in ("ya", "tidak"):
        subset = df[df["ac"] == ac]
        ax.scatter(subset["jarak_km"], subset["harga"] * SKALA_HARGA,
                   color=warna[ac], label="Dengan AC" if ac == "ya" else "Tanpa AC",
                   alpha=.82, edgecolors="white", linewidths=.6)
    ax.set(title="Jarak dan harga", xlabel="Jarak (km)", ylabel="Harga per bulan")
    _format_harga(ax)
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    return _ke_png(fig)


def _plot_rating(df):
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    ax.scatter(df["rating"], df["harga"] * SKALA_HARGA, color="#14a887",
               alpha=.82, edgecolors="white", linewidths=.6)
    ax.set(title="Rating dan harga", xlabel="Rating", ylabel="Harga per bulan")
    _format_harga(ax)
    ax.set_xlim(1, 5)
    ax.grid(alpha=.2)
    return _ke_png(fig)


def _plot_ac(df):
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    groups = [df.loc[df["ac"] == ac, "harga"] * SKALA_HARGA for ac in ("tidak", "ya")]
    box = ax.boxplot(groups, patch_artist=True, tick_labels=["Tanpa AC", "Dengan AC"])
    for patch, color in zip(box["boxes"], ("#f0aa4b", "#6480f5")):
        patch.set_facecolor(color)
        patch.set_alpha(.75)
    ax.set(title="Pengaruh fasilitas AC", ylabel="Harga per bulan")
    _format_harga(ax)
    ax.grid(axis="y", alpha=.2)
    return _ke_png(fig)


def _plot_importance(model):
    nama = {"jarak_km": "Jarak", "rating": "Rating", "ac_num": "Fasilitas AC"}
    pasangan = sorted(zip(model.feature_importances_, FITUR_MODEL))
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    nilai = [item[0] for item in pasangan]
    label = [nama[item[1]] for item in pasangan]
    bars = ax.barh(label, nilai, color=["#9aacef", "#7188ed", "#4263eb"])
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set(title="Faktor paling berpengaruh", xlabel="Feature importance", xlim=(0, 1))
    ax.grid(axis="x", alpha=.2)
    return _ke_png(fig)


def _plot_confusion(actual, predicted):
    classes = ["MURAH", "MAHAL"]
    matrix = [
        [sum(1 for a, p in zip(actual, predicted) if a == kelas_asli and p == kelas_prediksi)
         for kelas_prediksi in classes]
        for kelas_asli in classes
    ]
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Confusion matrix (validasi silang)", xlabel="Prediksi", ylabel="Aktual",
           xticks=range(2), yticks=range(2), xticklabels=classes, yticklabels=classes)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center",
                    color="white" if matrix[row][col] > max(map(max, matrix)) / 2 else "#17243b",
                    fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    return _ke_png(fig)


def _plot_posisi(df, jarak, rating, ac, harga, kategori, batas):
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    warna = {"MURAH": "#14a887", "MAHAL": "#e79932"}
    for kelas in ("MURAH", "MAHAL"):
        subset = df[df["kategori"] == kelas]
        ax.scatter(subset["jarak_km"], subset["harga"] * SKALA_HARGA,
                   color=warna[kelas], label=kelas.title(), alpha=.72,
                   edgecolors="white", linewidths=.5)
    ax.scatter([jarak], [harga * SKALA_HARGA], marker="*", s=260, color="#d83f87",
               edgecolors="white", linewidths=1.2, label=f"Kos baru ({kategori.title()})", zorder=5)
    ax.axhline(batas * SKALA_HARGA, color="#8993a5", linestyle="--", linewidth=1)
    ax.set(title=f"Posisi kos baru · rating {rating:.1f} · AC {'ya' if ac else 'tidak'}",
           xlabel="Jarak (km)", ylabel="Harga per bulan")
    _format_harga(ax)
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    return _ke_png(fig)


def _hasil_validasi_silang(data, aktual):
    hasil = []
    for i, row in data.iterrows():
        latih = data.drop(index=i)
        label_latih = aktual.drop(index=i)
        hasil.append(DecisionTreeClassifier(max_depth=3, random_state=42)
                     .fit(latih[FITUR_MODEL], label_latih)
                     .predict(row[FITUR_MODEL].to_frame().T)[0])
    return hasil


def _korelasi(x, y):
    nilai = x.corr(y)
    return None if pd.isna(nilai) else round(float(nilai), 2)


def _ringkasan_boxplot(df):
    ringkasan = {}
    for ac in ("ya", "tidak"):
        harga = df.loc[df["ac"] == ac, "harga"]
        ringkasan["Dengan AC" if ac == "ya" else "Tanpa AC"] = {
            "jumlah": len(harga),
            "kuartil_bawah": rp(harga.quantile(.25)) if not harga.empty else "-",
            "median": rp(harga.median()) if not harga.empty else "-",
            "kuartil_atas": rp(harga.quantile(.75)) if not harga.empty else "-",
        }
    return ringkasan


def _konteks_grafik(df, f):
    data = muat_data()
    batas = batas_harga()
    aktual = (data["harga"] <= batas).map({True: "MURAH", False: "MAHAL"})
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(data[FITUR_MODEL], aktual)
    hasil_cv = _hasil_validasi_silang(data, aktual)
    baru = pd.DataFrame({
        "jarak_km": [f["jarak_baru"]],
        "rating": [f["rating_baru"]],
        "ac_num": [f["ac_baru"]],
    })
    harga_baru = max(0.0, float(LinearRegression().fit(
        data[FITUR_MODEL], data["harga"]).predict(baru)[0]))
    kelas_baru = str(model.predict(baru)[0])
    hitung_konfusi = {
        "aktual_murah_prediksi_murah": sum(a == "MURAH" and p == "MURAH" for a, p in zip(aktual, hasil_cv)),
        "aktual_murah_prediksi_mahal": sum(a == "MURAH" and p == "MAHAL" for a, p in zip(aktual, hasil_cv)),
        "aktual_mahal_prediksi_murah": sum(a == "MAHAL" and p == "MURAH" for a, p in zip(aktual, hasil_cv)),
        "aktual_mahal_prediksi_mahal": sum(a == "MAHAL" and p == "MAHAL" for a, p in zip(aktual, hasil_cv)),
    }
    grup_ac = {}
    for ac, label in (("ya", "Dengan AC"), ("tidak", "Tanpa AC")):
        harga = df.loc[df["ac"] == ac, "harga"]
        grup_ac[label] = {"jumlah": len(harga), "rata_rata": rp(harga.mean()) if not harga.empty else "-"}

    return {
        "harga": {
            "jumlah_kos": len(df),
            "harga_terendah": rp(df["harga"].min()),
            "harga_tertinggi": rp(df["harga"].max()),
            "harga_rata_rata": rp(df["harga"].mean()),
            "di_bawah_batas": int((df["harga"] <= batas).sum()),
            "di_atas_batas": int((df["harga"] > batas).sum()),
            "batas_kategori": rp(batas),
        },
        "jarak": {
            "jumlah_kos": len(df),
            "rentang_jarak_km": [float(df["jarak_km"].min()), float(df["jarak_km"].max())],
            "korelasi_jarak_dan_harga": _korelasi(df["jarak_km"], df["harga"]),
            "jumlah_dan_rata_rata_harga_per_ac": grup_ac,
        },
        "rating": {
            "jumlah_kos": len(df),
            "rentang_rating": [float(df["rating"].min()), float(df["rating"].max())],
            "korelasi_rating_dan_harga": _korelasi(df["rating"], df["harga"]),
        },
        "ac": _ringkasan_boxplot(df),
        "importance": {
            "Jarak": round(float(model.feature_importances_[FITUR_MODEL.index("jarak_km")]), 2),
            "Rating": round(float(model.feature_importances_[FITUR_MODEL.index("rating")]), 2),
            "Fasilitas AC": round(float(model.feature_importances_[FITUR_MODEL.index("ac_num")]), 2),
        },
        "confusion": {
            **hitung_konfusi,
            "akurasi_validasi_silang": f"{sum(a == p for a, p in zip(aktual, hasil_cv)) / len(aktual):.1%}",
        },
        "posisi": {
            "jarak_kos_baru_km": f["jarak_baru"],
            "rating_kos_baru": f["rating_baru"],
            "ac_kos_baru": "Dengan AC" if f["ac_baru"] else "Tanpa AC",
            "estimasi_harga": rp(harga_baru),
            "kategori_prediksi": kelas_baru,
            "batas_kategori": rp(batas),
            "rentang_harga_kos_terfilter": [rp(df["harga"].min()), rp(df["harga"].max())],
        },
    }


def jelaskan_grafik(df, f):
    konteks = _konteks_grafik(df, f)
    kunci = ["harga", "jarak", "rating", "ac", "importance", "confusion", "posisi"]
    prompt = (
        "Buat satu penjelasan bahasa Indonesia untuk setiap grafik. Maksimal dua kalimat singkat "
        "per grafik, gunakan bahasa orang awam. Jelaskan cara membaca dan arti angka yang diberikan. "
        "Jangan mengarang angka, jangan menyimpulkan sebab-akibat dari korelasi, dan jangan menyebut "
        "faktor penting sebagai penyebab harga. Untuk confusion matrix, jelaskan jumlah tebakan benar/salah "
        "dan akurasi secara jujur. Kembalikan hanya JSON object dengan tepat tujuh kunci berikut: "
        + ", ".join(kunci)
        + ". Setiap nilai harus berupa string. Data grafik: "
        + json.dumps(konteks, ensure_ascii=False)
    )
    respon = Groq(api_key=settings.GROQ_API_KEY).chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "Kamu menjelaskan grafik analisis kos kepada pengguna awam secara ringkas dan akurat."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        reasoning_effort="low",
        max_completion_tokens=1200,
    )
    penjelasan = json.loads(respon.choices[0].message.content or "")
    if not isinstance(penjelasan, dict) or any(
        not isinstance(penjelasan.get(nama), str) for nama in kunci
    ):
        raise ValueError("Respons Groq tidak memuat penjelasan valid untuk semua grafik.")
    return {nama: penjelasan[nama].strip() for nama in kunci}


def analisis_visual(df, f):
    data = muat_data()
    batas = batas_harga()
    aktual = (data["harga"] <= batas).map({True: "MURAH", False: "MAHAL"})
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(data[FITUR_MODEL], aktual)
    hasil_cv = _hasil_validasi_silang(data, aktual)

    baru = pd.DataFrame({"jarak_km": [f["jarak_baru"]],
                         "rating": [f["rating_baru"]],
                         "ac_num": [f["ac_baru"]]})
    prediksi_harga = max(0.0, float(LinearRegression().fit(
        data[FITUR_MODEL], data["harga"]).predict(baru)[0]))
    prediksi_kelas = str(model.predict(baru)[0])
    gambar = {
        "harga": _plot_harga(df, batas),
        "jarak": _plot_jarak(df),
        "rating": _plot_rating(df),
        "ac": _plot_ac(df),
        "importance": _plot_importance(model),
        "confusion": _plot_confusion(aktual.to_numpy(), hasil_cv),
        "posisi": _plot_posisi(df, f["jarak_baru"], f["rating_baru"], f["ac_baru"],
                               prediksi_harga, prediksi_kelas, batas),
    }
    return {
        "gambar": gambar,
        "ringkas": {
            "batas": rp(batas),
            "prediksi_harga": rp(prediksi_harga),
            "prediksi_kategori": prediksi_kelas,
            "akurasi": f"{sum(a == p for a, p in zip(aktual, hasil_cv)) / len(aktual):.1%}",
        },
    }


def df_ke_records(df):
    return json.loads(df.to_json(orient="records"))