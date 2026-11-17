import json
import logging

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from groq import GroqError

from . import services as s

logger = logging.getLogger(__name__)


@require_GET
def index(request):
    df = s.muat_data()
    return render(request, "dashboard/index.html", {
        "total": len(df),
        "harga_min": int(df["harga"].min() * s.SKALA_HARGA),
        "harga_max": int(df["harga"].max() * s.SKALA_HARGA),
        "jarak_min": float(df["jarak_km"].min()),
        "jarak_maks": round(float(df["jarak_km"].max()), 1),
        "rating_min": float(df["rating"].min()),
        "rating_maks": float(df["rating"].max()),
    })


def _hitung(request):
    f = s.baca_filter(request.GET)
    df = s.terapkan_filter(f)
    if len(df) < s.MIN_DATA:
        return f, df, f"Hanya {len(df)} kos cocok dengan filter (minimal {s.MIN_DATA}). Longgarkan filter."
    return f, df, None


@require_GET
def api_analisis(request):
    f, df, error = _hitung(request)
    if df.empty:
        return JsonResponse({"ok": False, "pesan": error or "Tidak ada kos yang cocok."})

    tabel = s.df_ke_records(
        df.sort_values("harga")[["nama", "jarak_km", "harga", "ac", "rating", "kategori"]])
    for baris in tabel:
        baris["harga_rp"] = s.rp(baris["harga"])

    ringkasan = {
        "jumlah": len(df),
        "rata": s.rp(df["harga"].mean()),
        "termurah": s.rp(df["harga"].min()),
        "termahal": s.rp(df["harga"].max()),
    }
    if error:
        return JsonResponse({
            "ok": True,
            "pesan": error,
            "ringkasan": ringkasan,
            "visualisasi": None,
            "tabel": tabel,
        })

    visualisasi = s.analisis_visual(df, f)
    return JsonResponse({
        "ok": True,
        "ringkasan": ringkasan,
        "visualisasi": visualisasi,
        "tabel": tabel,
    })


@require_GET
def api_jelaskan(request):
    f, df, error = _hitung(request)
    if error:
        return JsonResponse({"ok": False, "pesan": error}, status=422)
    if not settings.GROQ_API_KEY:
        return JsonResponse({
            "ok": False,
            "pesan": "GROQ_API_KEY belum dikonfigurasi.",
        }, status=503)

    try:
        penjelasan = s.jelaskan_grafik(df, f)
    except (GroqError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Penjelasan grafik dari Groq gagal (%s).", type(exc).__name__)
        return JsonResponse({
            "ok": False,
            "pesan": "Penjelasan AI gagal dibuat. Periksa konfigurasi Groq lalu coba lagi.",
        }, status=502)
    return JsonResponse({"ok": True, "penjelasan": penjelasan})