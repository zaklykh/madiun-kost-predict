import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from . import services


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "kos_madiun.csv"


@override_settings(
    KOS_CSV_URL="file:///C:/missing-kos-dataset.csv",
    KOS_CSV_LOCAL=DATA_PATH,
)
class GroqChartExplanationTests(TestCase):
    def setUp(self):
        services._muat_csv.cache_clear()
        self.params = {"kategori": ["MURAH", "MAHAL"]}

    def test_dashboard_has_explanation_area_for_every_chart(self):
        response = self.client.get(reverse("index"))

        self.assertEqual(response.status_code, 200)
        for name in ("harga", "jarak", "rating", "ac", "importance", "confusion", "posisi"):
            self.assertContains(response, f'id="ai-{name}"')

    @override_settings(GROQ_API_KEY="test-key")
    @patch("dashboard.services.Groq")
    def test_returns_explanations_for_all_charts(self, groq_client):
        names = ["harga", "jarak", "rating", "ac", "importance", "confusion", "posisi"]
        content = json.dumps({name: f"Penjelasan {name}" for name in names})
        groq_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

        response = self.client.get(reverse("api_jelaskan"), self.params)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()["penjelasan"]), set(names))
        groq_client.return_value.chat.completions.create.assert_called_once()

    @override_settings(GROQ_API_KEY="")
    @patch("dashboard.services.Groq")
    def test_reports_missing_api_key(self, groq_client):
        response = self.client.get(reverse("api_jelaskan"), self.params)

        self.assertEqual(response.status_code, 503)
        self.assertIn("GROQ_API_KEY", response.json()["pesan"])
        groq_client.assert_not_called()

    @override_settings(GROQ_API_KEY="test-key")
    @patch("dashboard.services.Groq")
    def test_reports_invalid_groq_response(self, groq_client):
        groq_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not JSON"))]
        )

        response = self.client.get(reverse("api_jelaskan"), self.params)

        self.assertEqual(response.status_code, 502)
        self.assertIn("Penjelasan AI gagal", response.json()["pesan"])
