import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from qwen_learn.extractor import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    WORKSPACE_ROOT,
    extract_invoice,
    load_invoice_text,
)


class InvoiceExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.basic_invoice = (
            WORKSPACE_ROOT / "qwen_learn" / "sample_invoices" / "invoice_basic.txt"
        )

    def test_loads_sample_invoice(self) -> None:
        text = load_invoice_text(self.basic_invoice)
        self.assertIn("NS-1048", text)
        self.assertIn("Total due:                  1,620.00", text)

    def test_extracts_validated_invoice_via_configured_endpoint(self) -> None:
        invoice_data = {
            "supplier_name": "Northstar Office Supplies",
            "invoice_number": "NS-1048",
            "invoice_date": "2026-10-12",
            "due_date": "2026-11-11",
            "currency": "USD",
            "subtotal": "1,500.00",
            "tax": "120.00",
            "total": "1,620.00",
            "line_items": [
                {
                    "description": "Ergonomic desk chair",
                    "quantity": "1",
                    "unit_price": "1,250.00",
                    "amount": "1,250.00",
                }
            ],
        }
        response = unittest.mock.MagicMock()
        response.choices[0].message.content = json.dumps(invoice_data)

        with (
            patch("qwen_learn.extractor.load_dotenv"),
            patch("qwen_learn.extractor.OpenAI") as openai_client,
        ):
            openai_client.return_value.chat.completions.create.return_value = response
            invoice = extract_invoice(self.basic_invoice)

        self.assertEqual(invoice.invoice_number, "NS-1048")
        self.assertEqual(invoice.total, "1,620.00")
        openai_client.assert_called_once_with(
            api_key="local", base_url=DEFAULT_BASE_URL
        )
        request = openai_client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], DEFAULT_MODEL)
        self.assertEqual(request["response_format"], {"type": "json_object"})

    def test_rejects_unexpected_extracted_fields(self) -> None:
        response = unittest.mock.MagicMock()
        response.choices[0].message.content = json.dumps(
            {"invoice_number": "NS-1048", "unrequested_field": "ignore"}
        )

        with (
            patch("qwen_learn.extractor.load_dotenv"),
            patch("qwen_learn.extractor.OpenAI") as openai_client,
        ):
            openai_client.return_value.chat.completions.create.return_value = response
            with self.assertRaises(ValidationError):
                extract_invoice(self.basic_invoice)

    def test_rejects_unsupported_document_type(self) -> None:
        with self.assertRaises(ValueError):
            load_invoice_text(Path("invoice.pdf"))


if __name__ == "__main__":
    unittest.main()
