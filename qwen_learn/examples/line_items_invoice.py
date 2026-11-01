import json
from decimal import Decimal
from pathlib import Path

from qwen_learn.extractor import WORKSPACE_ROOT, extract_invoice


def main() -> None:
    document = (
        WORKSPACE_ROOT / "qwen_learn" / "sample_invoices" / "invoice_line_items.txt"
    )
    invoice = extract_invoice(document)

    if invoice.subtotal is not None and invoice.line_items:
        if any(item.amount is None for item in invoice.line_items):
            print("Subtotal check skipped: a line-item amount is missing.")
        else:
            items_total = sum(
                (
                    Decimal(item.amount.replace(",", ""))
                    for item in invoice.line_items
                    if item.amount is not None
                ),
                start=Decimal("0"),
            )
            subtotal = Decimal(invoice.subtotal.replace(",", ""))
            print(f"Extracted line-item total: {items_total:.2f}")
            print(f"Invoice subtotal: {subtotal:.2f}")
            print(f"Line items match subtotal: {items_total == subtotal}")

    print(json.dumps(invoice.model_dump(), indent=2))


if __name__ == "__main__":
    main()
