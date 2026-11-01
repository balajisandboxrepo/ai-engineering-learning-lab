import json
from pathlib import Path

from qwen_learn.extractor import WORKSPACE_ROOT, extract_invoice


def main() -> None:
    document = WORKSPACE_ROOT / "qwen_learn" / "sample_invoices" / "invoice_basic.txt"
    invoice = extract_invoice(document)
    print(json.dumps(invoice.model_dump(), indent=2))


if __name__ == "__main__":
    main()
