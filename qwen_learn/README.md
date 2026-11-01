# Qwen3.8-27B: basics to AI solution architecture

This package teaches Qwen3.8-27B through two invoice-extraction examples. It loads sample invoice text files, sends their contents to a Qwen-compatible chat endpoint, and validates the returned fields as a typed invoice object.

## What is Qwen3.8-27B?

Qwen3.8-27B is an open-weight, 27-billion-parameter model in the Qwen family. The official model card describes it as a causal language model with a vision encoder: it can work with text and visual inputs, including images and video. Its published native context is 262,144 tokens, with longer contexts described by the model provider. A large context limit does not mean every request should use it; context increases memory, latency, and cost.

The `27B` suffix refers to the model's parameter count, not a GPU size or a quality score. At BF16, just the weights require roughly 54 GB decimal before runtime and context-cache memory. Quantization can reduce weight memory with potential quality tradeoffs. Actual hardware requirements depend on the serving engine, quantization, concurrency, and context length.

Qwen is the model; Transformers, vLLM, and SGLang are examples of inference/serving software. An OpenAI-compatible API is an interface format and does not mean the request is sent to OpenAI.

## Architecture view

1. **Document intake:** the examples load UTF-8 `.txt` invoices. Production PDF and image invoices need a parser or OCR/layout step before text-only extraction; Qwen's visual capability may instead be used with an endpoint that accepts image inputs.
2. **Inference:** the client sends the invoice text and extraction instructions to a Qwen-serving endpoint. The default is a local vLLM-compatible endpoint at `http://127.0.0.1:8000/v1`.
3. **Structured extraction:** the request asks for JSON. Pydantic validates the fields and rejects unexpected keys. Monetary values stay strings to avoid rounding or changing how the source document wrote them.
4. **Business validation:** extraction is not accounting approval. Validate totals, currencies, supplier identity, duplicate invoices, and policy rules in deterministic application code.
5. **Operations and governance:** protect invoice data and credentials, restrict network egress, monitor latency/errors/token use, evaluate against labeled invoices, and route uncertain or high-impact cases to human review.

## Setup

The workspace's existing OpenAI SDK version is incompatible with some local model libraries, so use this package's isolated virtual environment:

```powershell
uv venv .venv-qwen-learn
uv pip install --python .\.venv-qwen-learn\Scripts\python.exe -r .\qwen_learn\requirements.txt
```

The current project `.env` contains `OPENAI_API_KEY`, which authenticates with OpenAI and is **not** a Qwen credential. This package deliberately does not use or send that key. For a remote Qwen-compatible service, set its endpoint and provider key in the project-root `.env`:

```dotenv
QWEN_BASE_URL=https://your-qwen-provider.example/v1
QWEN_API_KEY=your-provider-key
QWEN_MODEL=Qwen/Qwen3.8-27B
```

For a local endpoint, the defaults are `http://127.0.0.1:8000/v1`, a placeholder API key of `local`, and model id `Qwen/Qwen3.8-27B`.

### Optional local serving with vLLM

Qwen's model card lists vLLM as a supported serving framework. Start the server on a suitable Linux/CUDA host (Windows users can use a supported Linux/WSL environment; vLLM is not installed by this package):

```bash
vllm serve Qwen/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 8192
```

The first launch downloads model weights and needs substantial disk, GPU memory, and bandwidth. For a local-only machine, bind to `127.0.0.1` rather than exposing the inference endpoint publicly. Check current vLLM and Qwen model documentation for compatible versions and hardware options.

## Example 1: extract basic invoice fields

```powershell
.\.venv-qwen-learn\Scripts\python.exe -m qwen_learn.examples.basic_invoice
```

Loads [`invoice_basic.txt`](sample_invoices/invoice_basic.txt) and extracts supplier, invoice identifiers/dates, currency, totals, and line items as JSON.

## Example 2: extract and check line items

```powershell
.\.venv-qwen-learn\Scripts\python.exe -m qwen_learn.examples.line_items_invoice
```

Loads [`invoice_line_items.txt`](sample_invoices/invoice_line_items.txt), extracts multiple line items, and compares their extracted sum with the stated subtotal. This comparison is an example check; it does not replace full tax, rounding, or accounting validation.

## Tests

Unit tests use a mocked API response, so they check document loading, request configuration, and schema validation without downloading Qwen or calling a provider:

```powershell
.\.venv-qwen-learn\Scripts\python.exe -m unittest discover -s .\qwen_learn\tests -v
```

## References

- [Official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Official Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8)
- [vLLM OpenAI-compatible serving documentation](https://docs.vllm.ai/en/latest/serving/online_serving/)
