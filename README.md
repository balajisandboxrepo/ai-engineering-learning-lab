# Python and LLM learning samples

Small Python learning projects for OpenAI with LangChain, Scikit-LLM, and Qwen3.8-27B. Use [uv](https://docs.astral.sh/uv/) to manage the environments.

## Requirements

- Python 3.13 (the examples were verified with Python 3.13)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An OpenAI API key for `aiapplication.py`, `aisummarizer`, and the Scikit-LLM examples
- A separately hosted Qwen-compatible endpoint to run the Qwen examples; the OpenAI key is not a Qwen credential

## Configure credentials

Copy `.env.example` to `.env` and add the credentials required for the examples you intend to run. `.env` is ignored by Git; never commit credentials or put them in source files.

```powershell
Copy-Item .env.example .env
```

OpenAI-backed examples send prompts to OpenAI and may incur API charges. The Qwen examples send invoice text to the Qwen endpoint configured by `QWEN_BASE_URL`; do not use real or sensitive invoices in these samples.

## OpenAI/LangChain environment

Create and install the root environment:

```powershell
uv venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r .\requirements.txt
```

Run the five-line India sample:

```powershell
.\.venv\Scripts\python.exe .\aiapplication.py
```

Run the article summarizer:

```powershell
.\.venv\Scripts\python.exe -m aisummarizer
```

The summarizer fetches the configured KDnuggets article and sends its text to OpenAI. Review its privacy and usage implications before running it.

## Scikit-LLM examples

Scikit-LLM requires an OpenAI SDK version that conflicts with the root LangChain environment, so it has a separate environment and requirements file. Follow [`scikit_llm_learn/README.md`](scikit_llm_learn/README.md) for installation and its zero-shot classification and custom-entity NER examples.

## Qwen3.8-27B examples

The Qwen examples use an isolated environment and a configurable OpenAI-compatible Qwen endpoint; they do not use `OPENAI_API_KEY`. Follow [`qwen_learn/README.md`](qwen_learn/README.md) to start the endpoint and run the two sample invoice extraction examples.

## Run offline unit tests

The Qwen package tests use mocked responses and do not call a model provider:

```powershell
.\.venv-qwen-learn\Scripts\python.exe -m unittest discover -s .\qwen_learn\tests -v
```

Check dependency consistency for each isolated environment:

```powershell
uv pip check --python .\.venv\Scripts\python.exe
uv pip check --python .\.venv-scikit-llm-learn\Scripts\python.exe
uv pip check --python .\.venv-qwen-learn\Scripts\python.exe
```
"# ai-engineering-learning-lab" 
