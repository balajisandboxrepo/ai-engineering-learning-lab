# QAAgent

Question-answering over local PDF, Markdown, and text files, using OpenAI
embeddings and a GPT chat model with a persistent Chroma index.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key with access to the configured chat and embedding models

## Setup

From the repository root, copy the example environment file and add your key:

```powershell
Copy-Item .\package_qaagent\.env.example .\package_qaagent\.env
```

Open `package_qaagent\.env` and replace the placeholder value. Alternatively,
set `OPENAI_API_KEY` in your environment. Never commit a real API key.

Install the project dependencies:

```powershell
uv sync --project .\package_qaagent
```

Put files you are authorized to process in `package_qaagent\docs`. Supported
formats are PDF, Markdown, and plain text. Then start the agent:

```powershell
uv run --project .\package_qaagent python .\package_qaagent\src\qaagent\qaAgent.py
```

Ask questions at the `You:` prompt and enter `exit` to quit. The agent calls
OpenAI for embeddings and chat responses; API usage may incur charges. It
creates `db_openai` beside this README and reuses that index on later runs.
Delete that local index to rebuild it after changing documents or chunking
settings.

## Tests

The unit tests mock OpenAI clients and do not make API requests:

```powershell
uv run --project .\package_qaagent python -m unittest discover -s .\package_qaagent\tests -v
```

## Build

```powershell
uv build --project .\package_qaagent
```

The installable Python package source is under `src/qaagent/`.