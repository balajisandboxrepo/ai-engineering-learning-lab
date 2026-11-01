# Scikit-LLM: from basics to AI solution architecture

Scikit-LLM connects language models to familiar scikit-learn estimator patterns. It helps Python and scikit-learn users apply LLMs to text tasks without building every prompt and API call themselves.

## 1. The basic idea

In conventional machine learning, an estimator learns a mapping from examples. A Scikit-LLM estimator instead uses an LLM to perform a text task, while exposing methods such as `fit`, `predict`, `transform`, or `fit_transform`.

For example, `ZeroShotGPTClassifier` receives candidate labels and text, then returns one label per text. In a zero-shot setup, `fit(None, labels)` supplies the allowed labels without training on labeled examples. `GPTExplainableNER` receives text and an entity dictionary; each dictionary value describes the entity category for the model.

The LLM is still doing inference remotely: `fit` does not mean that model weights are being trained. The estimator is an adapter between a task, scikit-learn-style inputs and outputs, and a model API.

## 2. Install in an isolated environment

Scikit-LLM 1.4.3 declares `openai>=1.2,<2`. The other examples in this workspace use a newer OpenAI SDK, so install Scikit-LLM in its own environment rather than changing the workspace environment:

```powershell
uv venv .venv-scikit-llm-learn
uv pip install --python .\.venv-scikit-llm-learn\Scripts\python.exe -r .\scikit_llm_learn\requirements.txt
```

The examples read `OPENAI_API_KEY` from the workspace-root `.env` file or the process environment. Keep the key out of source control and logs.

## 3. Example one: zero-shot classification

Run from the workspace root:

```powershell
.\.venv-scikit-llm-learn\Scripts\python.exe -m scikit_llm_learn.zero_shot_classification
```

This example classifies three short support messages as positive, negative, or neutral. Its essential flow is:

```python
classifier = ZeroShotGPTClassifier(model="gpt-4o-mini")
classifier.fit(None, ["positive", "negative", "neutral"])
labels = classifier.predict(texts)
```

Because no labeled training examples are passed to `fit`, this is zero-shot classification. Candidate labels are task instructions, not learned classes.

## 4. Example two: custom named-entity recognition

Run:

```powershell
.\.venv-scikit-llm-learn\Scripts\python.exe -m scikit_llm_learn.custom_ner
```

This example defines four domain entity types and extracts matching spans from a short sentence:

```python
tagger = GPTExplainableNER(entities=entities, model="gpt-4o-mini")
predictions = tagger.fit_transform(texts)
```

The entity dictionary is part of the task specification: good definitions should distinguish categories clearly. This example selects `sparse_output=False`, which asks Scikit-LLM for tagged XML-like text rather than its JSON-array mode. Scikit-LLM formats the model output as tagged text; inspect and validate that output before treating it as canonical business data.

## 5. From a working demo to an architected solution

Treat these estimators as LLM-backed components, not as a complete production platform. A practical architecture usually adds:

1. **Input controls:** normalize text, enforce size and format limits, and remove or mask sensitive data where policy requires it.
2. **Task contract:** define labels or entity schemas, clarify ambiguous categories, and version prompts and definitions.
3. **Inference boundary:** isolate provider credentials, select an approved model and endpoint, and send only data permitted by privacy and residency policies.
4. **Reliability controls:** handle timeouts, rate limits, provider errors, malformed outputs, and retries with bounded policies.
5. **Quality checks:** evaluate representative labeled examples, measure per-label precision/recall for extraction tasks, and validate output against downstream schemas.
6. **Cost and latency controls:** estimate tokens and request volume, set budgets, measure end-to-end latency, and choose concurrency based on provider quotas.
7. **Observability and governance:** record model, prompt/config version, latency, usage, and validation outcomes without logging secrets or unnecessary raw text.
8. **Safe downstream use:** treat model output as a prediction, not a fact; add review or escalation for high-impact decisions.

LLMs may produce inconsistent or incorrect classifications and entity boundaries, even when output appears well-formed. Test against a task-specific baseline and real representative data before deployment. Both examples make live OpenAI API calls and may incur charges.

## References

- [Scikit-LLM documentation](https://skllm.beastbyte.ai/)
- [Scikit-LLM v1.4.3 source and package metadata](https://github.com/fnnx-ai/scikit-llm/tree/v1.4.3)
- [ZeroShotGPTClassifier source](https://github.com/fnnx-ai/scikit-llm/blob/v1.4.3/skllm/models/gpt/classification/zero_shot.py)
- [GPTExplainableNER source](https://github.com/fnnx-ai/scikit-llm/blob/v1.4.3/skllm/models/gpt/tagging/ner.py)
