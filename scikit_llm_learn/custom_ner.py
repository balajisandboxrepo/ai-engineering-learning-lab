import os
from pathlib import Path

from dotenv import load_dotenv
from skllm.config import SKLLMConfig
from skllm.models.gpt.tagging.ner import GPTExplainableNER


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the project-root .env file "
            "or your environment."
        )
    SKLLMConfig.set_openai_key(api_key)

    entities = {
        "PERSON": "A person's full name.",
        "COMPANY": "The name of a company or organization.",
        "PRODUCT": "A named software product or technology.",
        "LOCATION": "A city, region, or country.",
    }
    texts = [
        "Maya Patel demoed Contoso's Copilot assistant to customers in Bengaluru."
    ]

    tagger = GPTExplainableNER(
        entities=entities,
        model="gpt-4o-mini",
        sparse_output=False,
    )
    predictions = tagger.fit_transform(texts)

    for text, prediction in zip(texts, predictions):
        print(f"Text: {text}")
        print(f"Extracted entities: {prediction}")


if __name__ == "__main__":
    main()
