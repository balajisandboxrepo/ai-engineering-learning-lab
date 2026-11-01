import os
from pathlib import Path

from dotenv import load_dotenv
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the project-root .env file "
            "or your environment."
        )
    SKLLMConfig.set_openai_key(api_key)

    texts = [
        "The support team resolved my issue quickly and courteously.",
        "The app crashes every time I try to open my account.",
        "The product's current version is 4.2 and was released in June.",
    ]
    labels = ["positive", "negative", "neutral"]

    classifier = ZeroShotGPTClassifier(model="gpt-4o-mini")
    classifier.fit(None, labels)
    predictions = classifier.predict(texts)

    for text, label in zip(texts, predictions):
        print(f"{label}: {text}")


if __name__ == "__main__":
    main()
