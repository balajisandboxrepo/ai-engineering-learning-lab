import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

INDIA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Describe India in exactly five concise sentences, one sentence "
            "per line. Do not include an introduction, conclusion, or blank lines.",
        )
    ]
)


class IndiaDescription(BaseModel):
    lines: list[str] = Field(
        min_length=5,
        max_length=5,
        description="Exactly five concise sentences describing India.",
    )


def describe_india() -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the project .env file "
            "or your environment."
        )

    chain = INDIA_PROMPT | ChatOpenAI(
        model="gpt-4o-mini", temperature=0
    ).with_structured_output(IndiaDescription)
    response = chain.invoke({})
    lines = [line.strip() for line in response.lines]
    if any(not line for line in lines):
        raise ValueError("OpenAI returned an empty line in the India description.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_india())
