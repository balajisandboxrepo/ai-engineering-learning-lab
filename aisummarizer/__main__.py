import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


ARTICLE_URL = "https://www.kdnuggets.com/named-entity-recognition-using-scikit-llm"
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


@tool
def crawl_kdnuggets_article() -> str:
    """Fetch and extract the article text from the configured KDnuggets page."""
    parsed_url = urlparse(ARTICLE_URL)
    if parsed_url.scheme != "https" or parsed_url.hostname != "www.kdnuggets.com":
        raise ValueError("The article URL must use HTTPS on www.kdnuggets.com.")

    response = requests.get(
        ARTICLE_URL,
        headers={"User-Agent": "aisummarizer/1.0 (article summarization)"},
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    article = (
        soup.select_one("article")
        or soup.select_one("main")
        or soup.select_one("div.post")
    )
    if article is None:
        raise ValueError("Could not find the article content on the KDnuggets page.")

    for element in article.select("script, style, nav, footer, header, aside"):
        element.decompose()

    text = " ".join(article.stripped_strings)
    if not text:
        raise ValueError("The KDnuggets page contained no extractable article text.")
    return f"Source: {ARTICLE_URL}\n\n{text}"


class ArticleSummary(BaseModel):
    lines: list[str] = Field(
        min_length=10,
        max_length=10,
        description=(
            "Exactly ten concise, non-empty lines explaining the article "
            "for an AI solutions architect."
        ),
    )


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize only the supplied article for an AI solutions architect. "
            "Cover its purpose, approach, key implementation concepts, and "
            "important limitations or operational considerations. Use exactly "
            "ten concise, self-contained lines; do not add a title or blank lines.",
        ),
        ("human", "Article text:\n{article_text}"),
    ]
)


def summarize_article() -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the project .env file "
            "or your environment."
        )

    article_text = crawl_kdnuggets_article.invoke({})
    chain = SUMMARY_PROMPT | ChatOpenAI(
        model="gpt-4o-mini", temperature=0
    ).with_structured_output(ArticleSummary)
    summary = chain.invoke({"article_text": article_text})
    lines = [line.strip() for line in summary.lines]
    if any(not line or "\n" in line or "\r" in line for line in lines):
        raise ValueError("OpenAI returned an empty or multiline summary item.")
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, 1))


if __name__ == "__main__":
    print(summarize_article())
