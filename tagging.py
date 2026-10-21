import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


class PostTag(BaseModel):
    type: int = Field(ge=1, le=10, description="Номер типа поста по таксономии, 1–10")
    alt_type: int | None = Field(None, ge=1, le=10, description="Запасной тип для пограничного случая, иначе null")
    confidence: str = Field(pattern="^(high|low)$", description="high, если тип очевиден; low, если пост пограничный")


def main(path):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты размечаешь посты госпаблика по таксономии. Один тип на пост, по главному смыслу. "
                   "Следуй правилам пограничных случаев.\n\n{taxonomy}"),
        ("human", "{text}"),
    ]).partial(taxonomy=Path("data/taxonomy_v1.md").read_text(encoding="utf-8"))
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), temperature=0)
    chain = (prompt | llm.with_structured_output(PostTag)).with_retry(stop_after_attempt=4, wait_exponential_jitter=True)
    df = pd.read_csv(path)
    df = df[df.text.fillna("").str.strip() != ""]
    tags = chain.batch([{"text": t[:4000]} for t in df.text], config={"max_concurrency": 2})
    out = df[["id"]].reset_index(drop=True).join(pd.DataFrame([t.model_dump() for t in tags])) # type: ignore
    out.to_csv(Path(path).with_name(f"{Path(path).stem}_tagged.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
