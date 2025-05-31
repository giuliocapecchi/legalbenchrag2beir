from collections.abc import Sequence
import unicodedata

from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import Self


class Snippet(BaseModel):
    file_path: str
    span: tuple[int, int]
    gold: str | None = None

    @computed_field  # type: ignore[misc]
    @property
    def answer(self) -> str:
        if self.gold is not None:
            return self.gold
        self.file_path = unicodedata.normalize("NFC", self.file_path)
        with open(f"./data/corpus/{self.file_path}", encoding="utf-8") as f:
            return f.read()[self.span[0] : self.span[1]]
    


def validate_snippet_list(snippets: Sequence[Snippet]) -> None:
    snippets_by_file_path: dict[str, list[Snippet]] = {}
    for snippet in snippets:
        if snippet.file_path not in snippets_by_file_path:
            snippets_by_file_path[snippet.file_path] = [snippet]
        else:
            snippets_by_file_path[snippet.file_path].append(snippet)

    for _file_path, snippets in snippets_by_file_path.items():
        snippets = sorted(snippets, key=lambda x: x.span[0])
        for i in range(1, len(snippets)):
            if snippets[i - 1].span[1] >= snippets[i].span[0]:
                raise ValueError(
                    f"Spans are not disjoint! {snippets[i - 1].span} VS {snippets[i].span}"
                )


class QAGroundTruth(BaseModel):
    query: str
    snippets: list[Snippet]
    tags: list[str] = []

    @model_validator(mode="after")
    def validate_snippet_spans(self) -> Self:
        validate_snippet_list(self.snippets)
        return self


class Benchmark(BaseModel):
    tests: list[QAGroundTruth]


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    span: tuple[int, int]
    content: str

class Document(BaseModel):
    file_path: str
    content: str
