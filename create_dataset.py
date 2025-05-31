import json
import os
import random
import unicodedata
from typing import Dict, List
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from benchmark_types import Benchmark, Chunk, Snippet, Document, QAGroundTruth
from dataset_statistics import process_and_print_statistics


name_to_weight: dict[str, float] = {
    "privacy_qa": 0.25,
    "contractnli": 0.25,
    "maud": 0.25,
    "cuad": 0.25,
}


# This is the minimum overlap size to consider a match (in characters)
# In other words, we chunk the documents into smaller pieces, then check if the snippets (relevant documents) overlap with the chunks.
# This is used to filter out small overlaps that are likely to be noise
MIN_OVERLAP_SIZE = 5

# This is the chunk size used to split the documents into smaller pieces
# It was set to 500 characters in the paper, so we use the same value here
CHUNK_SIZE = 500


def create_corpus(corpus: List[Chunk], output_path: str) -> None:
    """Create a corpus file in JSONL format, given a list of chunks."""
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(corpus, desc="Creating corpus...", unit="chunk"): # treat each chunk as a document
            document = {
                "_id": f"{chunk.chunk_id}",
                "title": "",
                "text": chunk.content,
                "metadata": {
                    "document_id": chunk.document_id,
                    "span": chunk.span,
                }
            }
            f.write(json.dumps(document, ensure_ascii=False) + "\n")

def create_queries(all_tests: list[tuple[int, QAGroundTruth]], output_path: str) -> None:
    """Create a queries file in JSONL format, given a list of tests."""
    with open(output_path, "w", encoding="utf-8") as f:
        for query_id, test in tqdm(all_tests, desc="Creating queries...", unit="query"):
            snippets = [
                Snippet(
                    file_path = snippet.file_path,
                    span =  snippet.span,
                    answer =  snippet.answer,
                    )
                for snippet in test.snippets
            ]

            query = {
                "_id": str(query_id),  # Unique query ID
                "text": test.query,
                "metadata": {
                    "benchmark_name": test.tags[0],  # First tag as benchmark name
                    "snippets": [(snippet.file_path,snippet.span, snippet.answer) for snippet in snippets],  # Convert snippets to JSON
                },
            }
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

def create_qrels(all_tests: list[tuple[int, QAGroundTruth]], chunks: List[Chunk], output_path: str, split: str) -> None:
    """Create a qrels file in TSV format, given a list of tests and chunks."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for query_id, test in tqdm(all_tests, desc=f"Creating qrels [{split}]...", unit="query"):
            for snippet in test.snippets:
                # assuming that for each snippet there is a correct document match
                document_id = snippet.file_path
                # check if the answer was split into multiple chunks
                for chunk in [chunk for chunk in chunks if chunk.document_id == document_id]:
                    common_min = max(chunk.span[0], snippet.span[0])
                    common_max = min(chunk.span[1], snippet.span[1])
                    overlap_size = common_max - common_min
                    if overlap_size > MIN_OVERLAP_SIZE:
                        score = 1
                        f.write(f"{query_id}\t{chunk.chunk_id}\t{score}\n")
                
def split_data(all_tests: list[tuple[int, QAGroundTruth]], train_ratio=0.6, dev_ratio=0.2) -> Dict[str, list[tuple[int, QAGroundTruth]]]:
    """ Split the data into train, dev, and test sets.
    Args:
        all_tests (list[tuple[int, QAGroundTruth]]): List of all tests with their IDs.
        train_ratio (float): Ratio of training data.
        dev_ratio (float): Ratio of development data.
    Returns:
        Dict[str, list[tuple[int, QAGroundTruth]]]: Dictionary containing train, dev, and test sets.
    Note:
        The test set is the remaining data after splitting the train and dev sets (i.e., 1 - train_ratio - dev_ratio).
    """
    random.seed(42)  # reproducibility
    random.shuffle(all_tests)
    total = len(all_tests)
    train_size = int(train_ratio * total)
    dev_size = int(dev_ratio * total)


    return {
        "train": all_tests[:train_size],
        "dev": all_tests[train_size: train_size + dev_size],
        "test": all_tests[train_size + dev_size:],
    }


def main() -> None:
    all_tests: list[QAGroundTruth] = []
    weights: list[float] = []
    used_document_file_paths_set: set[str] = set()
    
    for benchmark_name, weight in name_to_weight.items():
        with open(f"./data/benchmarks/{benchmark_name}.json", encoding="utf-8") as f:
            benchmark = Benchmark.model_validate_json(f.read())
            tests = benchmark.tests
            used_document_file_paths_set |= {
                snippet.file_path for test in tests for snippet in test.snippets
            }
            for test in tests:
                test.tags = [benchmark_name]
            all_tests.extend(tests)
            weights.extend([weight / len(tests)] * len(tests))
    benchmark = Benchmark(
        tests=all_tests,
    )

    # gold documents stats
    all_snippets = [snippet for test in all_tests for snippet in test.snippets]
    all_snippets_length = [snippet.span[1] - snippet.span[0] for snippet in all_snippets]
    print(f"Min length of snippets: {min(all_snippets_length)}")
    print(f"Max length of snippets: {max(all_snippets_length)}")
    print(f"Mean length of snippets: {sum(all_snippets_length) / len(all_snippets_length)}")
    
    # create corpus
    corpus: list[Document] = []
    for document_file_path in sorted(used_document_file_paths_set):

        document_file_path = unicodedata.normalize("NFC", document_file_path)
        with open(f"./data/corpus/{document_file_path}", encoding="utf-8") as f:
            corpus.append(
                Document(
                    file_path=document_file_path,
                    content=f.read(),
                )
            )

    print(f"Number of documents: {len(corpus)}")
    print(f"Number of corpus characters: {sum(len(document.content) for document in corpus)}")
    print(f"Number of queries: {len(benchmark.tests)}")

    documents: dict[str, Document] = {}
    
    for document in corpus:
        documents[document.file_path] = document
    
    # get the chunks with the Recursive Character TextSplitter (rtcs) and 0 overlap chunking strategy
    # since it was the one that worked best in the paper
    chunks: list[Chunk] = []
    for document_id, document in tqdm(documents.items(), desc="Chunking documents...", unit="document"):
        synthetic_data_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n",
                "\n",
                "!",
                "?",
                ".",
                ":",
                ";",
                ",",
                " ",
                "",
            ],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            strip_whitespace=False,
        )
        text_splits = synthetic_data_splitter.split_text(document.content)
        assert sum(len(text_split) for text_split in text_splits) == len(
            document.content
        )
        assert "".join(text_splits) == document.content

        # get spans from chunks
        prev_span: tuple[int, int] | None = None
        for text_split in text_splits:
            prev_index = prev_span[1] if prev_span is not None else 0
            span = (prev_index, prev_index + len(text_split))
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_id=str(len(chunks)),
                    span=span,
                    content=text_split.replace("\n", " ").replace("\t", " ").strip(),
                )
            )
            prev_span = span

    assert len(set(chunk.document_id for chunk in chunks)) == len(documents)
    print(f"Num Chunks created: {len(chunks)}")
    
    # create the dataset
    os.makedirs("./data/legalbenchrag/qrels", exist_ok=True)
    all_tests_with_id = list(enumerate(all_tests))
    create_corpus(chunks, "./data/legalbenchrag/corpus.jsonl")
    create_queries(all_tests_with_id, "./data/legalbenchrag/queries.jsonl")
    splits = split_data(all_tests_with_id)
    create_qrels(splits["train"], chunks, "./data/legalbenchrag/qrels/train.tsv", "train")
    create_qrels(splits["dev"], chunks, "./data/legalbenchrag/qrels/dev.tsv", "dev")
    create_qrels(splits["test"], chunks, "./data/legalbenchrag/qrels/test.tsv", "test")

    # print statistics
    queries_path = "./data/legalbenchrag/queries.jsonl"
    qrels_folder = "./data/legalbenchrag/qrels"
    process_and_print_statistics(queries_path, qrels_folder)


if __name__ == "__main__":
    main()