import json


def load_jsonl(file_path):
    """Reads a JSONL file and returns a dictionary."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            entry_id = entry.pop('_id') # each entry has an '_id' field
            data[entry_id] = entry        
    return data

def load_tsv(file_path):
    """Reads a qrel TSV file and returns a dictionary with qid -> {doc_id: score}."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            qid, doc_id, score = line.strip().split('\t')
            if qid not in data:
                data[qid] = {}  # initialize the dictionary
            data[qid][doc_id] = float(score)
    return data

def process_and_print_statistics(queries_path, qrels_folder):
    """Loads qrels from the specified folder and prints statistics for each split."""
    queries = load_jsonl(queries_path)
    relevant_queries_number = 0
    for split in ["train", "dev", "test"]:
        qrels_path = f"{qrels_folder}/{split}.tsv"
        qrels = load_tsv(qrels_path)
        print(f"{'-'*5} Statistics for {split} qrels\t{'-'*30}")
        relevant_queries = {qid: q for qid, q in queries.items() if qid in qrels.keys()}
        positives_per_query = [len(qrels[qid]) for qid in qrels.keys()]
        relevant_queries_number += len(qrels)
        print(f"Number of queries contained: {len(relevant_queries)}")
        print(f"Average number of positive documents per query: {sum(positives_per_query) / len(positives_per_query):.3f}")
        print(f"Number of unique documents: {len(set(doc_id for qid in qrels.keys() for doc_id in qrels[qid].keys()))}")

    print(f"\nTotal number of queries: {len(queries)}. The sum of relevant queries across all splits is {relevant_queries_number} which is {relevant_queries_number / len(queries) * 100:.2f}% of the total queries.")


if __name__ == "__main__":
    queries_path = "./data/legalbenchrag/queries.jsonl"
    qrels_folder = "./data/legalbenchrag/qrels"
    process_and_print_statistics(queries_path, qrels_folder)
