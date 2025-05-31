# legalbenchrag2beir

[**LegalBench-RAG**](https://github.com/zeroentropy-ai/legalbenchrag?tab=readme-ov-file) is a newly introduced benchmark by [ZeroEntropy](https://www.zeroentropy.dev/) designed to evaluate the retrieval capabilities of Retrieval-Augmented Generation (RAG) systems in the legal domain. 

However, the dataset in its original form is not structured according to standard IR benchmarks, such as those used in [BEIR](https://github.com/beir-cellar/beir). This repository provides a converter (`create_dataset.py`) that restructures it, making it easier to integrate with existing retrieval pipelines, perform evaluations, and experiment with it.

---

## 💻 Requirements and setup

* Python 3.10
* Install dependencies with:

```bash
pip install -r requirements.txt
```
* Download the LegalBench-RAG dataset at [this link](https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&e=1&st=0hu354cq&dl=0). Then unzip it inside ./data. You should have obtained the following structure:

```
./data/
├── benchmarks/
│   ├── cuad.json
│   └── ...
├── corpus/
│   ├── maud/...
│   └── ...
```

## ❓How to use

When everything is set up just run the script:

   ```bash
   python create_dataset.py
   ```

The formatted dataset will be saved in `./data/legalbenchrag/`. By default, the script uses the best settings provided in the original paper (LangChain's `Recursive Character TextSplitter` and 0 overlap between chunks).

## 📊 Final Statistics

The script prints:

* Number of original documents and generated chunks
* Number of queries
* Min / max / average answer span lengths
* QREL counts per split

You should get the following:

```bash
----- Statistics for train qrels        ------------------------------
Number of queries contained: 4133
Average number of positive documents per query: 2.764
Number of unique documents: 10128
----- Statistics for dev qrels  ------------------------------
Number of queries contained: 1377
Average number of positive documents per query: 2.674
Number of unique documents: 3550
----- Statistics for test qrels ------------------------------
Number of queries contained: 1379
Average number of positive documents per query: 2.750
Number of unique documents: 3642

Total number of queries: 6889. The sum of relevant queries across all splits is 6889 which is 100.00% of the total querie
```

---

## 📚 Citation

If you use this dataset, please cite the [original paper](https://arxiv.org/abs/2408.10343) and this repository.
