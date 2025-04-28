# Simple UMLS Pipeline

End‑to‑end pipeline that:
* extracts UMLS concepts with QuickUMLS
* enriches each note with **hybrid embeddings** (CUI2Vec + SentenceTransformer)
* supports **SapBERT** or **MiniLM** encoders, concat or learnable linear fusion
* fills missing CUI vectors via **Text‑to‑Vec** or **Graph‑completion** fallbacks
* persists note‑level vectors (`.npy` + meta CSV)
* offers interactive visualisation & unsupervised clustering (UMAP + HDBSCAN / GMM / Spectral) via CLI

## Features

- Extract UMLS concepts from medical text documents
- Support for CSV and TXT input files
- Parallel processing capability for large datasets
- Configurable batch processing
- Sentence‑level embeddings (SapBERT / MiniLM)
- Hybrid fusion & multiple fallback strategies
- Document‑vector persistence for reuse
- Visualisation & clustering utilities (UMAP + HDBSCAN,GMM,Spectral)

## Dependencies (see `requirements.txt`)

Python 3.8+ and the packages in `requirements.txt`, notably

* spaCy + medspaCy + QuickUMLS
* gensim, sentence‑transformers, faiss‑cpu, torch
* umap‑learn, hdbscan, scikit‑learn, plotly

## Installation

### 1. UMLS Setup (Required)

**Obtain a UMLS installation**
This tool requires you to have a valid UMLS installation on disk. To set up UMLS:

1. [Obtain a UMLS license](https://www.nlm.nih.gov/databases/umls.html) from the National Library of Medicine
2. Download all UMLS files from their download page
3. Install UMLS using the MetamorphoSys tool as explained in their [guide](https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html).


### 2. Installation

1. Clone this repository:
```bash
git clone https://github.com/BMIN-5100-Spring-2025/Sy_UMLSpipeline.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Basic usage:
```bash
python3 main.py \
  -i data/mtsamples-demo.csv \
  -o notes.jsonl \
  -u 2020AB-full/2020AB-quickumls-install/ \
  --embeddings data/cui2vec_pretrained.txt \
  --sbert-model sapbert \
  --fusion concat \
  --fallback graph \
  --mrrel 2020AB-full/MRREL.RRF \
  --vectors-out doc_vectors
```

Options (key additions bold):
- `-i, --input`: Path to input file (CSV or TXT)
- `-o, --output`: Path to output file (JSONL format)
- `-u, --umls`: Path to UMLS database
- `-p, --parallel`: Enable parallel processing
- `--batch-size`: Number of documents to process in each batch (default: 100)
- `--workers`: Number of worker processes for parallel processing (default: CPU count)
- `--embeddings`: Path to CUI2Vec file (word2vec format)
- **`--sbert-model`**: `sapbert` | `minilm`
- **`--fusion`**: `concat` | `linear` (learnable)
- **`--fallback`**: `text2vec` | `graph`
- **`--mrrel`**: Path to `MRREL.RRF` (needed for graph fallback)
- **`--vectors-out`**: Output basename for saved `.npy` + `_meta.csv`

### Input File Format

For CSV files:
- Must contain columns for document ID and text
- Default column names: "note_id" for ID and "AGG_TEXT" for text content
- Can be configured using the ProcessingConfig class

For TXT files:
- One document per line

### Output Format

The tool outputs JSONL (JSON Lines) format with the following structure:
```json
{
    "row_id": "document_id",
    "text": "original text",
    "umls": [
        {
            "start_char": 0,
            "end_char": 10,
            "raw": "extracted text",
            "cui": "UMLS CUI",
            "score": 0.85,
            "semtype": ["semantic type"]
        }
    ]
}
```

`notes.jsonl` will include a `doc_vector` field (list of floats).

If `--vectors-out` was provided you also get:
* `doc_vectors.npy` – N × D float32 matrix
* `doc_vectors_meta.csv` – mapping of row_id → row index

## Parallel Processing

- For large datasets, enable parallel processing using the `-p` flag
- Adjust `batch_size` based on available memory
- Number of workers defaults to CPU count but can be adjusted using `--workers`
- The tool automatically chunks data for memory-efficient processing

## Visualisation & Clustering

### CLI quick look

```
python doc_visualizer.py \
  --vectors doc_vectors.npy \
  --meta doc_vectors_meta.csv \
  --cluster gmm \
  --out-html note_map.html
```

Pick `hdbscan` (default), `gmm`, or `spectral`.

### Roadmap

Upcoming items: vector‑cache pickles, UMLS graph serialization, performance benchmarks.
