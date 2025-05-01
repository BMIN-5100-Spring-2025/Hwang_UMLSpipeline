# ConceptMap: Enhancing Medical Document Representation through Concept Embeddings

End‑to‑end pipeline that:
* Extracts UMLS concepts from text using QuickUMLS.
* Generates **hybrid document embeddings** by fusing CUI2Vec concept vectors and SentenceTransformer text vectors.
* Supports configurable **SentenceTransformer models** (e.g., `minilm`, `sapbert`, or local paths).
* Offers **fusion strategies** (`concat` or `linear`).
* Provides **fallback mechanisms** for missing CUI vectors (`text2vec` similarity or `graph` traversal using MRREL).
* Implements **SIF weighting** for pooling concept vectors.
* Optionally **persists** document vectors (`.npy` + meta CSV) and calculates isotropy.
* Can perform **unsupervised clustering** (HDBSCAN, KMeans, GMM, Spectral) on generated vectors.
* Includes **interactive visualization** generation (UMAP + Plotly).
* Supports both **local execution** and **AWS Fargate deployment** via Docker, with S3 integration.

## Features

- UMLS concept extraction (QuickUMLS).
- Hybrid document embedding generation (CUI2Vec + SBERT).
- Configurable SBERT models, fusion strategies, and fallback options.
- Document vector persistence and reuse.
- Optional clustering (HDBSCAN, KMeans, GMM, Spectral).
- Interactive 2D/3D visualization output (UMAP + Plotly).
- Parallel processing support for local execution.
- S3 integration for input/output in Fargate mode.
- Support for CSV and TXT input files.

## Dependencies (see `requirements.txt`)

Python 3.8+ and the packages listed in `requirements.txt`, notably:

*   `spacy`, `quickumls` (via `medspacy`)
*   `gensim`, `sentence-transformers`, `torch` (optional for linear fusion)
*   `numpy`, `pandas`, `scikit-learn`, `umap-learn`, `hdbscan`
*   `plotly`, `tqdm`
*   `boto3` (for Fargate/S3 mode)

## Setup

### 1. UMLS Data (Required)

*   **QuickUMLS Installation:** This pipeline relies on a QuickUMLS installation directory derived from UMLS data.
    *   Obtain a [UMLS license](https://www.nlm.nih.gov/databases/umls.html).
    *   Download the UMLS `MRCONSO.RRF` and `MRSTY.RRF` files.
    *   Use a tool like `medspaCy`'s QuickUMLS installer or follow [QuickUMLS setup instructions](https://github.com/Georgetown-IR-Lab/QuickUMLS) to create the necessary installation directory (e.g., `/path/to/2020AB-quickumls-install`).
    *   If using the `graph` fallback strategy, you also need the `MRREL.RRF` file from the UMLS distribution.

### 2. Pre-trained Embeddings (Required)

*   **CUI2Vec:** Download or train CUI concept embeddings in the word2vec text format (e.g., `cui2vec_pretrained.txt`).
*   **SentenceTransformer Models:** The specified SBERT models (e.g., `all-MiniLM-L6-v2`, `SapBERT-from-PubMedBERT-fulltext`) will be downloaded automatically by the `sentence-transformers` library on first use unless a local path is provided or they are pre-downloaded for Fargate mode.

### 3. Code Installation

1.  Clone this repository:
    ```bash
    git clone <your-repo-url> UMLSpipeline
    cd UMLSpipeline
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Local Execution (`main.py`)

The main script `main.py` processes input files locally.

**Example:**

```bash
python main.py \
    --input path/to/your/data.csv \
    --output output/results.jsonl \
    --umls /path/to/2020AB-quickumls-install \
    --tcol text_column_name \
    --idcol id_column_name \
    --embeddings path/to/cui2vec_pretrained.txt \
    --sbert-model minilm \
    --fusion concat \
    --fallback text2vec \
    --cluster hdbscan \
    --hdb-min-cluster-size 10 \
    --vectors-out output/doc_vectors \
    --visualize output/note_map.html \
    --viz-dimension 3 \
    --parallel \
    --workers 4 
```

**Key Command-Line Arguments:**

*   `-i, --input`: Path to input file (CSV or TXT). (Required)
*   `-o, --output`: Path for the main output JSONL file. (Required)
*   `-u, --umls`: Path to the QuickUMLS installation directory. (Required)
*   `-t, --tcol`: Name of the text column in the input CSV. (Required for CSV)
*   `-d, --idcol`: Name of the document ID column in the input CSV. (Required for CSV)
*   `--embeddings`: Path to the CUI2Vec embeddings file (word2vec text format). (Required for embedding generation)
*   `--sbert-model`: SentenceTransformer model ID (HuggingFace) or local path (default: `minilm`). Aliases: `minilm`, `sapbert`.
*   `--fusion`: How to combine CUI and SBERT vectors: `concat` (default) or `linear` (requires PyTorch).
*   `--fallback`: Strategy for CUIs missing embeddings: `text2vec` (default, uses SBERT similarity) or `graph` (uses UMLS MRREL graph).
*   `--mrrel`: Path to `MRREL.RRF` file (Required only for `--fallback graph`).
*   `--vectors-out`: Base path to save document vectors (`.npy`) and metadata (`_meta.csv`). (Optional)
*   `--visualize`: Path to save the interactive Plotly HTML visualization. (Optional, requires `--vectors-out`)
*   `--viz-dimension`: Output dimension for visualization (default: 2, options: 2, 3).
*   `--cluster-method`: Clustering algorithm: `hdbscan`, `kmeans`, `gmm`, `spectral`. (Optional, requires `--vectors-out`)
*   `--n-clusters`: Number of clusters for KMeans/GMM/Spectral. (Required for these methods)
*   `--hdb-min-cluster-size`: `min_cluster_size` for HDBSCAN (default: 5).
*   `--hdb-min-samples`: `min_samples` for HDBSCAN (optional).
*   `-p, --parallel`: Enable parallel processing using multiprocessing. (Optional)
*   `--batch-size`: Documents per batch for processing (default: 100).
*   `--workers`: Number of worker processes for parallel mode (default: CPU count).

### Fargate / Docker Execution

The pipeline can run within a Docker container, suitable for AWS Fargate. See `Dockerfile`, `compose.yaml`, and `README.Docker.md`.

**Key Environment Variables for Fargate Mode:**

*   `MODE=fargate`: Activates Fargate mode (triggers S3 downloads/uploads).
*   `S3_BUCKET`: Name of the S3 bucket for inputs/outputs.
*   `SESSION_ID`: Unique identifier for the processing job, used in S3 paths.
*   `PARAMETERS`: Base64 encoded JSON string containing job parameters (e.g., `input_filename`, `cluster_method`, `n_clusters`, `viz_dimension`).
*   `INPUT_DIR`: Local container path for downloads (default: `/tmp/input`).
*   `OUTPUT_DIR`: Local container path for outputs before upload (default: `/tmp/output`).

**Expected S3 Structure (Fargate):**

*   Input File: `s3://{S3_BUCKET}/{SESSION_ID}/input/{safe_input_filename}` (Filename is sanitized).
*   UMLS Data: `s3://{S3_BUCKET}/2020AB-full/` (Contains QuickUMLS install dir and MRREL.RRF).
*   Embeddings: `s3://{S3_BUCKET}/data/embeddings/` (Contains `cui2vec_pretrained.txt`).
*   SBERT Model: `s3://{S3_BUCKET}/models/miniLM/` (Example for MiniLM, needs pre-downloaded model files).
*   Output: Files generated in `OUTPUT_DIR` are uploaded to `s3://{S3_BUCKET}/{SESSION_ID}/output/`.

### Output Formats

1.  **Main Output (`--output`)**: JSON Lines (`.jsonl`) file. Each line is a JSON object representing a processed document:
    ```json
    {
        "row_id": "document_id",
        "text": "original text...",
        "umls": [
            {
                "start_char": 10, "end_char": 15, "raw": "term",
                "cui": "C0000001", "score": 0.95, "semtype": ["T047"]
            }, ...
        ],
        "doc_vector": [0.1, -0.2, ..., 0.5],
        "cluster": 3 
    }
    ```
2.  **Document Vectors (`--vectors-out`)**:
    *   `<basename>.npy`: NumPy array (`float32`) of shape (N_docs, Embedding_dim) containing the final document vectors (after SIF pooling, fusion, and optional PC removal).
    *   `<basename>_meta.csv`: CSV file mapping `row_id` to the corresponding row index in the `.npy` file.
3.  **Visualization (`--visualize`)**:
    *   `<filename>.html`: Interactive Plotly scatter plot (2D or 3D) generated using UMAP on the document vectors. Points are colored by cluster ID (if available) and hover text shows `row_id` and top concepts.
4.  **Metrics (`<OUTPUT_DIR>/metrics.json`)**: JSON file containing processing metrics:
    ```json
    {
      "num_documents": 1000,
      "isotropy_score": 0.85,
      "eigenvalue_spectrum_percent": [0.15, 0.10, ...]
    }
    ```
5.  **Embedding Stats (Logged)**: Coverage statistics for CUI embeddings (exact match, fallback hits, misses) are logged to the console.

### Standalone Visualization (`doc_visualizer.py`)

Use `doc_visualizer.py` to visualize pre-computed document vectors stored in `.npy` format.

**Example:**

```bash
python doc_visualizer.py \
    --vectors output/doc_vectors.npy \
    --meta output/doc_vectors_meta.csv \
    --cluster hdbscan \
    --hdb-min-cluster-size 10 \
    --out-html output/standalone_viz.html
```

This script performs UMAP and clustering (HDBSCAN, GMM, or Spectral) directly on the loaded vectors.

## Methodology

Refer to `METHODOLOGY.md` for a more detailed explanation of the embedding generation, fusion, and fallback techniques.
