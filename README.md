# Simple UMLS Pipeline

A very simple pipeline for processing medical text documents to extract medical concepts and CUIs using medspaCy and QuickUMLS.

## Features

- Extract UMLS concepts from medical text documents
- Support for CSV and TXT input files
- Parallel processing capability for large datasets
- Configurable batch processing

## Dependencies

- Python 3.8+
- medspaCy
- QuickUMLS installation with UMLS database
- spaCy

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
git clone https://github.com/yourusername/UMLSpipeline.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Basic usage:
```bash
python3 main.py -i input_file.csv -o output.jsonl -u /path/to/umls
```

Options:
- `-i, --input`: Path to input file (CSV or TXT)
- `-o, --output`: Path to output file (JSONL format)
- `-u, --umls`: Path to UMLS database
- `-p, --parallel`: Enable parallel processing
- `--batch-size`: Number of documents to process in each batch (default: 100)
- `--workers`: Number of worker processes for parallel processing (default: CPU count)

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

### Python API Usage

```python
from utils import ProcessingConfig, DataLoader
from processor import NLPProcessor
import medspacy

# Configure processing
config = ProcessingConfig(
    input_file="input.csv",
    output_file="output.jsonl",
    umls_path="/path/to/umls",
    parallelize=True
)

# Initialize components
nlp = medspacy.load(enable=['medspacy_pyrush', 'medspacy_context'])
loader = DataLoader()
processor = NLPProcessor(nlp, config)

# Process documents
docs = loader.from_file(config.input_path, text_column="AGG_TEXT")
note_ids = loader.from_file(config.input_path, id_column="note_id")

# Write results
with open(config.output_path, 'w') as outfile:
    for result in processor.process_documents(note_ids, docs):
        outfile.write(json.dumps(result) + '\n')
```

## Parallel Processing

- For large datasets, enable parallel processing using the `-p` flag
- Adjust `batch_size` based on available memory
- Number of workers defaults to CPU count but can be adjusted using `--workers`
- The tool automatically chunks data for memory-efficient processing

## Running on i2c2 cluster

Note that QuickUMLS requires `libstdc++` so you may need to load a gcc module. You may want to check `module avail` to load a compatible version, like below:

```
module load gcc/12.2.0
```
