import argparse
import spacy
import json
from pathlib import Path
from utils import ProcessingConfig, DataLoader
from processor import NLPProcessor
from embeddings import ConceptEmbedder
from visualizer import ConceptVisualizer
import os
import logging


def initialize_nlp():
    """Initialize the NLP pipeline."""
    try:
        # Use basic spaCy model instead of medspacy
        nlp = spacy.load("en_core_web_sm")
        
        # Verify the pipes were added
        print(f"Active pipes: {nlp.pipe_names}")
        
        return nlp
    except Exception as e:
        print(f"Error initializing NLP pipeline: {e}")
        raise

def parse_args() -> ProcessingConfig:
	"""Parse command line arguments and create config."""
	parser = argparse.ArgumentParser(description="Medical text NLP processing")
	parser.add_argument('-i', '--input', 
					   required=True,
					   help='Path to input text file')
	parser.add_argument('-u', '--umls',
					   required=True,
					   help='Path to UMLS database')
	parser.add_argument('-o', '--output',
					   required=True,
					   help='Path to output file')
	parser.add_argument('-t', '--tcol',
                     	required=True)
	parser.add_argument('-d', '--idcol',
                     	required=True)
	parser.add_argument('-p', '--parallel',
					   action='store_true',
					   help='Enable parallel processing')
	parser.add_argument('--batch-size',
					   type=int,
					   default=100,
					   help='Batch size for document processing')
	parser.add_argument('--workers',
					   type=int,
					   help='Number of worker processes (defaults to CPU count)')
	parser.add_argument('--embeddings',
					   help='Path to CUI embeddings file')
	parser.add_argument('--visualize',
					   help='Path to save visualization HTML')
	
	args = parser.parse_args()
	
	return ProcessingConfig(
		input_file=args.input,
		output_file=args.output,
		text_column=args.tcol,
		id_column=args.idcol,
		umls_path=args.umls,
		batch_size=args.batch_size,
		num_workers=args.workers,
		parallelize=args.parallel,
		embeddings_path=args.embeddings,
		visualization_path=args.visualize
	)

def main() -> None:
    config = parse_args()
    
    # --- Modular S3 logic ---
    mode = os.environ.get('MODE', 'local').lower()
    s3_bucket = os.environ.get('S3_BUCKET')
    session_id = os.environ.get('SESSION_ID')
    if session_id:
        input_prefix = f"{session_id}/input/"
        output_prefix = f"{session_id}/output/"
    else:
        input_prefix = os.environ.get('INPUT_PREFIX', 'input/')
        output_prefix = os.environ.get('OUTPUT_PREFIX', 'output/')
    input_dir = Path(os.environ.get('INPUT_DIR', '/tmp/input'))
    output_dir = Path(os.environ.get('OUTPUT_DIR', '/tmp/output'))

    # If running in Fargate mode, pull input from S3
    if mode == 'fargate' and download_from_s3 and s3_bucket:
        logging.info(f"[Fargate] Downloading input from s3://{s3_bucket}/{input_prefix} to {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)
        download_from_s3(s3_bucket, input_prefix, input_dir) # Download primary input

        # --- Download UMLS data --- 
        umls_s3_prefix = "2020AB-full/" 
        local_umls_base_path = input_dir / "2020AB-full" 
        logging.info(f"[Fargate] Downloading UMLS from s3://{s3_bucket}/{umls_s3_prefix} to {local_umls_base_path}")
        local_umls_base_path.mkdir(parents=True, exist_ok=True)
        download_from_s3(s3_bucket, umls_s3_prefix, local_umls_base_path) 

        # --- ADDED: Download Embeddings --- 
        embeddings_s3_prefix = "data/embeddings/" 
        local_embeddings_path = input_dir / "data" / "embeddings"
        logging.info(f"[Fargate] Downloading Embeddings from s3://{s3_bucket}/{embeddings_s3_prefix} to {local_embeddings_path}")
        local_embeddings_path.mkdir(parents=True, exist_ok=True)
        download_from_s3(s3_bucket, embeddings_s3_prefix, local_embeddings_path) 
        # --- END ADDED --- 

        # Note: MRREL.RRF should be downloaded as part of the UMLS download above
        # as it shares the same S3 prefix.

        # --- Update config paths --- 
        # Find the primary input file (e.g., the CSV)
        files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix in ['.csv', '.txt']] 
        if files:
            config.input_file = str(files[0])
            logging.info(f"Set input file to: {config.input_file}")
        else:
            raise RuntimeError(f"No input CSV/TXT file found in {input_dir} after S3 download.")
        
        # Set output file path
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_file = str(output_dir / 'output.jsonl')
        logging.info(f"Set output file to: {config.output_file}")

        # Update config paths for downloaded data (overrides command-line args if in Fargate)
        # These paths MUST match where the data was downloaded locally above
        config.umls_path = str(local_umls_base_path / "2020AB-quickumls-install")
        config.embeddings_path = str(local_embeddings_path / "cui2vec_pretrained.txt")
        # Add mrrel path if needed, ensure it was downloaded with UMLS
        mrrel_local_path = local_umls_base_path / "MRREL.RRF"
        if mrrel_local_path.exists():
             config.mrrel_path = str(mrrel_local_path) 
        else:
             # Decide how to handle if MRREL isn't downloaded/present
             logging.warning(f"MRREL.RRF not found at {mrrel_local_path}, mrrel_path not set in config.")
             config.mrrel_path = None 
        
        logging.info(f"Set UMLS path to: {config.umls_path}")
        logging.info(f"Set Embeddings path to: {config.embeddings_path}")
        logging.info(f"Set MRREL path to: {config.mrrel_path}")

    # Initialize NLP pipeline
    nlp = initialize_nlp()
    
    # Initialize components
    loader = DataLoader()
    processor = NLPProcessor(nlp, config)
    
    # Load and process documents
    docs = loader.from_file(config.input_file, text_column=config.text_column)
    note_ids = loader.from_file(config.input_file, id_column=config.id_column)
    
    # Initialize embedder if path provided
    if config.embeddings_path:
        print(f"Loading embeddings from {config.embeddings_path}")
        embedder = ConceptEmbedder(config.embeddings_path)
    else:
        print("No embeddings path provided, skipping embeddings")
        embedder = None

    # Set up dictionaries for frequency tracking if visualization is requested
    if config.visualization_path:
        concept_raw_counts = {}  # raw frequency: total occurrences across documents
        concept_doc_counts = {}  # document frequency: number of documents where the concept appears
        doc_counter = 0
    else:
        concept_raw_counts = None
        concept_doc_counts = None

    # Process and write results
    with open(config.output_file, 'w') as outfile:
        for result in processor.process_documents(note_ids, docs):
            if concept_raw_counts is not None:
                doc_counter += 1
                unique_cuis = set()
            if embedder:
                # Add embeddings to the result only if embedder exists
                result['embeddings'] = embedder.embed_document(result['umls'])
            
            # Update concept frequencies
            if concept_raw_counts is not None:
                for entity in result['umls']:
                    cui = entity['cui']
                    concept_raw_counts[cui] = concept_raw_counts.get(cui, 0) + 1
                    unique_cuis.add(cui)
                # For document frequency, count each concept only once per document
                for cui in unique_cuis:
                    concept_doc_counts[cui] = concept_doc_counts.get(cui, 0) + 1
            
            outfile.write(json.dumps(result) + '\n')
    
    # Create visualization if requested
    if config.visualization_path and embedder and concept_raw_counts is not None:
        try:
            import math
            weights = {}
            # Calculate a tfidf-like weight for each concept
            for cui, raw_count in concept_raw_counts.items():
                df_count = concept_doc_counts.get(cui, 0)
                # Avoid division by zero; if a concept appears in every document, the weight becomes very low
                if df_count > 0 and doc_counter > 0:
                    weights[cui] = raw_count * math.log(doc_counter / df_count)
                else:
                    weights[cui] = 0
            
            visualizer = ConceptVisualizer()
            df = visualizer.prepare_data(
                embedder.get_all_embeddings(),
                frequency_dict=concept_raw_counts,
                weight_dict=weights
            )
            fig = visualizer.create_plot(df)
            visualizer.save_plot(fig, config.visualization_path)
        except Exception as e:
            print(f"Error creating visualization: {e}")

if __name__ == "__main__":
	main()