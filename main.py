import argparse
import spacy
import json
from pathlib import Path
from utils import ProcessingConfig, DataLoader
from processor import NLPProcessor
from embeddings import ConceptEmbedder
from visualizer import ConceptVisualizer
from document_embedder import DocumentEmbedder
import os
import logging
from collections import defaultdict
import math

# S3 utilities for modular data movement
try:
    from s3_utils import download_from_s3, upload_to_s3
except ImportError:
    download_from_s3 = upload_to_s3 = None

def initialize_nlp():
    """Initialize the NLP pipeline."""
    try:
        # Use basic spaCy model instead of medspacy
        nlp = spacy.load("en_core_web_sm")
        
        # Verify the pipes were added
        logging.info(f"Active pipes: {nlp.pipe_names}")
        
        return nlp
    except Exception as e:
        logging.error(f"Error initializing NLP pipeline: {e}")
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
	parser.add_argument('--sbert-model',
					   choices=['sapbert', 'minilm'],
					   default='sapbert',
					   help='SentenceTransformer model to use')
	parser.add_argument('--fusion',
					   choices=['concat', 'linear'],
					   default='concat',
					   help='Fusion strategy for embeddings')
	parser.add_argument('--fallback',
					   choices=['text2vec', 'graph'],
					   default='text2vec',
					   help='Fallback strategy for missing CUI embeddings')
	parser.add_argument('--mrrel',
					   help='Path to MRREL.RRF for graph fallback')
	parser.add_argument('--vectors-out',
					   help='Path to save document vectors as .npy (and meta csv)')
	
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
		visualization_path=args.visualize,
		sbert_model=args.sbert_model,
		fusion_strategy=args.fusion,
		fallback_strategy=args.fallback,
		mrrel_path=args.mrrel,
		vectors_out=args.vectors_out
	)

def main() -> None:
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
        download_from_s3(s3_bucket, input_prefix, input_dir)
        # Update config.input_file to point to the downloaded file (assume single file for now)
        files = [f for f in input_dir.iterdir() if f.suffix in ['.csv', '.txt']]
        if files:
            config.input_file = str(files[0])
        else:
            raise RuntimeError(f"No input file found in {input_dir} after S3 download.")
        # Set output file to output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_file = str(output_dir / 'output.jsonl')

    # Initialize NLP pipeline
    nlp = initialize_nlp()
    
    # Initialize components
    loader = DataLoader()
    processor = NLPProcessor(nlp, config)
    
    # Load and process documents
    docs = loader.from_file(config.input_file, text_column=config.text_column)
    note_ids = loader.from_file(config.input_file, id_column=config.id_column)
    
    # -------- frequency dictionaries (always, for SIF) --------
    concept_raw_counts: dict[str, int] = defaultdict(int)
    concept_doc_counts: dict[str, int] = defaultdict(int)
    doc_counter = 0

    # -------- Pass 1: gather concept stats & cache docs --------
    doc_cache = []  # cache results so we don't rerun NLP in second pass

    for result in processor.process_documents(note_ids, docs):
        if concept_raw_counts is not None:
            doc_counter += 1
            unique_cuis = set()

        # Record frequencies only (no embedding yet)
        if concept_raw_counts is not None:
            for entity in result['umls']:
                cui = entity['cui']
                concept_raw_counts[cui] = concept_raw_counts.get(cui, 0) + 1
                unique_cuis.add(cui)
            for cui in unique_cuis:
                concept_doc_counts[cui] = concept_doc_counts.get(cui, 0) + 1

        doc_cache.append(result)

    # -------- Build IDF dictionary --------
    idf_dict = {}
    if doc_counter > 0:
        for cui, df_count in concept_doc_counts.items():
            idf_dict[cui] = math.log(doc_counter / df_count)

    # Initialize embedder if path provided
    if config.embeddings_path:
        logging.info(f"Loading embeddings from {config.embeddings_path}")
        embedder = ConceptEmbedder(
            config.embeddings_path,
            fallback_strategy=config.fallback_strategy,
            sbert_model=config.sbert_model,
            mrrel_path=config.mrrel_path
        )
    else:
        logging.info("No embeddings path provided, skipping embeddings")
        embedder = None

    if embedder is not None:
        doc_embedder = DocumentEmbedder(
            embedder,
            sbert_model=config.sbert_model,
            fusion_strategy=config.fusion_strategy,
        )
    else:
        doc_embedder = None

    # Attach IDF weights to embedder
    if doc_embedder is not None:
        doc_embedder.set_idf_weights(idf_dict)

    # Prepare persistence if requested
    vectors_out_path = config.vectors_out
    meta_rows = []
    doc_vecs = []

    # -------- Pass 2: generate embeddings & collect vectors --------
    for result in doc_cache:
        if doc_embedder:
            vec = doc_embedder.embed_document(result['umls'], result['text'])
            if vectors_out_path:
                doc_vecs.append(vec)
                meta_rows.append({'row_id': result['row_id']})
            result['doc_vector'] = vec  # keep as numpy for now
        elif embedder:
            result['embeddings'] = embedder.embed_document(result['umls'])

    # -------- Principal component removal on doc vectors --------
    if doc_vecs:
        import numpy as np, pandas as pd
        vec_arr = np.stack(doc_vecs)  # (n_docs, dim)
        # SVD to get first PC
        try:
            u, s, vh = np.linalg.svd(vec_arr, full_matrices=False)
            pc = vh[0]  # (dim,)
            vec_arr_deflated = vec_arr - (vec_arr @ pc[:, None]) * pc[None, :]
            doc_vecs = [v for v in vec_arr_deflated]
        except Exception as e:
            logging.warning(f"PC removal failed: {e}; continuing without deflation")
            vec_arr_deflated = vec_arr

        # Update cached results with deflated vectors (keep list aligned)
        idx = 0
        for res in doc_cache:
            if 'doc_vector' in res:
                res['doc_vector'] = vec_arr_deflated[idx].tolist()
                idx += 1

        # Save vectors if requested
        if vectors_out_path:
            npy_path = vectors_out_path if vectors_out_path.endswith('.npy') else f"{vectors_out_path}.npy"
            csv_path = vectors_out_path.replace('.npy', '') + "_meta.csv"
            np.save(npy_path, vec_arr_deflated)
            pd.DataFrame(meta_rows).to_csv(csv_path, index=False)
            logging.info(f"Saved document vectors to {npy_path} and metadata to {csv_path}")

    # -------- Write JSONL output --------
    with open(config.output_file, 'w') as outfile:
        for res in doc_cache:
            outfile.write(json.dumps(res) + '\n')

    # Benchmark summary
    if embedder is not None:
        logging.info("Embedding coverage stats:")
        for k, v in embedder.stats.items():
            logging.info(f"  {k}: {v}")

    # Create visualization if requested
    if config.visualization_path and embedder and concept_raw_counts is not None:
        try:
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
                embedder.embeddings,
                frequency_dict=concept_raw_counts,
                weight_dict=weights
            )
            fig = visualizer.create_plot(df)
            visualizer.save_plot(fig, config.visualization_path)
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")

    # After processing, if in Fargate mode, upload output to S3
    if mode == 'fargate' and upload_to_s3 and s3_bucket:
        logging.info(f"[Fargate] Uploading output from {output_dir} to s3://{s3_bucket}/{output_prefix}")
        upload_to_s3(s3_bucket, output_prefix, output_dir)

if __name__ == "__main__":
	main()