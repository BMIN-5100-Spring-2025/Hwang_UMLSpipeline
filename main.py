import argparse
import spacy
import json
import base64
from pathlib import Path
from utils import ProcessingConfig, DataLoader
from processor import NLPProcessor
from embeddings import ConceptEmbedder
from visualizer import ConceptVisualizer
from document_embedder import DocumentEmbedder
import os
import logging
from collections import defaultdict
import re
from tqdm import tqdm
import uuid
import numpy as np
import pandas as pd 

try:
    from s3_utils import download_from_s3, upload_to_s3, download_single_file
except ImportError:
    download_from_s3 = upload_to_s3 = download_single_file = None
    

def initialize_nlp():
    """Initialize the NLP pipeline."""
    try:
        # Disable unused pipes for speed
        disable = ["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        nlp = spacy.load("en_core_web_sm", disable=disable)
        nlp.max_length = 2_000_000

        # Ensure sentence boundaries are detected if needed elsewhere
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

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
					   default='minilm',
					   help='SentenceTransformer model to use (HuggingFace ID or local path)')
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
	parser.add_argument('--viz-dimension',
					   type=int,
					   default=2,
					   choices=[2, 3],
					   help='Output dimension for visualization (2 or 3)')
	parser.add_argument('--cluster-method',
					   choices=['hdbscan', 'kmeans', 'gmm', 'spectral'],
					   help='Clustering method to use')
	parser.add_argument('--n-clusters',
					   type=int,
					   help='Number of clusters for KMeans clustering')
	parser.add_argument('--hdb-min-cluster-size', type=int, help='HDBSCAN min_cluster_size')
	parser.add_argument('--hdb-min-samples', type=int, help='HDBSCAN min_samples')
	
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
		viz_dimension=args.viz_dimension,
		sbert_model=args.sbert_model,
		fusion_strategy=args.fusion,
		fallback_strategy=args.fallback,
		mrrel_path=args.mrrel,
		vectors_out=args.vectors_out,
		cluster_method=args.cluster_method,
		n_clusters=args.n_clusters,
		hdb_min_cluster_size=args.hdb_min_cluster_size,
		hdb_min_samples=args.hdb_min_samples
	)


def make_safe_s3_key_py(filename):
    safe_name = re.sub(r'[^a-z0-9]', '_', filename, flags=re.IGNORECASE)
    return safe_name.lower()

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    config = parse_args()
    
    # Modular S3 logic
    mode = os.environ.get('MODE', 'local').lower()
    s3_bucket = os.environ.get('S3_BUCKET')
    session_id = os.environ.get('SESSION_ID')
    
    # Get Parameters
    params_encoded = os.environ.get('PARAMETERS')
    params = None
    if params_encoded:
        try:
            params_json = base64.b64decode(params_encoded).decode('utf-8')
            params = json.loads(params_json)
            logging.info(f"Decoded parameters: {params}")
        except Exception as e:
            logging.error(f"Failed to decode/parse PARAMETERS environment variable: {e}")
            raise RuntimeError(f"Could not process PARAMETERS: {e}")
    
    # Define S3 Prefixes based on Session ID
    if not session_id:
        session_id = str(uuid.uuid4())
        logging.warning(f"SESSION_ID not found in environment, generated new one: {session_id}")
    
    # Define prefixes using the confirmed session_id
    # input_prefix is now derived within the Fargate block from params
    output_prefix = f"{session_id}/output/" 
    logging.info(f"Using Session ID: {session_id}")
    logging.info(f"Using Output Prefix: {output_prefix}")
    
    input_dir = Path(os.environ.get('INPUT_DIR', '/tmp/input'))
    output_dir = Path(os.environ.get('OUTPUT_DIR', '/tmp/output'))

    # If running in Fargate mode, handle downloads
    if mode == 'fargate':
        if not (download_single_file and download_from_s3 and s3_bucket and session_id and params):
             raise RuntimeError("Fargate mode requires S3 utilities, S3_BUCKET, SESSION_ID, and PARAMETERS to be set.")

        # Download Specific Primary Input File
        input_filename_original = params.get('input_filename')
        if not input_filename_original:
            raise RuntimeError("'input_filename' not found in PARAMETERS.")
        
        # Apply the SAME safe key logic as the frontend 
        input_filename_safe = make_safe_s3_key_py(input_filename_original)
        logging.info(f"Identified target input filename from parameters: {input_filename_original}")
        logging.info(f"Applying safe key logic for S3 key, using safe name: {input_filename_safe}")
        
        # Construct S3 key using the SAFE filename
        primary_input_s3_key = f"{session_id}/input/{input_filename_safe}"
        # Construct LOCAL path using the ORIGINAL filename (with suffix)
        primary_input_local_path = input_dir / input_filename_original 
        
        input_dir.mkdir(parents=True, exist_ok=True)

        # Pass the LOCAL path to the download function
        logging.info(f"Calling download_single_file: s3://{s3_bucket}/{primary_input_s3_key} -> {primary_input_local_path}")
        success = download_single_file(s3_bucket, primary_input_s3_key, primary_input_local_path)

        if not success:
            raise RuntimeError(f"Failed to download primary input file from S3.")
            
        # Verify existence after reported success (using the original local path)
        if not primary_input_local_path.is_file():
             raise RuntimeError(
                 f"download_single_file reported success for {primary_input_local_path}, "
                 f"but the file does not exist or is not a file."
             )
             
        logging.info(f" Primary input download and validation successful: {primary_input_local_path} ")
        
        config.input_file = str(primary_input_local_path)
        logging.info(f"Config updated: input_file = {config.input_file}")
        
        # Download UMLS data
        umls_s3_prefix = "2020AB-full/" 
        local_umls_base_path = input_dir / "2020AB-full" 
        logging.info(f"[Fargate] Downloading UMLS from s3://{s3_bucket}/{umls_s3_prefix} to {local_umls_base_path}")
        local_umls_base_path.mkdir(parents=True, exist_ok=True)
        download_from_s3(s3_bucket, umls_s3_prefix, local_umls_base_path) 

        # Download Embeddings
        embeddings_s3_prefix = "data/embeddings/" 
        local_embeddings_path = input_dir / "data" / "embeddings"
        logging.info(f"[Fargate] Downloading Embeddings from s3://{s3_bucket}/{embeddings_s3_prefix} to {local_embeddings_path}")
        local_embeddings_path.mkdir(parents=True, exist_ok=True)
        download_from_s3(s3_bucket, embeddings_s3_prefix, local_embeddings_path) 
        
        # Download SentenceTransformer Model
        sbert_model_s3_prefix = "models/miniLM/" 
        local_sbert_model_path = Path("/tmp/models/miniLM")
        logging.info(f"[Fargate] Downloading SentenceTransformer model from s3://{s3_bucket}/{sbert_model_s3_prefix} to {local_sbert_model_path}")
        local_sbert_model_path.mkdir(parents=True, exist_ok=True)
        download_from_s3(s3_bucket, sbert_model_s3_prefix, local_sbert_model_path)
        if not any(local_sbert_model_path.iterdir()):
            logging.warning(f"SentenceTransformer model download from S3 seemed successful, but target directory {local_sbert_model_path} is empty.")
        fargate_sbert_model_path = str(local_sbert_model_path)

        # Update Config Paths and Params from Environment/Downloads
        config.output_file = str(output_dir / 'output.jsonl')
        config.umls_path = str(local_umls_base_path / "2020AB-quickumls-install")
        config.embeddings_path = str(local_embeddings_path / "cui2vec_pretrained.txt")
        config.sbert_model = fargate_sbert_model_path
        
        # Check for MRREL path
        mrrel_local_path = local_umls_base_path / "MRREL.RRF"
        if mrrel_local_path.exists():
             config.mrrel_path = str(mrrel_local_path) 
        else:
             logging.warning(f"MRREL.RRF not found at {mrrel_local_path}, mrrel_path not set in config.")
             config.mrrel_path = None 

        # Override clustering params from decoded environment params
        if params:
            if 'cluster_method' in params:
                 config.cluster_method = params['cluster_method']
                 logging.info(f"Overriding cluster_method from params: {config.cluster_method}")
            if 'n_clusters' in params:
                 try: 
                     config.n_clusters = int(params['n_clusters'])
                     logging.info(f"Overriding n_clusters from params: {config.n_clusters}")
                 except (ValueError, TypeError):
                      logging.warning(f"Could not parse n_clusters '{params['n_clusters']}' from params as integer. Using default/CLI value.")
            if 'viz_dimension' in params:
                 try:
                      dim = int(params['viz_dimension'])
                      if dim in [2, 3]:
                           config.viz_dimension = dim
                           logging.info(f"Overriding viz_dimension from params: {config.viz_dimension}")
                      else:
                           logging.warning(f"Invalid viz_dimension '{dim}' in params. Must be 2 or 3. Using default/CLI value.")
                 except (ValueError, TypeError):
                      logging.warning(f"Could not parse viz_dimension '{params['viz_dimension']}' from params as integer. Using default/CLI value.")
            if 'hdb_min_cluster_size' in params:
                 try:
                      config.hdb_min_cluster_size = int(params['hdb_min_cluster_size'])
                      logging.info(f"Overriding hdb_min_cluster_size from params: {config.hdb_min_cluster_size}")
                 except (ValueError, TypeError):
                      logging.warning(f"Could not parse hdb_min_cluster_size '{params['hdb_min_cluster_size']}' as integer.")
            if 'hdb_min_samples' in params:
                 try:
                      config.hdb_min_samples = int(params['hdb_min_samples'])
                      logging.info(f"Overriding hdb_min_samples from params: {config.hdb_min_samples}")
                 except (ValueError, TypeError):
                      logging.warning(f"Could not parse hdb_min_samples '{params['hdb_min_samples']}' as integer.")
                          
        logging.info("--- Fargate Config Overrides Applied ---")
        logging.info(f"Using Output File: {config.output_file}")
        logging.info(f"Using UMLS Path: {config.umls_path}")
        logging.info(f"Using Embeddings Path: {config.embeddings_path}")
        logging.info(f"Using MRREL Path: {config.mrrel_path}")
        logging.info(f"Using SBERT Model Path: {config.sbert_model}")
        logging.info(f"Using Cluster Method: {config.cluster_method}")
        logging.info(f"Using N Clusters: {config.n_clusters}")
        logging.info(f"Using Viz Dimension: {config.viz_dimension}")
    
    # Initialize NLP pipeline
    logging.info("Initializing spaCy NLP pipeline...")
    nlp = initialize_nlp()
    logging.info("spaCy NLP pipeline initialized.")
    
    # Initialize components
    logging.info("Initializing DataLoader and NLPProcessor...")
    loader = DataLoader()
    processor = NLPProcessor(nlp, config)
    logging.info("DataLoader and NLPProcessor initialized.")
    
    # Load documents
    logging.info(f"Loading documents from {config.input_file}...")
    docs = loader.from_file(config.input_file, text_column=config.text_column)
    note_ids = loader.from_file(config.input_file, id_column=config.id_column)
    logging.info(f"Documents loaded.")
    
    # frequency dictionaries
    logging.info("Initializing frequency dictionaries...")
    concept_raw_counts: dict[str, int] = defaultdict(int)
    concept_doc_counts: dict[str, int] = defaultdict(int)
    doc_counter = 0

    # Pass 1: gather concept stats & cache docs
    logging.info("Starting Pass 1: Gathering concept stats and caching docs...")
    doc_cache = [] 
    pass1_counter = 0
    log_interval = 100 

    for result in processor.process_documents(note_ids, docs):
        pass1_counter += 1
        if concept_raw_counts is not None:
            doc_counter += 1
            unique_cuis = set()
            for entity in result['umls']:
                cui = entity['cui']
                concept_raw_counts[cui] = concept_raw_counts.get(cui, 0) + 1
                unique_cuis.add(cui)
            for cui in unique_cuis:
                concept_doc_counts[cui] = concept_doc_counts.get(cui, 0) + 1
        doc_cache.append(result)
        if pass1_counter % log_interval == 0:
            logging.info(f"  Pass 1: Processed {pass1_counter} documents...")

    logging.info(f"Finished Pass 1. Processed {pass1_counter} documents total. Cached {len(doc_cache)} results.")

    # Build IDF dictionary
    logging.info("Building document frequency dictionary…")
    df_ratio = {}
    if doc_counter > 0:
        for cui, df_count in concept_doc_counts.items():
            df_ratio[cui] = df_count / doc_counter
    logging.info(f"Computed DF ratios for {len(df_ratio)} CUIs.")
    
 
    logging.info("Initializing Embedders...")
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


    if doc_embedder is not None:
        logging.info("Attaching IDF weights...")
        doc_embedder.set_idf_weights(df_ratio)
        sif_a = doc_embedder.sif_a
    else:
        sif_a = 1e-3
    logging.info("Embedders initialized.")

    # Precompute SIF weights
    sif_lookup = {}
    if doc_embedder:
        sif_lookup = {cui: sif_a / (sif_a + df)
                      for cui, df in df_ratio.items()}
        logging.info(f"Precomputed SIF weights for {len(sif_lookup)} CUIs.")

    vectors_out_path = config.vectors_out
    meta_rows = []
    doc_vecs = []
    note_hover_meta = []

    # Pass 2 Setup: Batch SBERT Encoding
    sbert_vecs = None
    if doc_embedder:
        texts = [res['text'] for res in doc_cache]
        logging.info(f"Encoding {len(texts)} documents with SBERT (batch_size={config.batch_size}). Using multi-process for CPU.")
        sbert_model = doc_embedder.sentence_embedder._model
        
        # Use encode_multi_process for CPU parallelization
        pool = sbert_model.start_multi_process_pool()
        sbert_vecs = sbert_model.encode_multi_process(
            texts,
            pool,
            batch_size=config.batch_size,
        )
        sbert_model.stop_multi_process_pool(pool)
        
        sbert_vecs = np.array(sbert_vecs) 

        logging.info("SBERT encoding complete.")

    # Pass 2: generate embeddings & collect vectors
    logging.info("Starting Pass 2: Generating embeddings, collecting vectors, and hover meta...")
    for i, result in enumerate(tqdm(doc_cache, desc="Pass 2: Combining Embeddings", total=len(doc_cache), unit="doc")):
        if doc_embedder and sbert_vecs is not None:
            # 1. Get CUI embeddings (including pooling)
            cui_to_vec = embedder.embed_document(result['umls'])
            pooled_cui_vec = doc_embedder._aggregate_cui_vectors(cui_to_vec)

            # 2. Get pre-computed SBERT vector
            current_sbert_vec = sbert_vecs[i]

            # 3. Combine
            vec = doc_embedder.combiner.combine(pooled_cui_vec, current_sbert_vec)

            if vectors_out_path:
                doc_vecs.append(vec)
                meta_rows.append({'row_id': result['row_id']})
            result['doc_vector'] = vec

        # If ConceptEmbedder runs (no DocumentEmbedder/SBERT)
        elif embedder: 
            result['embeddings'] = embedder.embed_document(result['umls'])


        umls_entities = result.get('umls', [])
        cui_to_term_map = {ent['cui']: (ent.get('raw') or ent.get('text', ent['cui']))
                           for ent in umls_entities}
        
        top_str = ""
        if umls_entities:
            cuis_in_doc = [ent['cui'] for ent in umls_entities]
            unique_cuis, counts = np.unique(cuis_in_doc, return_counts=True)
            
            sif_weights = np.array([sif_lookup.get(cui, sif_a / (sif_a + 1e-9)) 
                                    for cui in unique_cuis])
                                    
            combined_scores = counts * sif_weights
            
            num_top = min(5, len(unique_cuis))
            top_indices = np.argsort(combined_scores)[::-1][:num_top]
            
            top_cuis = unique_cuis[top_indices]
            top_scores = combined_scores[top_indices]
            
            # Build hover string with unique terms 
            hover_items = []
            added_terms = set()
            for cui, score in zip(top_cuis, top_scores):
                term = cui_to_term_map.get(cui, cui)
                if term not in added_terms:
                    hover_items.append(f"{term} ({score:.2f})")
                    added_terms.add(term)
            top_str = "; ".join(hover_items)

        note_hover_meta.append({"row_id": result['row_id'], "top_concepts": top_str})

    logging.info(f"Finished Pass 2. Processed {len(doc_cache)} documents total.")

    # All but the Top
    if doc_vecs:
        logging.info("Starting Principal Component Removal...")
        vec_arr = np.stack(doc_vecs)  
        try:
            u, s, vh = np.linalg.svd(vec_arr, full_matrices=False)
            pc = vh[0]
            vec_arr_deflated = vec_arr - (vec_arr @ pc[:, None]) * pc[None, :]
            doc_vecs = [v for v in vec_arr_deflated]
            
            # Calculate and Log Isotropy
            total_variance = np.sum(s**2)
            eigenvalue_spectrum_percent = None
            if total_variance > 1e-9:
                isotropy_score = 1.0 - (s[0]**2 / total_variance)
                logging.info(f"Isotropy score (1 - λ1/Σλ): {isotropy_score:.4f}")
                # Add Eigenvalue Spectrum
                eigenvalues_normalized = (s**2 / total_variance).tolist()
                num_to_keep = min(50, len(eigenvalues_normalized))
                eigenvalue_spectrum_percent = eigenvalues_normalized[:num_to_keep]
            else:
                isotropy_score = None
                logging.warning("Total variance of singular values is near zero, cannot calculate isotropy.")
            
        except Exception as e:
            logging.warning(f"PC removal or Isotropy calculation failed: {e}; continuing without deflation")
            vec_arr_deflated = vec_arr
            isotropy_score = None

        # Update cached results with deflated vectors
        idx = 0
        for res in doc_cache:
            if 'doc_vector' in res:
                res['doc_vector'] = vec_arr_deflated[idx].tolist() 
                idx += 1

        # Save vectors
        if vectors_out_path:
            npy_path = vectors_out_path if vectors_out_path.endswith('.npy') else f"{vectors_out_path}.npy"
            csv_path = vectors_out_path.replace('.npy', '') + "_meta.csv"
            output_save_dir = Path(npy_path).parent
            output_save_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(npy_path, vec_arr_deflated)
            pd.DataFrame(meta_rows).to_csv(csv_path, index=False)
            logging.info(f"Saved document vectors to {npy_path} and metadata to {csv_path}")

        final_vec_arr = vec_arr_deflated
    elif doc_cache:
        try:
            final_vec_arr = np.stack([res['doc_vector'] for res in doc_cache if 'doc_vector' in res])
        except ValueError:
            logging.error("Could not stack document vectors for clustering. Skipping clustering.")
            final_vec_arr = None 
    else:
        final_vec_arr = None

    # Clustering
    cluster_labels = None
    if final_vec_arr is not None and final_vec_arr.shape[0] > 1:
        logging.info(f"Starting clustering using method: {config.cluster_method}")
        try:
            if config.cluster_method == 'hdbscan':
                import hdbscan
                    
                min_cs = config.hdb_min_cluster_size or 5
                hdb_kwargs = {
                    'min_cluster_size': min_cs,
                    'metric': 'euclidean',
                    'allow_single_cluster': True,
                }
                if config.hdb_min_samples is not None:
                    hdb_kwargs['min_samples'] = config.hdb_min_samples
                logging.info(f"Running HDBSCAN with params: {hdb_kwargs}")
                clusterer = hdbscan.HDBSCAN(**hdb_kwargs)
                cluster_labels = clusterer.fit_predict(final_vec_arr)
                logging.info(f"HDBSCAN finished. Found {len(set(cluster_labels) - {-1})} clusters (plus noise points labeled -1).")
            elif config.cluster_method == 'kmeans':
                from sklearn.cluster import KMeans
                k = min(config.n_clusters or 10, final_vec_arr.shape[0])
                logging.info(f"Running KMeans with k={k}")
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(final_vec_arr)
                logging.info(f"KMeans finished.")
            elif config.cluster_method == 'gmm':
                from sklearn.mixture import GaussianMixture
                k = min(config.n_clusters or 10, final_vec_arr.shape[0])
                logging.info(f"Running GaussianMixture with n_components={k}")
                gmm = GaussianMixture(n_components=k, random_state=42)
                cluster_labels = gmm.fit_predict(final_vec_arr)
                logging.info(f"GaussianMixture finished.")
            elif config.cluster_method == 'spectral':
                from sklearn.cluster import SpectralClustering
                k = min(config.n_clusters or 10, final_vec_arr.shape[0])
                logging.info(f"Running SpectralClustering with n_clusters={k}")
                spectral = SpectralClustering(n_clusters=k, random_state=42, assign_labels='kmeans')
                cluster_labels = spectral.fit_predict(final_vec_arr)
                logging.info(f"SpectralClustering finished.")
            else:
                logging.warning(f"Unknown clustering method '{config.cluster_method}', skipping clustering")
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            cluster_labels = None
    else:
        logging.warning("Not enough document vectors to perform clustering. Skipping.")
        
    # Add cluster labels to doc_cache if clustering was successful
    if cluster_labels is not None:
        if len(cluster_labels) == len(doc_cache):
            for i, res in enumerate(doc_cache):
                res['cluster'] = int(cluster_labels[i])
        else:
             logging.error("Cluster label count mismatch with doc cache count. Skipping cluster assignment.")
             cluster_labels = None


    # Update cached results with deflated vectors if PC removal ran
    if 'vec_arr_deflated' in locals():
        idx = 0
        for res in doc_cache:
            if 'doc_vector' in res:
                res['doc_vector'] = vec_arr_deflated[idx].tolist()
                idx += 1
        logging.info("Updated doc_cache with PC-removed vectors.")
    elif final_vec_arr is not None:
         for res in doc_cache:
             if 'doc_vector' in res and isinstance(res['doc_vector'], np.ndarray):
                  res['doc_vector'] = res['doc_vector'].tolist()

    # Write JSONL output
    logging.info(f"Writing JSONL output to {config.output_file}...")
    with open(config.output_file, 'w') as outfile:
        for res in doc_cache:
            outfile.write(json.dumps(res) + '\n')
    logging.info(f"Finished writing JSONL output.")

    # Save Metrics
    metrics = {
        'num_documents': len(doc_cache),
        'isotropy_score': isotropy_score, 
        'eigenvalue_spectrum_percent': eigenvalue_spectrum_percent
    }
    metrics_path = output_dir / 'metrics.json'
    try:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f_metrics:
            json.dump(metrics, f_metrics, indent=2)
        logging.info(f"Saved metrics to {metrics_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics to {metrics_path}: {e}")

    # Benchmark summary
    if embedder is not None:
        logging.info("Embedding coverage stats:")
        for k, v in embedder.stats.items():
            logging.info(f"  {k}: {v}")
    
    # Create visualization if requested
    if config.visualization_path and doc_embedder and note_hover_meta:
        logging.info(f"Creating {config.viz_dimension}D visualization and saving to {config.visualization_path}...")
        try:
            visualizer = ConceptVisualizer()

            # Use final_vec_arr for visualization if available
            vec_array_for_viz = final_vec_arr if final_vec_arr is not None else np.array([])
            
            # Extract cluster labels if they exist
            viz_cluster_labels = [res.get('cluster') for res in doc_cache] if cluster_labels is not None else None
            
            if vec_array_for_viz.shape[0] > 0:
                vis_df = visualizer.prepare_note_data(
                    vec_array_for_viz,
                    note_ids=[m["row_id"] for m in note_hover_meta],
                    top_strings=[m["top_concepts"] for m in note_hover_meta],
                    cluster_labels=viz_cluster_labels,
                    dimensions=config.viz_dimension
                )
                
                fig = visualizer.create_plot(
                    vis_df, 
                    title="Note Map Visualization", 
                    dimensions=config.viz_dimension
                )
                visualizer.save_plot(fig, config.visualization_path)
                logging.info(f"Visualization saved.")
            else:
                logging.warning("No vectors available for visualization. Skipping plot generation.")

        except Exception as e:
            logging.error(f"Error creating visualization: {e}")

    # if in Fargate mode, upload output to S3
    if mode == 'fargate' and upload_to_s3 and s3_bucket:
        logging.info(f"[Fargate] Uploading output from {output_dir} to s3://{s3_bucket}/{output_prefix}...")
        upload_to_s3(s3_bucket, output_prefix, output_dir)
        logging.info(f"[Fargate] Output upload complete.")

    logging.info("Main script finished successfully.")

if __name__ == "__main__":
	main()