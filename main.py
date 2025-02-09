import argparse
import spacy
import json
from pathlib import Path
from utils import ProcessingConfig, DataLoader
from processor import NLPProcessor
from embeddings import ConceptEmbedder
from visualizer import ConceptVisualizer


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
	
	# Track concept frequencies if visualization is requested
	concept_frequencies = {} if config.visualization_path else None
	
	# Process and write results
	with open(config.output_file, 'w') as outfile:
		for result in processor.process_documents(note_ids, docs):
			if embedder:
				# Add embeddings to the result only if embedder exists
				result['embeddings'] = embedder.embed_document(result['umls'])
				
			# Update concept frequencies if tracking
			if concept_frequencies is not None:
				for entity in result['umls']:
					cui = entity['cui']
					concept_frequencies[cui] = concept_frequencies.get(cui, 0) + 1
			
			outfile.write(json.dumps(result) + '\n')
	
	# Create visualization if requested
	if config.visualization_path and embedder:
		try:
			visualizer = ConceptVisualizer()
			df = visualizer.prepare_data(
				embedder.get_all_embeddings(),
				concept_frequencies
			)
			fig = visualizer.create_plot(df)
			visualizer.save_plot(fig, config.visualization_path)
		except Exception as e:
			print(f"Error creating visualization: {e}")

if __name__ == "__main__":
	main()