import argparse
import medspacy
import json
from pathlib import Path
from utils import ProcessingConfig, DataLoader
from processor import NLPProcessor

MEDSPACY_PIPES = ['medspacy_pyrush', 'medspacy_context']

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
	
	args = parser.parse_args()
	
	return ProcessingConfig(
		input_file=args.input,
		output_file=args.output,
		umls_path=args.umls,
		batch_size=args.batch_size,
		num_workers=args.workers,
		parallelize=args.parallel
	)

def main() -> None:

	config = parse_args()
	
	# Initialize NLP pipeline
	nlp = medspacy.load(enable=MEDSPACY_PIPES)
	
	# Initialize components
	loader = DataLoader()
	processor = NLPProcessor(nlp, config)
	
	# Load and process documents
	docs = loader.from_file(config.input_file, text_column=config.text_column)
	note_ids = loader.from_file(config.input_file, id_column=config.id_column)
	
	# Process and write results
	with open(config.output_file, 'w') as outfile:
		for result in processor.process_documents(note_ids, docs):
			outfile.write(json.dumps(result) + '\n')

if __name__ == "__main__":
	main()